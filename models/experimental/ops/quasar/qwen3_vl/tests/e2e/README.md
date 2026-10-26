# Qwen3-VL-4B e2e on Quasar

One pytest (`test_qwen3_vl_e2e.py`) runs Qwen3-VL-4B-Instruct cut down to V vision blocks, T text layers and K decode steps, and compares every stage against the same cut-down HF model on the host. Three wrapper scripts run it on WH/BH (ttsim or hardware), craq-sim and the emu-quasar-2x3 emulator with identical settings.

Ops that work on WH/BH but not on Quasar, simulator issues and model-copy bugs found along the way are logged in [QUASAR_GAPS.md](QUASAR_GAPS.md).

## Run

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) TT_METAL_RUNTIME_ROOT=$(pwd) PYTHONPATH=$(pwd):$PYTHONPATH

E2E=models/experimental/ops/quasar/qwen3_vl/tests/e2e
$E2E/run_wh_bh.sh --ttsim wh           # WH baseline on ttsim, emulator-sized grid
$E2E/run_wh_bh.sh --grid native        # WH/BH hardware, full grid
$E2E/run_craq.sh                       # craq-sim, emulator-sized grid
$E2E/run_craq.sh --grid 8x4            # craq-sim, full grid
$E2E/run_emu.sh                        # emulator (needs NNG_SOCKET_ADDR from your IRD reservation)
```

The emulator backs only 64 MiB of DRAM per channel although its descriptor says 1 GiB; larger allocations silently overwrite earlier ones, and the e2e (~1.3 GiB of weights) does not fit (QUASAR_GAPS S7, #55417).

Script defaults: `--size tiny --vision-layers 2 --text-layers 2 --decode-steps 1 --deepstack-at 0 --debug default` (`--debug fast` on craq-sim, where the watcher slows each op 150-300x).

| Flag | Meaning |
|---|---|
| `--size tiny\|demo` | tiny: 256x256 image, 64 image tokens, 78-token prompt padded to 128. demo: the graph-capture sizes (2752 image tokens, 4096 prefill). |
| `--vision-layers N`, `--text-layers N` | blocks of the 24-block vision tower / 36-layer text model to run |
| `--decode-steps K` | teacher-forced decode steps after prefill (0 = prefill only) |
| `--deepstack-at I\|real` | move the deepstack tap to vision block I (in TT and HF alike); `real` keeps 5/11/17 |
| `--kv-blocks N` | paged KV-cache blocks of 32 tokens (default: preset) |
| `--host-ops a,b\|all` | run these ops on the host (bisecting only; the run becomes DIAGNOSTIC) |
| `--disable-wa a,b` | turn off named workarounds from `op_overrides.py` |
| `--debug fast\|default\|deep` | fast: no watcher. default: watcher + LLK asserts, NoC sanitize off. deep: plus dump-all, noinline, per-RISC DPRINT, debug logging |
| `--noc-sanitize` | re-enable NoC sanitize (20-30x slower) |
| `--fp32-dest-acc` | keep fp32 dest accumulation in the Quasar config (hardware A/B; undefined on ttsim WH) |
| `--timeout S` | pytest timeout (WH/BH and emulator 14400, craq-sim 10800) |
| `--ttsim wh\|bh`, `--grid 2x3\|native`, `--config quasar\|native` | `run_wh_bh.sh` only; `--config native` runs the unmodified bf8 config as a hardware control |
| `--grid 2x3\|8x4` | `run_craq.sh` only |
| `-- <args>` | passed to pytest, e.g. `-- --qwen-dump-stages` to save golden and TT stage tensors, or `-- --qwen-allow-uncertified` to use host fallbacks not yet checked against the real op (`test_fallbacks.py`) |

Exit codes: 0 PASS, 3 DIAGNOSTIC (some ops ran on the host), anything else FAIL or error.

## Output

Each run writes `generated/qwen3_vl_quasar/<target>/<UTC time>/`:

| File | Content |
|---|---|
| `pcc.md` | verdict, first failing stage, per-stage PCC / threshold / max abs error / seconds, workaround and host-op hit counts |
| `verdict.txt` | `PASS`, `FAIL` or `DIAGNOSTIC` |
| `progress.log` | one `PRE`/`POST` line per ttnn op with shapes, memory configs and model stage. After a hang, the last `PRE` without a `POST` is the op that never finished. |
| `run.log`, `command.txt`, `env.txt`, `git.txt` | full output, exact command, environment, commit and branch commits |
| `ops.pt` | with `-- --qwen-capture-ops STAGE_PREFIX` (e.g. `vision.block0`, `text.decode`): every top-level op output of those stages. `repros/diff_ops.py RUN_A RUN_B` prints where two runs first diverge |
| `prefill_snapshot.pt` | KV cache, decode position, rope delta, page table and prefill stage tensors at the end of prefill. `-- --qwen-resume-prefill <run folder>` skips vision and prefill and decodes from it (same size, layers, grid and config only; the run notes it in `pcc.md`) |

Thresholds per stage and preset live in `thresholds.json`.

`progress.log` also records each op's program, compute and memory configs, so a failing call can be replayed in a repro. `-- --qwen-vision-only` stops after the vision tower (vision stages only), for short runs on slow targets.

## Grids

The emulator build exposes a 2x1 compute grid (`tt_metal/core_descriptors/quasar_simulation_2x3_arch.yaml`). `--grid 2x3` reproduces it elsewhere with `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE`, which sets the grid's *end* coordinate, so the value depends on where each target's compute grid starts:

| Target | Native grid | Override for 2x1 |
|---|---|---|
| ttsim WH | 8x8 | `1,0` |
| ttsim BH | 11x10 | `1,0` |
| craq-sim | 8x4 | `3,2` |
| emu-quasar-2x3 | 2x1 | none |

## ttsim setup (WH/BH baselines)

```bash
for a in wh:wormhole_b0_80_arch bh:blackhole_140_arch; do
  n="${a%%:*}"; y="${a##*:}"; d="/localdev/$USER/ttsim/sim_${n}"
  mkdir -p "$d" && cp "/localdev/$USER/ttsim/src/_out/release_${n}/libttsim.so" "$d/" \
    && cp "tt_metal/soc_descriptors/${y}.yaml" "$d/soc_descriptor.yaml"
done
```

Override locations with `QWEN_TTSIM_DIR`, `QWEN_CRAQ_SIM`, `QWEN_EMU_DIR`.

## Hangs

A hang shows up as the pytest timeout. The script prints the last lines of `progress.log`; the watcher log has per-core waypoints (default/deep profiles). On simulators `TT_METAL_OPERATION_TIMEOUT_SECONDS` is ignored (`tt_metal/llrt/llrt.cpp`, `wait_until_cores_done`). For a hard-to-localize hang, temporarily patch it locally to honor that variable when set; do not commit that change.

## Droppable commits carried on the branch

Kernel fixes not yet in main, each a separate `[droppable]` commit (details in the plan and QUASAR_GAPS.md):

| Commit | Why | Drop when |
|---|---|---|
| `[droppable] tensor_layout: skip the duplicate from_shard_spec call ...` | left over from the #57753 PR head; not in main | it is no longer needed by the e2e |
| `[droppable] slice (Quasar): write back and drop the reader's scratchpad cache lines ...` | Q17: stale DM-cache lines from the slice reader corrupt a later decode rms_norm | a general fix lands (no PR yet) |
| `[droppable] revert #57646 (DFB drain in the DataflowBuffer destructor) ...` | Q18: with the destructor drain, `paged_update_cache` and the sharded rms_norm still hang on craq-sim | those kernels are fixed (#60027) |
| `[droppable] layernorm (Quasar): keep the reduce scaler DataflowBuffer at kernel scope ...` | Q18: layer_norm hangs under the drain (draft [#62427](https://github.com/tenstorrent/tt-metal/pull/62427)) | #62427 lands |

## Baselines and run times

Wall time is the first-to-last-op span from `progress.log` (setup and weight loading add a few minutes). Tiny size, 2 vision / 2 text layers, 1 decode step, unless noted.

| Target | Grid | Verdict | Wall time | Date | Run folder (`generated/qwen3_vl_quasar/...`) |
|---|---|---|---|---|---|
| craq-sim (a928410c / staged .so md5 74c8eb33) | 2x1 | PASS (prefill logits 0.99727, decode0 0.99918) | 41 min | 2026-10-22 | `craq_2x3/20261022T134319Z` |
| craq-sim (same) | 8x4 | PASS (prefill 0.99710, decode0 0.99915) | 14 min | 2026-10-22 | `craq_8x4/20261022T142425Z` |
| craq-sim, without `quasar_rms_norm_bf16_dest` | 2x1 / 8x4 | PASS (prefill 0.99685 / 0.99687) | 42 min / 15 min | 2026-10-21 | `craq_2x3/20261021T170032Z`, `craq_8x4/20261021T175443Z` |
| craq-sim (older build) | 2x1 / 8x4 | PASS | 36 min / 12 min | 2026-10-16 | `craq_2x3/20261016T165030Z`, `craq_8x4/20261016T183306Z` |
| craq-sim (older build, 695 device ops) | 2x1 | PASS | 8 min | 2026-10-08 | `craq_2x3/20261008T161235Z` |
| emu-quasar-2x3 | 2x1 | FAIL: DRAM wraps at 64 MiB per channel (S7), logits NaN | 165 min (2 h 47 min wall) | 2026-10-08 | `emu_2x3/20261008T212107Z` |
| emu-quasar-2x3, vision only, V=1 | 2x1 | FAIL (S7) | 13-43 min | 2026-10-09 | `emu_2x3/20261009T0*` |
| ttsim WH | 2x1 / 8x8 / 8x4 | PASS | 10 / 9 / 10 min | 2026-10-16 | `wh_bh_wh_2x3_quasar/20261016T180210Z`, `wh_bh_wh_native_quasar/20261016T18*` |
| WH N150 silicon | 8x9 | PASS (all stages >= 0.9993), with the #59510 fix | — | 2026-10-06 | aus-wh-08 `wh_bh_hw_native_quasar/20261006T182734Z` |

The 2026-10-08 craq-sim run logged 695 device ops against 1212 from 2026-10-16 on (more of the model runs on device now), so its 8 min is not comparable with today's 41 min. Blackhole (ttsim and hardware) is not part of the baseline.
