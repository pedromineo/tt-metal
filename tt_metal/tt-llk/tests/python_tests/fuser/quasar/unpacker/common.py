# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from fuser.fpu_node import FpuNode
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestSync

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


def hw_configure_unpack(
    compute_node: FpuNode,
    dest_acc: str,
    unpack_A_src: DataFormat,
    unpack_A_dst: DataFormat,
    unpack_B_src: DataFormat,
    unpack_B_dst: DataFormat,
) -> str:
    return (
        f"_llk_unpack_hw_configure_<p_unpacr::UNP_A>(static_cast<DataFormat>({unpack_A_dst.cpp_underlying_value}));\n"
        f"_llk_unpack_hw_configure_<p_unpacr::UNP_B>(static_cast<DataFormat>({unpack_B_dst.cpp_underlying_value}));\n"
    )


def configure_unpack(
    compute_node: FpuNode,
    dest_acc: str,
    old_A_src: DataFormat,
    new_A_src: DataFormat,
    new_A_dst: DataFormat,
    old_B_src: DataFormat,
    new_B_src: DataFormat,
    new_B_dst: DataFormat,
    srca_changed: bool,
    srcb_changed: bool,
    srca_tile_changed: bool,
    srcb_tile_changed: bool,
) -> str:
    if srca_changed and srcb_changed:
        return f"_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(static_cast<DataFormat>({new_A_dst.cpp_underlying_value}), static_cast<DataFormat>({new_B_dst.cpp_underlying_value}));\n"
    if srca_changed:
        return f"_llk_unpack_configure_unary_<p_unpacr::UNP_A>(static_cast<DataFormat>({new_A_dst.cpp_underlying_value}));\n"
    if srcb_changed:
        return f"_llk_unpack_configure_unary_<p_unpacr::UNP_B>(static_cast<DataFormat>({new_B_dst.cpp_underlying_value}));\n"
    return ""


def dvalid_init(config: "GlobalConfig" = None, operation: "L1Operation" = None) -> str:
    from helpers.llk_params import PerfRunType

    if config.quasar_use_dvalid:
        if config.perf_run_type in (None, PerfRunType.L1_TO_L1):
            return "set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});\n"
        return "set_up_zero_dest_dvalid_handshake_for_unpack();\n"
    return ""


def sync_with_packer(config: "GlobalConfig", operation: "L1Operation") -> str:
    if operation.needs_pack_sync:
        return (
            "_llk_sync_wait_<p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(semaphore::PACK_UNPACK);\n"
            "_llk_sync_get_<>(semaphore::PACK_UNPACK);\n"
        )
    return ""


def configure_unpack_to_dest_sync(
    config: "GlobalConfig", pipeline: list["L1Operation"]
) -> None:
    config.quasar_unpack_to_dest = (
        config.architecture == ChipArchitecture.QUASAR
        and any(
            isinstance(node, FpuNode) and node.unpack_to_dest.value
            for operation in pipeline
            for node in operation.math_nodes
        )
    )
    if not config.quasar_unpack_to_dest:
        return
    if config.quasar_use_dvalid:
        raise ValueError("Quasar unpack_to_dest requires quasar_use_dvalid: false")
    if len({operation.dest_sync for operation in pipeline}) != 1:
        raise ValueError(
            "Quasar unpack_to_dest requires a consistent dest_sync across the pipeline"
        )
    for stage, operation in enumerate(pipeline, start=1):
        _validate_unpack_to_dest_schedule(operation, stage)


def _validate_unpack_to_dest_schedule(operation: "L1Operation", stage: int) -> None:
    math_started = False
    for index, node in enumerate(operation.math_nodes, start=1):
        if not isinstance(node, FpuNode) or node.unpacker is None:
            math_started = True
            continue
        location = f"Operation {stage}, math node {index}"
        if not node.unpack_to_dest.value:
            raise ValueError(
                f"{location}: Quasar unpack_to_dest synchronization applies to the whole pipeline; "
                "SrcA/SrcB unpacking is unsupported because math waits for the complete unpack section. "
                "Use direct-to-DEST input copies or a separate pipeline for source-register streaming."
            )
        if math_started:
            raise ValueError(
                f"{location}: unpack_to_dest copies must precede all math/SFPU work in the section; "
                "Metal's section-level protocol completes all direct writes before math starts."
            )


def uses_unpack_to_dest_sync(config: "GlobalConfig") -> bool:
    return config.quasar_unpack_to_dest and not config.skip_unpack_init


def unpack_to_dest_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    if not uses_unpack_to_dest_sync(config) or operation.stage_id != 1:
        return ""
    num_banks = 2 if operation.dest_sync == DestSync.Half else 1
    return (
        "set_up_zero_dest_dvalid_handshake_for_unpack();\n"
        "_llk_unpack_dest_init_();\n"
        f"_llk_sync_init_(semaphore::UNPACK_MATH, {num_banks}, 0);\n"
        f"_llk_sync_init_(semaphore::UNPACK_PACK, {num_banks}, 0);\n"
        "_llk_sync_barrier_raise_(semaphore::UNPACK_INIT);\n"
    )


def unpack_to_dest_wait_for_dest_available(config: "GlobalConfig") -> str:
    if not uses_unpack_to_dest_sync(config):
        return ""
    return "_llk_sync_wait_<p_stall::STALL_UNPACK, p_stall::STALL_ON_MAX>(semaphore::UNPACK_PACK);\n"


def unpack_to_dest_section_done(
    config: "GlobalConfig", operation: "L1Operation"
) -> str:
    if not uses_unpack_to_dest_sync(config):
        return ""
    code = (
        "_llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_PACK);\n"
        "_llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_MATH);\n"
    )
    if operation.dest_sync == DestSync.Half:
        code += (
            f"_llk_unpack_dest_section_advance_<{config.dest_acc.cpp_enum_value}>();\n"
        )
    return code
