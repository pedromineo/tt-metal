// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DevicePublish's call sites (api/debug/device_publish.h).
//
// The metadata never reaches the device. The device compiler copies it, the source location and the type names into
// sections the device never loads, next to one entry per call site (.tt_record_meta). The call site's id is the
// address of a one-byte handle in .tt_record_ids, which the host rebases into the process's id space when it loads
// the ELF (llrt/call_sites.cpp).
#pragma once

#include <algorithm>
#include <limits>
#include <tuple>
#include <type_traits>

#include "hostdev/record_site.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/risc_attribs.h"

namespace tt::tt_metal::experimental::streaming_profiler {
/** @brief The DevicePublish option that makes a record visible to the host as soon as it is written. */
inline constexpr struct PublishImmediately {
} kPublishImmediately{};
/** @brief The DevicePublish option that writes the record out of line even when the profiler is inlined, so the call
 * site costs only a call. */
inline constexpr struct NoInline {
} kNoInline{};
}  // namespace tt::tt_metal::experimental::streaming_profiler

namespace kernel_profiler {
namespace sp = tt::tt_metal::experimental::streaming_profiler;
namespace spd = sp::detail;
}  // namespace kernel_profiler

namespace kernel_profiler::records {

template <typename... Options>
constexpr bool check_options() {
    constexpr bool known =
        ((std::is_same_v<std::remove_cv_t<Options>, sp::PublishImmediately> ||
          std::is_same_v<std::remove_cv_t<Options>, sp::NoInline>) &&
         ...);
    static_assert(known, "unknown DevicePublish option; the options are kPublishImmediately and kNoInline");
    return known;
}
template <typename Option, auto... Options>
inline constexpr bool kHasOption = (std::is_same_v<std::remove_cv_t<decltype(Options)>, Option> || ...);

// The type a call site describes an argument as: its own, or a std::array for a C array.
template <typename Argument>
struct ArgumentValue {
    using type = std::remove_cv_t<Argument>;
};
template <typename Element, size_t N>
struct ArgumentValue<Element[N]> {
    using type = std::array<std::remove_cv_t<Element>, N>;
};
template <typename Arg>
using ValueOf = typename ArgumentValue<std::remove_reference_t<Arg>>::type;
template <typename Arg>
inline constexpr bool is_text_literal = std::is_array_v<std::remove_reference_t<Arg>> &&
                                        std::is_same_v<std::remove_extent_t<std::remove_reference_t<Arg>>, const char>;

// The data a call's arguments make: a struct passed alone is the data itself, and any other arguments are a tuple of
// their values.
template <typename... Args>
struct RecordDataOf {
    using type = std::tuple<ValueOf<Args>...>;
};
template <typename Arg>
struct RecordDataOf<Arg> {
    using Value = ValueOf<Arg>;
    using type = std::conditional_t<
        std::is_class_v<Value> && !spd::is_leaf_type<Value> && !spd::is_std_array<Value>,
        Value,
        std::tuple<Value>>;
};
template <typename... Args>
using RecordData = typename RecordDataOf<Args...>::type;

template <typename Data>
using ArgumentList = std::conditional_t<spd::is_tuple<Data>, Data, std::tuple<Data>>;
template <typename Data, size_t I>
using ArgumentAt = std::tuple_element_t<I, ArgumentList<Data>>;
template <typename Data>
using ArgumentIndices = std::make_index_sequence<std::tuple_size_v<ArgumentList<Data>>>;

// A zone's data, the time it opened. Its call site describes its records as Intervals whose end, their second field, is
// when they are published, which the host fills in.
struct ZoneStart {
    sp::Timestamp start;
};

inline constexpr uint16_t kCodeOpen = uint16_t('{') << 8;
inline constexpr uint16_t kCodeClose = uint16_t('}') << 8;
// A struct's field codes with each nested struct's between brackets, which lay the struct out.
template <size_t N>
struct BracketedCodeList : spd::CodeList<N> {
    constexpr void open() { this->put(kCodeOpen); }
    constexpr void close() { this->put(kCodeClose); }
};
template <typename T>
inline constexpr const auto& leaf_type_codes_with_struct_brackets = spd::leaf_type_codes<T, BracketedCodeList>;

struct CodeLayout {
    uint32_t size = 0;
    uint32_t align = 1;
};
constexpr uint32_t round_up(uint32_t bytes, uint32_t align) { return (bytes + align - 1) / align * align; }
// Lays out one leaf or bracketed struct of `codes`, from `next_code` on, as the compiler lays out a struct: each leaf's
// offset within it goes to `offsets[next_leaf]` on, and both cursors move past what it took.
constexpr CodeLayout lay_out_codes(const uint16_t* codes, size_t& next_code, uint32_t* offsets, size_t& next_leaf) {
    const uint16_t code = codes[next_code++];
    if (code != kCodeOpen) {
        offsets[next_leaf++] = 0;
        return {spd::code_bytes(code), spd::code_bytes(code)};
    }
    CodeLayout layout;
    while (codes[next_code] != kCodeClose) {
        const size_t first_leaf = next_leaf;
        const CodeLayout member = lay_out_codes(codes, next_code, offsets, next_leaf);
        const uint32_t member_offset = round_up(layout.size, member.align);
        for (size_t leaf = first_leaf; leaf < next_leaf; ++leaf) {
            offsets[leaf] += member_offset;
        }
        layout.size = member_offset + member.size;
        layout.align = std::max(layout.align, member.align);
    }
    ++next_code;
    layout.size = round_up(layout.size, layout.align);
    return layout;
}

// Where V's fields are as its field codes lay V out, and whether that layout is the compiler's, which it is when it
// gives V's own size and alignment (an empty V has neither). A record sends an argument's first sent_bytes: up to the
// end of its last field when the layouts match, all of it when they don't.
template <size_t kLeaves>
struct FieldLayout {
    bool matches_compiler = false;
    std::array<uint16_t, kLeaves> offsets{};
    std::array<uint16_t, kLeaves> sizes{};
    uint32_t sent_bytes = 0;
    size_t gap_bytes = 0;  // bytes before sent_bytes that no field covers
    constexpr bool is_covered(uint32_t byte) const {
        for (size_t field = 0; field < kLeaves; ++field) {
            if (offsets[field] <= byte && byte < offsets[field] + sizes[field]) {
                return true;
            }
        }
        return false;
    }
};
template <typename V>
constexpr FieldLayout<spd::leaf_type_codes<V>.size()> field_layout() {
    std::array<uint32_t, spd::leaf_type_codes<V>.size()> offsets{};
    size_t next_code = 0, next_leaf = 0;
    const CodeLayout whole =
        lay_out_codes(leaf_type_codes_with_struct_brackets<V>.data(), next_code, offsets.data(), next_leaf);
    FieldLayout<spd::leaf_type_codes<V>.size()> layout;
    layout.matches_compiler = std::is_empty_v<V> || (whole.size == sizeof(V) && whole.align == alignof(V));
    layout.sent_bytes = layout.matches_compiler ? 0 : uint32_t(sizeof(V));
    for (size_t field = 0; field < layout.offsets.size(); ++field) {
        layout.offsets[field] = uint16_t(offsets[field]);
        layout.sizes[field] = uint16_t(spd::code_bytes(spd::leaf_type_codes<V>[field]));
        if (layout.matches_compiler) {
            layout.sent_bytes = std::max<uint32_t>(layout.sent_bytes, layout.offsets[field] + layout.sizes[field]);
        }
    }
    for (uint32_t byte = 0; layout.matches_compiler && byte < layout.sent_bytes; ++byte) {
        layout.gap_bytes += layout.is_covered(byte) ? 0 : 1;
    }
    return layout;
}
template <typename V>
inline constexpr auto kFieldLayout = field_layout<V>();

template <typename T>
constexpr bool holds(uint16_t code) {
    for (const uint16_t leaf : spd::leaf_type_codes<T>) {
        if (leaf == code) {
            return true;
        }
    }
    return false;
}

// A Timestamp travels as a distance when the field codes describe every argument's layout; otherwise it goes whole.
template <typename Data, size_t... I>
constexpr bool sends_distances(std::index_sequence<I...>) {
    return holds<Data>(spd::kTimestampCode) && (kFieldLayout<ArgumentAt<Data, I>>.matches_compiler && ...);
}
template <typename Data>
inline constexpr bool kSendsDistances = sends_distances<Data>(ArgumentIndices<Data>{});

// A record's payload is made of slices of its arguments. Each slice is 1, 2, 4 or 8 bytes at a multiple of its size
// within its argument, so it loads whole. When the field codes describe an argument's layout, its slices are its fields
// and every byte before its last field's end that no field covers, so a bit-field the codes place elsewhere still goes
// out. Otherwise they are the whole argument cut at its alignment, at most 8 bytes each.
struct Slice {
    bool is_distance;
    uint16_t argument, from, bytes;  // the argument it is from, its offset in that argument, and its size
};
template <typename V>
inline constexpr uint16_t kWholeSliceBytes = uint16_t(std::min<size_t>(alignof(V), 8));
template <typename V>
inline constexpr size_t kArgumentSlices =
    kFieldLayout<V>.matches_compiler ? spd::leaf_type_codes<V>.size() + kFieldLayout<V>.gap_bytes
                                     : sizeof(V) / kWholeSliceBytes<V>;
template <typename V>
constexpr std::array<Slice, kArgumentSlices<V>> argument_slices(uint16_t argument, bool as_distances) {
    std::array<Slice, kArgumentSlices<V>> slices{};
    if constexpr (kFieldLayout<V>.matches_compiler) {
        constexpr auto& layout = kFieldLayout<V>;
        size_t k = 0;
        for (; k < layout.offsets.size(); ++k) {
            const bool is_distance = as_distances && spd::leaf_type_codes<V>[k] == spd::kTimestampCode;
            slices[k] = {is_distance, argument, layout.offsets[k], layout.sizes[k]};
        }
        for (uint32_t byte = 0; byte < layout.sent_bytes; ++byte) {
            if (!layout.is_covered(byte)) {
                slices[k++] = {false, argument, uint16_t(byte), 1};
            }
        }
    } else {
        for (size_t k = 0; k < slices.size(); ++k) {
            slices[k] = {false, argument, uint16_t(k * kWholeSliceBytes<V>), kWholeSliceBytes<V>};
        }
    }
    return slices;
}
template <typename Data, size_t... I>
constexpr auto record_slices(std::index_sequence<I...>) {
    std::array<Slice, (kArgumentSlices<ArgumentAt<Data, I>> + ... + 0)> slices{};
    [[maybe_unused]] size_t count = 0;
    [[maybe_unused]] const auto add = [&](const auto& argument_slices) {
        for (const Slice& slice : argument_slices) {
            slices[count++] = slice;
        }
    };
    (add(argument_slices<ArgumentAt<Data, I>>(uint16_t(I), kSendsDistances<Data>)), ...);
    return slices;
}
template <typename Data>
inline constexpr auto kSlices = record_slices<Data>(ArgumentIndices<Data>{});

// Data without distances takes the medium form only, with its arguments whole, so a struct's payload is its bytes,
// which the host copies into its own struct as they are. Arguments go largest alignment first and, within one
// alignment, those whose bytes end on a multiple of it first, each at a multiple of its alignment up to 4, so none of
// its slices straddles a word.
template <typename Data, size_t... I>
constexpr std::array<spd::ArgumentSlice, sizeof...(I)> place_arguments(std::index_sequence<I...>) {
    constexpr std::array<uint32_t, sizeof...(I)> alignments{uint32_t(alignof(ArgumentAt<Data, I>))...};
    std::array<spd::ArgumentSlice, sizeof...(I)> placements{
        spd::ArgumentSlice{uint16_t(I), 0, 0, uint16_t(kFieldLayout<ArgumentAt<Data, I>>.sent_bytes)}...};
    uint32_t end = 0;
    for (uint32_t align = std::max({uint32_t{1}, alignments[I]...}); align != 0; align /= 2) {
        for (const bool ragged : {false, true}) {
            for (size_t argument = 0; argument < placements.size(); ++argument) {
                if (alignments[argument] == align && (placements[argument].bytes % align != 0) == ragged) {
                    placements[argument].to = uint16_t(round_up(end, std::min<uint32_t>(align, 4)));
                    end = placements[argument].to + placements[argument].bytes;
                }
            }
        }
    }
    return placements;
}
template <typename Data>
inline constexpr auto kArgumentPlacement = place_arguments<Data>(ArgumentIndices<Data>{});

// The words the record's time takes after word 0, and the bytes each Timestamp's distance takes, in each form.
constexpr uint32_t time_words(spd::TimeForm form) { return uint32_t(form); }
constexpr uint32_t distance_bytes(spd::TimeForm form) { return 2u << uint32_t(form); }

template <size_t N>
struct PlacedLayout {
    uint32_t body_words;
    std::array<spd::ArgumentSlice, N> slices;
};
template <typename Data>
constexpr PlacedLayout<kSlices<Data>.size()> place_layout(spd::TimeForm form) {
    constexpr auto& slices = kSlices<Data>;
    PlacedLayout<slices.size()> placed{};
    uint32_t end = 0;
    if constexpr (!kSendsDistances<Data>) {
        for (size_t k = 0; k < slices.size(); ++k) {
            const uint32_t to = kArgumentPlacement<Data>[slices[k].argument].to + slices[k].from;
            placed.slices[k] = {slices[k].argument, slices[k].from, uint16_t(to), slices[k].bytes};
            end = std::max(end, to + slices[k].bytes);
        }
    } else {
        // Each slice first-fits at a multiple of its size up to 4, widest first; a payload byte map with room for the
        // short form's time and every slice at its widest can't run out.
        bool taken[4 + 8 * slices.size()] = {};
        const auto is_free = [&](uint32_t offset, uint32_t bytes) {
            for (uint32_t byte = offset; byte < offset + bytes; ++byte) {
                if (taken[byte]) {
                    return false;
                }
            }
            return true;
        };
        if (form == spd::TimeForm::Short) {
            // The short form's record time takes bytes 2 and 3 of every call site's payload, so the decoder finds it
            // before it knows the call site.
            taken[2] = taken[3] = true;
            end = 4;
        }
        for (uint32_t align = 4; align != 0; align /= 2) {
            for (size_t k = 0; k < slices.size(); ++k) {
                const uint32_t bytes = slices[k].is_distance ? distance_bytes(form) : slices[k].bytes;
                if (std::min<uint32_t>(bytes, 4) != align) {
                    continue;
                }
                uint32_t offset = 0;
                while (!is_free(offset, bytes)) {
                    offset += align;
                }
                for (uint32_t byte = offset; byte < offset + bytes; ++byte) {
                    taken[byte] = true;
                }
                placed.slices[k] = {slices[k].argument, slices[k].from, uint16_t(offset), uint16_t(bytes)};
                end = std::max(end, offset + bytes);
            }
        }
    }
    placed.body_words = time_words(form) + (end + 3) / 4;
    return placed;
}
template <typename Data, spd::TimeForm F>
inline constexpr auto kLayout = place_layout<Data>(F);

// Each ELF has its own copy of the lane's cursor (internal/profiler/kernel_profiler_streaming.hpp), and the host keeps
// one per lane. A kernel's launch resets its copy, so only kernels write short records.
#if defined(KERNEL_BUILD)
inline constexpr bool kWritesShortRecords = true;
#else
inline constexpr bool kWritesShortRecords = false;
#endif
template <typename Data>
inline constexpr bool kHasShort =
    kWritesShortRecords && kSendsDistances<Data> &&
    kLayout<Data, spd::TimeForm::Short>.body_words < kLayout<Data, spd::TimeForm::Medium>.body_words;
// The body length in word 0 of a call site's packets: its first form's, shortest first.
template <typename Data>
inline constexpr uint32_t kFirstBody =
    kHasShort<Data> ? kLayout<Data, spd::TimeForm::Short>.body_words : kLayout<Data, spd::TimeForm::Medium>.body_words;
template <typename Data, spd::TimeForm F>
inline constexpr uint32_t kPayloadWords = kLayout<Data, F>.body_words - time_words(F);

// Each slice's first word among the words its record's slices take, an 8-byte slice taking two, low word first.
template <typename Data>
constexpr auto slice_first_words() {
    std::array<uint32_t, kSlices<Data>.size() + 1> first{};
    for (size_t k = 0; k < kSlices<Data>.size(); ++k) {
        first[k + 1] = first[k] + (kSlices<Data>[k].bytes == 8 ? 2 : 1);
    }
    return first;
}
template <typename Data>
inline constexpr auto kSliceFirstWord = slice_first_words<Data>();
// The words a record crosses an out-of-line writer's call in, as arguments in registers: its payload's when it has
// one form, and otherwise its slices', since the writer picks the form.
template <typename Data>
struct RegisterWords {
    std::array<
        uint32_t,
        kSendsDistances<Data> ? kSliceFirstWord<Data>.back() : kPayloadWords<Data, spd::TimeForm::Medium>>
        words;
};

// A record's payload, as the parts it stores in. A run is kRunWords or more payload words that are one argument's
// consecutive 4-aligned words as it lies in memory, which copy straight from there. Every other word is built from the
// slices that land in it. A shorter run builds its words too: the compiler keeps those in registers, and a copy would
// need the argument in memory.
inline constexpr uint32_t kRunWords = 9;
struct PayloadPart {
    uint16_t first_word = 0;
    uint16_t words = 1;
    int16_t argument = -1;  // the argument a run copies from; -1 for a built word
    uint16_t argument_offset = 0;
};
template <size_t kWords>
struct PayloadLayout {
    std::array<PayloadPart, kWords> parts{};
    size_t count = 0;
    bool has_run = false;
};

// Which argument and byte offset each payload word of form F copies whole from, or argument -1 for a word that must be
// built: one holding bytes of two arguments, a Timestamp's distance, or the short form's time.
template <typename Data, spd::TimeForm F, size_t... I>
constexpr std::array<PayloadPart, kPayloadWords<Data, F>> copyable_words(std::index_sequence<I...>) {
    constexpr auto& layout = kLayout<Data, F>;
    constexpr auto& slices = kSlices<Data>;
    constexpr uint32_t sizes[] = {uint32_t(sizeof(ArgumentAt<Data, I>))...};
    constexpr uint32_t alignments[] = {uint32_t(alignof(ArgumentAt<Data, I>))...};
    std::array<PayloadPart, kPayloadWords<Data, F>> words{};
    for (uint32_t word = 0; word < words.size(); ++word) {
        // A copyable word's bytes all come from one argument, each at the same distance from its place in the payload.
        int32_t argument = -1, distance = 0;
        bool copyable = !(F == spd::TimeForm::Short && word == 0);
        for (uint32_t byte = 4 * word; byte < 4 * word + 4; ++byte) {
            for (size_t k = 0; k < slices.size(); ++k) {
                const spd::ArgumentSlice placed = layout.slices[k];
                if (placed.to <= byte && byte < placed.to + placed.bytes) {
                    const int32_t byte_distance = int32_t(placed.from + (byte - placed.to)) - int32_t(byte);
                    copyable &= !slices[k].is_distance &&
                                (argument < 0 || (argument == placed.argument && distance == byte_distance));
                    argument = placed.argument;
                    distance = byte_distance;
                }
            }
        }
        const int32_t offset = int32_t(4 * word) + distance;
        if (copyable && argument >= 0 && alignments[argument] >= 4 && offset >= 0 && offset % 4 == 0 &&
            uint32_t(offset) + 4 <= sizes[argument]) {
            words[word] = {uint16_t(word), 1, int16_t(argument), uint16_t(offset)};
        } else {
            words[word] = {uint16_t(word), 1};
        }
    }
    return words;
}
template <typename Data, spd::TimeForm F, bool kCopyRuns>
constexpr auto payload_layout() {
    constexpr auto words = copyable_words<Data, F>(ArgumentIndices<Data>{});
    PayloadLayout<words.size()> payload;
    for (uint32_t word = 0; word < words.size();) {
        const PayloadPart start = words[word];
        uint32_t end = word + 1;
        while (start.argument >= 0 && end < words.size() && words[end].argument == start.argument &&
               words[end].argument_offset == start.argument_offset + 4 * (end - word)) {
            ++end;
        }
        if (kCopyRuns && start.argument >= 0 && end - word >= kRunWords) {
            payload.parts[payload.count++] = {
                uint16_t(word), uint16_t(end - word), start.argument, start.argument_offset};
            payload.has_run = true;
            word = end;
        } else {
            payload.parts[payload.count++] = {uint16_t(word)};
            ++word;
        }
    }
    return payload;
}
template <typename Data, spd::TimeForm F, bool kCopyRuns>
inline constexpr auto kPayloadLayout = payload_layout<Data, F, kCopyRuns>();
template <typename Data>
inline constexpr bool kHasRun = kPayloadLayout<Data, spd::TimeForm::Medium, true>.has_run;

template <typename Data, size_t... I>
constexpr bool check_arguments(std::index_sequence<I...>) {
    return (spd::check_value<ArgumentAt<Data, I>>() & ... & true);
}
// The rules a DevicePublish call's types follow, given its arguments' types as forwarding references deduce them, with
// the profiler on or off; true when they all hold. Each mistake is one error: a string literal passed as an argument is
// one rather than a cascade. A string travels only in metadata, so a call that would pick one at runtime branches over
// DevicePublish calls instead, one per string.
template <typename Metadata, typename... Args>
constexpr bool check_call() {
    if constexpr ((is_text_literal<Args> || ...)) {
        static_assert(
            !(is_text_literal<Args> || ...),
            "strings are not supported in a payload; send the text as metadata, a DevicePublish call per string");
        return false;
    } else {
        using Data = RecordData<Args...>;
        constexpr bool metadata = spd::check_struct<Metadata>();
        constexpr bool data = check_arguments<Data>(ArgumentIndices<Data>{});
        if constexpr (metadata && data) {
            if constexpr (holds<Data>(spd::kStringCode)) {
                static_assert(
                    !holds<Data>(spd::kStringCode),
                    "strings are not supported in a payload; send the text as metadata, a DevicePublish call per "
                    "string");
                return false;
            } else {
                constexpr auto kLongest = kSendsDistances<Data> ? spd::TimeForm::Full : spd::TimeForm::Medium;
                static_assert(
                    kLayout<Data, kLongest>.body_words <= ppfmt::LENGTH_MASK,
                    "this record is over the 504-byte limit (500 with Timestamp fields); split it into several "
                    "records");
            }
        }
        return metadata && data;
    }
}

// Every object below is a static template, so a call site in an inline function, a template or a class-template
// member is never COMDAT, which GCC can't put in a named section. Nothing is `used`: the call site's asm references its
// handle and entry, so a copy of an inline function that LTO drops takes its entry with it.

// A string's exact-length copy, one per distinct text.
template <typename C, C... cs>
static constexpr C section_string[] __attribute__((section(".tt_record_data"), unused)) = {cs..., C(0)};

// A text's 64-bit FNV-1a hash, which a String holds.
template <typename C, C... cs>
inline constexpr uint64_t kStringHash = [] {
    uint64_t hash = 0xcbf29ce484222325;
    ((hash = (hash ^ uint8_t(cs)) * 0x100000001b3), ...);
    return hash;
}();
// A String's text by its hash, for the host to look it up by. A String doesn't point at it, so nothing would keep it
// but `used`.
template <typename C, C... cs>
static const spd::StringEntry section_string_entry __attribute__((section(".tt_record_strings"), used)) = {
    kStringHash<C, cs...>, section_string<C, cs...>, uint32_t(sizeof...(cs))};

template <typename T>
static const auto section_type_name __attribute__((section(".tt_record_data"), unused)) = spd::kTypeNameChars<T>;

// The size of a call site's function name: its __PRETTY_FUNCTION__ without the " [with ...]" GCC appends inside a
// template, so every instantiation of a template is one call site.
constexpr size_t function_name_size(const char* pretty) {
    const char* with = __builtin_strstr(pretty, " [with ");
    return (with ? static_cast<size_t>(with - pretty) : __builtin_strlen(pretty)) + 1;
}
template <size_t N>
struct CallSiteLocation {
    template <size_t P>
    constexpr CallSiteLocation(sp::String file, uint32_t line, const char (&pretty)[P]) : file(file), line(line) {
        for (size_t k = 0; k + 1 < N; ++k) {
            function[k] = pretty[k];
        }
    }
    sp::String file;
    uint32_t line;
    char function[N] = {};
};
template <auto kLocation, size_t... I>
constexpr const char* section_function_name(std::index_sequence<I...>) {
    return section_string<char, kLocation.function[I]...>;
}

template <auto kMetadata>
static const auto section_metadata __attribute__((section(".tt_record_data"), unused)) = kMetadata;

template <typename T>
static const auto section_codes __attribute__((section(".tt_record_data"), unused)) = spd::leaf_type_codes<T>;

// A struct's samples, one per field: the struct with that field's bits set and every other field zero.
inline constexpr size_t kEveryField = ~size_t{0};
struct SampleSource {
    size_t field_to_set = 0;  // or kEveryField
    size_t next_field = 0;
    constexpr void open() {}
    constexpr void close() {}
    // A float's or a double's smallest negative value sets its lowest and highest bits, a bool's true its only one.
    template <typename V>
    constexpr V leaf() {
        const bool set = next_field++ == field_to_set || field_to_set == kEveryField;
        if (!set) {
            return V{};
        }
        if constexpr (std::is_same_v<V, sp::String> || std::is_same_v<V, sp::Timestamp>) {
            return __builtin_bit_cast(V, ~uint64_t{0});
        } else if constexpr (std::is_same_v<V, bool>) {
            return true;
        } else if constexpr (std::is_floating_point_v<V>) {
            return -std::numeric_limits<V>::denorm_min();
        } else if constexpr (std::is_enum_v<V>) {
            return V(~std::underlying_type_t<V>(0));
        } else {
            return V(~V(0));
        }
    }
};
template <typename T>
constexpr T sample_of(size_t field) {
    SampleSource source{field};
    return spd::build<T>(source);
}

// One byte as eight one-bit objects, so that each bit of a bit_cast is defined or not on its own.
struct ByteBits {
    unsigned char b0 : 1, b1 : 1, b2 : 1, b3 : 1, b4 : 1, b5 : 1, b6 : 1, b7 : 1;
};
template <size_t N>
struct Bytes {
    unsigned char at[N];
};
template <size_t N>
struct Bits {
    ByteBits at[N];
};
// An object's N bytes, read whole and bit by bit.
template <size_t N>
struct ObjectBytes {
    template <typename T>
    constexpr explicit ObjectBytes(const T& object) :
        bytes(__builtin_bit_cast(Bytes<N>, object)), bits(__builtin_bit_cast(Bits<N>, object)) {}
    // Byte i's value bits: __builtin_constant_p of a byte or a bit is 1 exactly when the language defines all of it,
    // which it does for value bits and not for padding bits.
    constexpr unsigned char value_bits(size_t i) const {
        if (__builtin_constant_p(bytes.at[i])) {
            return 0xFF;
        }
        const ByteBits& b = bits.at[i];
        return __builtin_constant_p(b.b0) | __builtin_constant_p(b.b1) << 1 | __builtin_constant_p(b.b2) << 2 |
               __builtin_constant_p(b.b3) << 3 | __builtin_constant_p(b.b4) << 4 | __builtin_constant_p(b.b5) << 5 |
               __builtin_constant_p(b.b6) << 6 | __builtin_constant_p(b.b7) << 7;
    }
    // Byte i with its padding bits zero, given its value bits.
    constexpr unsigned char defined(size_t i, unsigned char value_bits) const {
        if (value_bits == 0xFF) {
            return bytes.at[i];
        }
        const ByteBits& b = bits.at[i];
        return (value_bits & 0x01 ? b.b0 : 0) | (value_bits & 0x02 ? b.b1 : 0) << 1 |
               (value_bits & 0x04 ? b.b2 : 0) << 2 | (value_bits & 0x08 ? b.b3 : 0) << 3 |
               (value_bits & 0x10 ? b.b4 : 0) << 4 | (value_bits & 0x20 ? b.b5 : 0) << 5 |
               (value_bits & 0x40 ? b.b6 : 0) << 6 | (value_bits & 0x80 ? b.b7 : 0) << 7;
    }
    Bytes<N> bytes;
    Bits<N> bits;
};
// Each byte's value bits in T. Which of T's bits are value bits doesn't depend on the values, so the sample with every
// field set tells them.
template <typename T>
constexpr Bytes<sizeof(T)> value_bit_mask() {
    const ObjectBytes<sizeof(T)> full(sample_of<T>(kEveryField));
    Bytes<sizeof(T)> mask{};
    for (size_t i = 0; i < sizeof(T); ++i) {
        mask.at[i] = full.value_bits(i);
    }
    return mask;
}
// T's samples, each its representation as the language defines it: each value bit as the sample sets it and every
// padding bit zero, which nothing about how the compiler writes padding can change. An empty type has no samples, and
// one with unique object representations has no padding bits (the language says so), so its samples as written are
// exactly its value bits.
template <typename T, size_t... K>
constexpr auto samples_of(std::index_sequence<K...>) {
    if constexpr (std::is_empty_v<T> || std::has_unique_object_representations_v<T>) {
        return std::array<T, sizeof...(K)>{sample_of<T>(K)...};
    } else {
        const Bytes<sizeof(T)> mask = value_bit_mask<T>();
        const T each[] = {sample_of<T>(K)...};
        Bytes<sizeof(T) * sizeof...(K)> samples{};
        size_t at = 0;
        for (const T& sample : each) {
            const ObjectBytes<sizeof(T)> object(sample);
            for (size_t i = 0; i < sizeof(T); ++i) {
                samples.at[at++] = object.defined(i, mask.at[i]);
            }
        }
        return samples;
    }
}
template <typename T>
static const auto section_samples __attribute__((section(".tt_record_data"), unused)) =
    samples_of<T>(std::make_index_sequence<spd::leaf_type_codes<T>.size()>{});

template <typename V>
constexpr spd::CallArgument call_site_argument() {
    if constexpr (spd::is_std_array<V>) {
        spd::CallArgument argument = call_site_argument<typename V::value_type>();
        argument.count = uint32_t(std::tuple_size_v<V>);
        argument.is_array = 1;
        return argument;
    } else if constexpr (spd::is_leaf_type<V>) {
        const char* const name = std::is_enum_v<V> ? section_type_name<V>.data() : nullptr;
        return {.type = name, .field_count = 1, .count = 1, .is_array = 0, .samples = nullptr, .size = sizeof(V)};
    } else {
        return {
            .type = section_type_name<V>.data(),
            .field_count = uint32_t(spd::leaf_type_codes<V>.size()),
            .count = 1,
            .is_array = 0,
            .samples = &section_samples<V>,
            .size = sizeof(V)};
    }
}
template <typename Data, size_t... I>
constexpr std::array<spd::CallArgument, sizeof...(I)> argument_table(std::index_sequence<I...>) {
    return {call_site_argument<ArgumentAt<Data, I>>()...};
}
template <typename Data>
static const auto section_arguments __attribute__((section(".tt_record_data"), unused)) =
    argument_table<Data>(ArgumentIndices<Data>{});

// A form's slices for the host: each argument whole for data without distances, whose slices lie together; each slice
// otherwise.
template <typename Data, spd::TimeForm F>
static const auto section_layout_slices __attribute__((section(".tt_record_data"), unused)) = [] {
    if constexpr (kSendsDistances<Data>) {
        return kLayout<Data, F>.slices;
    } else {
        return kArgumentPlacement<Data>;
    }
}();
template <typename Data, spd::TimeForm F>
constexpr spd::PacketLayout call_site_layout() {
    return {
        .time_form = F,
        .body_words = kLayout<Data, F>.body_words,
        .slices = section_layout_slices<Data, F>.data(),
        .slice_count = uint32_t(section_layout_slices<Data, F>.size())};
}
// The forms a call site keeps, shortest first.
template <typename Data>
constexpr auto layout_list() {
    constexpr bool kShort = kHasShort<Data>, kFull = kSendsDistances<Data>;
    std::array<spd::PacketLayout, 1 + kShort + kFull> list{};
    size_t n = 0;
    if constexpr (kShort) {
        list[n++] = call_site_layout<Data, spd::TimeForm::Short>();
    }
    list[n++] = call_site_layout<Data, spd::TimeForm::Medium>();
    if constexpr (kFull) {
        list[n] = call_site_layout<Data, spd::TimeForm::Full>();
    }
    return list;
}
template <typename Data>
static const auto section_layouts __attribute__((section(".tt_record_data"), unused)) = layout_list<Data>();

// The data type a call site describes its records as, and which of its fields is their publish time and isn't sent.
template <typename Data>
struct Described {
    using type = Data;
    static constexpr uint32_t kPublishedField = spd::kNoPublishedField;
};
template <>
struct Described<ZoneStart> {
    using type = sp::Interval;
    static constexpr uint32_t kPublishedField = 1;
};

// The call site's id is this byte's address. It is writable so that no flag can merge it with another site's:
// -fmerge-all-constants folds identical constants, and distinct writable objects always have distinct addresses.
template <auto kLocation, auto kMetadata, typename Data>
static char call_site_handle __attribute__((section(".tt_record_ids"), unused)) = 0;

template <auto kLocation, auto kMetadata, typename Data>
static const spd::CallSiteEntry call_site_entry __attribute__((section(".tt_record_meta"), unused)) = {
    .id = &call_site_handle<kLocation, kMetadata, Data>,
    .metadata_type = section_type_name<decltype(kMetadata)>.data(),
    .file = kLocation.file.hash,
    .line = kLocation.line,
    .function = section_function_name<kLocation>(std::make_index_sequence<sizeof(kLocation.function) - 1>{}),
    .metadata = &section_metadata<kMetadata>,
    .metadata_size = sizeof(kMetadata),
    .metadata_samples = &section_samples<decltype(kMetadata)>,
    .metadata_codes = section_codes<decltype(kMetadata)>.data(),
    .metadata_field_count = uint32_t(section_codes<decltype(kMetadata)>.size()),
    .data_codes = section_codes<typename Described<Data>::type>.data(),
    .data_field_count = uint32_t(section_codes<typename Described<Data>::type>.size()),
    .data_arguments = section_arguments<typename Described<Data>::type>.data(),
    .data_argument_count = uint32_t(section_arguments<typename Described<Data>::type>.size()),
    .data_layouts = section_layouts<Data>.data(),
    .data_layout_count = uint32_t(section_layouts<Data>.size()),
    .data_published_field = Described<Data>::kPublishedField,
};
}  // namespace kernel_profiler::records

// Records travel on the streaming profiler's rings, so internal/profiler/kernel_profiler_streaming.hpp includes this
// header once its ring is defined. Elsewhere a call checks its types by the same rules and compiles to nothing.
#if defined(PROFILE_STREAMING) && defined(PROFILE_KERNEL) && !defined(STREAMING_PROFILER_RELAY_KERNEL)

namespace kernel_profiler::records {

// A stall is a zone's medium record, and the smallest room check reserves a zone's longest packet.
static_assert(kLayout<ZoneStart, spd::TimeForm::Medium>.body_words == ppfmt::STALL_BODY_WORDS);
static_assert(1 + kLayout<ZoneStart, spd::TimeForm::Full>.body_words == MIN_RESERVE_WORDS);

// Argument I of a record where the publishing call holds it: the data struct itself, or the I-th argument.
template <size_t I, typename Source>
FORCE_INLINE const auto& argument_at(const Source& source) {
    if constexpr (spd::is_tuple<Source>) {
        return std::get<I>(source);
    } else {
        return source;
    }
}

// The record's arguments where the publishing call holds them: the data struct it passed, or references to its
// arguments, so each payload word reads its argument where it lies rather than from a copy.
template <typename... Args>
FORCE_INLINE decltype(auto) record_source(const Args&... args) {
    if constexpr (spd::is_tuple<RecordData<Args...>>) {
        return std::tuple<const Args&...>(args...);
    } else {
        return (args, ...);
    }
}

template <uint32_t kBytes>
using UintOfSize = std::conditional_t<
    kBytes == 1,
    uint8_t,
    std::conditional_t<kBytes == 2, uint16_t, std::conditional_t<kBytes == 4, uint32_t, uint64_t>>>;

// Slice K's bits, from the record's arguments where the publishing call holds them or from its register words. The load
// is typed rather than a memcpy through an aligned pointer, so the fields of an argument held in registers stay there.
template <typename Data, size_t K, typename Source>
FORCE_INLINE auto slice_bits(const Source& source) {
    constexpr Slice slice = kSlices<Data>[K];
    using Bits = std::conditional_t<slice.bytes == 8, uint64_t, uint32_t>;
    if constexpr (std::is_same_v<Source, RegisterWords<Data>>) {
        constexpr uint32_t first = kSliceFirstWord<Data>[K];
        if constexpr (slice.bytes == 8) {
            return uint64_t{source.words[first + 1]} << 32 | source.words[first];
        } else {
            return Bits{source.words[first]};
        }
    } else {
        using Load __attribute__((may_alias)) = UintOfSize<slice.bytes>;
        const auto* bytes = reinterpret_cast<const unsigned char*>(&argument_at<slice.argument>(source)) + slice.from;
        return Bits{*reinterpret_cast<const Load*>(bytes)};
    }
}

// Slice K's bits where form F's payload word `word` holds them, or 0 when the slice lies elsewhere. Every caller passes
// a constant word, so the tests fold away.
template <typename Data, spd::TimeForm F, size_t K, typename Bits>
FORCE_INLINE uint32_t slice_in_word(Bits bits, uint32_t word) {
    constexpr spd::ArgumentSlice copy = kLayout<Data, F>.slices[K];
    if (copy.to / 4u == word) {
        return uint32_t(bits) << (8 * (copy.to % 4));
    }
    if (copy.bytes == 8 && copy.to / 4u + 1 == word) {
        return uint32_t(uint64_t(bits) >> 32);
    }
    return 0;
}

// Where a record's payload words go: ring slots after its head, or the ring word by word. Each takes a payload word's
// index, which only the slots use.
struct SlotStore {
    volatile tt_l1_ptr uint32_t* slots;

    FORCE_INLINE void write_word(uint32_t index, uint32_t value) const { slots[index] = value; }

    // Each batch's loads go ahead of its stores, so they overlap instead of each waiting out the one before: a copy a
    // word at a time runs slower than building the words in registers. Four words a batch copy a 32-word run in about
    // half the cycles of one at a time, with no more code; eight are about 10% faster again but larger.
    template <uint32_t kWords>
    FORCE_INLINE void copy_run(uint32_t index, const AliasingWord* from) const {
        constexpr uint32_t kBatch = 4;
        constexpr uint32_t kBatched = kWords / kBatch * kBatch;
        for (uint32_t done = 0; done < kBatched; done += kBatch) {
            copy_batch(slots + index + done, from + done, std::make_index_sequence<kBatch>{});
        }
        if constexpr (kWords != kBatched) {
            copy_batch(slots + index + kBatched, from + kBatched, std::make_index_sequence<kWords - kBatched>{});
        }
    }

private:
    template <size_t... B>
    static FORCE_INLINE void copy_batch(
        volatile tt_l1_ptr uint32_t* to, const AliasingWord* from, std::index_sequence<B...>) {
        const uint32_t batch[] = {from[B]...};
        ((to[B] = batch[B]), ...);
    }
};
struct WordStore {
    static FORCE_INLINE void write_word(uint32_t, uint32_t value) { ring_write_word(value); }
    template <uint32_t kWords>
    static FORCE_INLINE void copy_run(uint32_t, const AliasingWord* from) {
        ring_copy_words(from, kWords);
    }
};

// The writer of every record: a DevicePublish call's, or a zone's, whose data is its ZoneStart. Source is where it
// reads the record's arguments, as record_source gives them or as RegisterWords. Data with Timestamps writes the
// shortest form they fit; other data is always medium.
template <typename Data, typename Source>
struct RecordWriter {
    using Form = spd::TimeForm;

    // The longest packet: a full one, or a medium one with the timer's high word ahead of it.
    static constexpr uint32_t kMaxWords = [] {
        if constexpr (kSendsDistances<Data>) {
            return 1 + kLayout<Data, Form::Full>.body_words;
        } else {
            return 2 + kLayout<Data, Form::Medium>.body_words;
        }
    }();

    // `inputs` are the record's register words, or the address of its source.
    template <typename... Inputs>
    static FORCE_INLINE void write(uint32_t header, uint32_t time_hi, uint32_t time_lo, Inputs... inputs) {
        if constexpr (kFromRegisters) {
            write_record(header, time_hi, time_lo, Source{{inputs...}}, kSliceIndices);
        } else {
            write_record(header, time_hi, time_lo, *inputs..., kSliceIndices);
        }
    }

private:
    static constexpr auto kSliceIndices = std::make_index_sequence<kSlices<Data>.size()>{};
    static constexpr bool kFromRegisters = std::is_same_v<Source, RegisterWords<Data>>;
    static constexpr bool kCopyRuns = !kFromRegisters;

    template <size_t K>
    static constexpr bool kIsDistance = kSlices<Data>[K].is_distance;

    // The record being written: where its arguments are, its time, and each slice's Timestamp bits, which are read
    // before anything is written because they choose the form (0 for any other slice, which is read as it is stored).
    struct OutgoingRecord {
        const Source& source;
        uint32_t time_hi;
        uint32_t time_lo;
        std::array<uint64_t, kSlices<Data>.size()> stamps;
    };

    template <size_t... K>
    static FORCE_INLINE void write_record(
        uint32_t header, uint32_t time_hi, uint32_t time_lo, const Source& source, std::index_sequence<K...>) {
        const OutgoingRecord record{source, time_hi, time_lo, {timestamp_bits<K>(source)...}};
        if constexpr (kSendsDistances<Data>) {
            write_shortest_form(header, record, kSliceIndices);
        } else {
            write_form<Form::Medium>(header, record);
        }
    }
    template <size_t K>
    static FORCE_INLINE uint64_t timestamp_bits(const Source& source) {
        if constexpr (kIsDistance<K>) {
            return slice_bits<Data, K>(source);
        } else {
            return 0;
        }
    }

    // A Timestamp's distance back from the record's time, its low word and its high word, which borrows when the
    // Timestamp is later, as a torn read is; 0 for any other slice.
    template <size_t K, typename Record>
    static FORCE_INLINE uint32_t distance_lo(const Record& record) {
        if constexpr (kIsDistance<K>) {
            return record.time_lo - uint32_t(record.stamps[K]);
        } else {
            return 0;
        }
    }
    template <size_t K, typename Record>
    static FORCE_INLINE uint32_t distance_hi(const Record& record) {
        if constexpr (kIsDistance<K>) {
            const uint64_t stamp = record.stamps[K];
            return record.time_hi - uint32_t(stamp >> 32) - (record.time_lo < uint32_t(stamp));
        } else {
            return 0;
        }
    }

    template <Form F, size_t K, typename Record>
    static FORCE_INLINE auto sent_slice(const Record& record) {
        if constexpr (kIsDistance<K> && F == Form::Full) {
            return uint64_t{distance_hi<K>(record)} << 32 | distance_lo<K>(record);
        } else if constexpr (kIsDistance<K>) {
            return distance_lo<K>(record);
        } else {
            return slice_bits<Data, K>(record.source);
        }
    }

    // A short record needs its time within 16 bits after the lane's cursor and in the cursor's high-word period, and
    // every Timestamp within 16 bits back; a medium one needs every Timestamp within 32 bits back. No cursor (high word
    // kUnset) fails the high-word test. The tests are one OR-tree per form, and the short record, the commonest on a
    // busy lane, falls through first.
    template <typename Record, size_t... K>
    static FORCE_INLINE void write_shortest_form(uint32_t header, const Record& record, std::index_sequence<K...>) {
        const uint32_t beyond_32_bits = (distance_hi<K>(record) | ... | 0u);
        if constexpr (kHasShort<Data>) {
            const uint32_t since_cursor = record.time_lo - g_cursor_lo;
            const uint32_t beyond_16_bits = (since_cursor | ... | distance_lo<K>(record)) >> 16;
            if (__builtin_expect((beyond_16_bits | (record.time_hi ^ g_cursor_hi) | beyond_32_bits) == 0, 1)) {
                write_form<Form::Short>(header, record);
                g_cursor_lo = record.time_lo;
                g_cursor_hi = record.time_hi;
                return;
            }
        }
        if (__builtin_expect(beyond_32_bits == 0, 1)) {
            write_form<Form::Medium>(header, record);
            if constexpr (kHasShort<Data>) {
                g_cursor_lo = record.time_lo;
                g_cursor_hi = record.time_hi;
            }
            return;
        }
        write_full(header, record.time_hi, record.time_lo, sent_slice<Form::Full, K>(record)...);
    }

    // Out of line, so call sites don't carry the full form's code; its slices come as values.
    template <typename... Sent>
    static __attribute__((noinline)) void write_full(
        uint32_t header, uint32_t time_hi, uint32_t time_lo, Sent... sent) {
        ring_write_word(header + length_change<Form::Full>());
        ring_write_word(time_lo);
        ring_write_word(time_hi);
        write_full_payload(std::make_index_sequence<kPayloadWords<Data, Form::Full>>{}, sent...);
    }
    template <size_t... W, typename... Sent>
    static FORCE_INLINE void write_full_payload(std::index_sequence<W...>, Sent... sent) {
        (ring_write_word(full_word<W>(std::index_sequence_for<Sent...>{}, sent...)), ...);
    }
    template <size_t W, size_t... K, typename... Sent>
    static FORCE_INLINE uint32_t full_word(std::index_sequence<K...>, Sent... sent) {
        return (slice_in_word<Data, Form::Full, K>(sent, W) | ... | 0u);
    }

    // Word 0 is the call site's, with its first form's body length, so another form's adjusts it.
    template <Form F>
    static constexpr uint32_t length_change() {
        return (kLayout<Data, F>.body_words - kFirstBody<Data>) << ppfmt::LENGTH_SHIFT;
    }

    // A record stores as one block when it fits before the ring's end and is long enough for the check to pay for
    // itself: a medium record with payload, or one with a run. A short record is otherwise a word or two, cheaper word
    // by word.
    template <Form F, typename Record>
    static FORCE_INLINE void write_form(uint32_t header, const Record& record) {
        constexpr auto& payload = kPayloadLayout<Data, F, kCopyRuns>;
        constexpr uint32_t kHeadWords = F == Form::Short ? 1 : 2;
        constexpr uint32_t kWords = kHeadWords + kPayloadWords<Data, F>;
        constexpr bool kBlock = kWords > 2 && (F == Form::Medium || payload.has_run);
        constexpr auto parts = std::make_index_sequence<payload.count>{};
        // The short form keeps the time's distance from the cursor in the top half of its first payload word.
        const uint32_t short_time = F == Form::Short ? (record.time_lo - g_cursor_lo) << 16 : 0;
        if constexpr (F == Form::Medium) {
            ring_write_sticky_timer(record.time_hi);
        }
        if constexpr (kBlock) {
            const uint32_t index = wIndex;
            if (__builtin_expect(ring_fits_before_end(index, kWords), 1)) {
                volatile tt_l1_ptr uint32_t* const slots = ring_slots(index);
                slots[0] = header + length_change<F>();
                if constexpr (F == Form::Medium) {
                    slots[1] = record.time_lo;
                }
                store_payload<F>(SlotStore{slots + kHeadWords}, short_time, record, parts);
                wIndex = index + kWords;
                return;
            }
        }
        ring_write_word(header + length_change<F>());
        if constexpr (F == Form::Medium) {
            ring_write_word(record.time_lo);
        }
        store_payload<F>(WordStore{}, short_time, record, parts);
    }

    template <Form F, typename Store, typename Record, size_t... P>
    static FORCE_INLINE void store_payload(
        Store store, uint32_t short_time, const Record& record, std::index_sequence<P...>) {
        (store_part<F, kPayloadLayout<Data, F, kCopyRuns>.parts[P]>(store, short_time, record), ...);
    }
    template <Form F, PayloadPart kPart, typename Store, typename Record>
    static FORCE_INLINE void store_part(Store store, uint32_t short_time, const Record& record) {
        if constexpr (kPart.argument < 0) {
            const uint32_t word = built_word<F, kPart.first_word>(record, kSliceIndices);
            store.write_word(kPart.first_word, kPart.first_word == 0 ? word | short_time : word);
        } else {
            const auto* argument = reinterpret_cast<const unsigned char*>(&argument_at<kPart.argument>(record.source));
            store.template copy_run<kPart.words>(
                kPart.first_word, reinterpret_cast<const AliasingWord*>(argument + kPart.argument_offset));
        }
    }

    template <Form F, size_t W, typename Record, size_t... K>
    static FORCE_INLINE uint32_t built_word(const Record& record, std::index_sequence<K...>) {
        if constexpr (kFromRegisters && !kSendsDistances<Data>) {
            return record.source.words[W];
        } else {
            return (slice_in_word<Data, F, K>(sent_slice<F, K>(record), W) | ... | 0u);
        }
    }
};

template <typename Data, size_t A, typename Source, size_t... K>
FORCE_INLINE uint32_t register_word(const Source& source, std::index_sequence<K...>) {
    if constexpr (!kSendsDistances<Data>) {
        return (slice_in_word<Data, spd::TimeForm::Medium, K>(slice_bits<Data, K>(source), A) | ... | 0u);
    } else {
        constexpr size_t slice = [] {
            size_t k = 0;
            while (kSliceFirstWord<Data>[k + 1] <= A) {
                ++k;
            }
            return k;
        }();
        const auto bits = slice_bits<Data, slice>(source);
        if constexpr (A == kSliceFirstWord<Data>[slice]) {
            return uint32_t(bits);
        } else {
            return uint32_t(uint64_t(bits) >> 32);
        }
    }
}

// A record with a run crosses the writer's call by its source's address, which the run copies from; any other record
// as its register words, so a small record stays in registers when the writer isn't inlined.
template <bool kImmediately, bool kNoInline, typename Data, typename Source, size_t... A>
FORCE_INLINE void publish_record(uint32_t header, const Source& source, std::index_sequence<A...>) {
    if constexpr (kHasRun<Data>) {
        publish_with<RecordWriter<Data, Source>, kImmediately, kNoInline>(header, &source);
    } else {
        constexpr auto slices = std::make_index_sequence<kSlices<Data>.size()>{};
        publish_with<RecordWriter<Data, RegisterWords<Data>>, kImmediately, kNoInline>(
            header, register_word<Data, A>(source, slices)...);
    }
}

// A call site, unique to the file, line and function DevicePublishSite names. Everything that depends only on the types
// is shared between call sites, so each call site instantiates this one function.
template <auto kLocation>
struct CallSite {
    template <auto kMetadata = spd::NoMetadata{}, auto... Options, typename... Arguments>
    static FORCE_INLINE void publish(Arguments&&... arguments) {
        if constexpr (check_call<decltype(kMetadata), Arguments...>() && check_options<decltype(Options)...>()) {
            using Data = RecordData<Arguments...>;
            constexpr bool immediately = kHasOption<sp::PublishImmediately, Options...>;
            // Only an inlined profiler has a writer to keep out of line; otherwise every call site shares one.
            constexpr bool no_inline = kHasOption<sp::NoInline, Options...> && kProfilerInlined;
            const uint32_t header = site_header<kMetadata, Data, ppfmt::record_w0(kFirstBody<Data>)>();
            publish_record<immediately, no_inline, Data>(
                header,
                record_source(arguments...),
                std::make_index_sequence<std::tuple_size_v<decltype(RegisterWords<Data>::words)>>{});
        }
    }

private:
    // Word 0 of the call site's packets: lui/addi of the handle, one pair per call site, so the loader can rebase it
    // into the call site's id. The relocation's addend carries kTypeAndLength, which the loader keeps. The asm also
    // references the entry, so the entry exists exactly when the call site's code does.
    template <auto kMetadata, typename Data, uint32_t kTypeAndLength>
    static FORCE_INLINE uint32_t site_header() {
        uint32_t header;
        asm("lui %0, %%hi(%1+%3)\n\taddi %0, %0, %%lo(%1+%3)  # %2"
            : "=r"(header)
            : "i"(&call_site_handle<kLocation, kMetadata, Data>),
              "i"(&call_site_entry<kLocation, kMetadata, Data>),
              "i"(kTypeAndLength));
        return header;
    }
};

}  // namespace kernel_profiler::records

FORCE_INLINE tt::tt_metal::experimental::streaming_profiler::Timestamp
tt::tt_metal::experimental::streaming_profiler::Timestamp::now() {
    uint32_t high, low;
    kernel_profiler::read_wall_clock(high, low);
    Timestamp time;
    time.ticks_ = uint64_t{high} << 32 | low;
    return time;
}

#else

namespace kernel_profiler::records {
template <auto kLocation>
struct CallSite {
    template <auto kMetadata = spd::NoMetadata{}, auto... Options, typename... Arguments>
    static FORCE_INLINE void publish(Arguments&&...) {
        [[maybe_unused]] constexpr bool checked =
            check_call<decltype(kMetadata), Arguments...>() && check_options<decltype(Options)...>();
    }
};
}  // namespace kernel_profiler::records

// A Timestamp only reaches DevicePublish, which drops it here, so taking one reads no clock.
FORCE_INLINE tt::tt_metal::experimental::streaming_profiler::Timestamp
tt::tt_metal::experimental::streaming_profiler::Timestamp::now() {
    return {};
}

#endif

// Usable as a template argument (a string literal is not). The String is its text's hash, and taking the text's entry
// keeps the text in the image, where the host looks it up by that hash.
template <typename C, C... cs>
constexpr tt::tt_metal::experimental::streaming_profiler::String operator""_s() {
    [[maybe_unused]] const auto* const entry = &kernel_profiler::records::section_string_entry<C, cs...>;
    tt::tt_metal::experimental::streaming_profiler::String string;
    string.hash = kernel_profiler::records::kStringHash<C, cs...>;
    return string;
}

// Appends _s to a literal after the argument expands, so a tool's macro can take a plain literal, a macro that expands
// to one, or #__VA_ARGS__.
#define TT_RECORD_STRING_(literal) literal##_s
#define TT_RECORD_STRING(literal) TT_RECORD_STRING_(literal)

// Names the call site where it expands. A scope that publishes when it closes takes it in its macro, because
// DevicePublish in its destructor would name the destructor's line.
#define DevicePublishSite                                                              \
    ::kernel_profiler::records::CallSite<::kernel_profiler::records::CallSiteLocation< \
        ::kernel_profiler::records::function_name_size(__PRETTY_FUNCTION__)>{          \
        TT_RECORD_STRING(__FILE__), __LINE__, __PRETTY_FUNCTION__}>
#define DevicePublish DevicePublishSite::template publish
