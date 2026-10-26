// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The link between a profiled process and the external process: a Unix socket under one abstract name per user, which
// the external process listens on, and the only thing the two processes share. Each end drops a peer that runs as
// another user. Only the profiled process sends, and it never waits for the external process. It connects when its
// capture starts, and again while it runs whenever none is connected. It sends the messages of
// protobuf/external_process.proto, one Envelope per length-prefixed frame: its CaptureDescriptor, then the call sites
// whose records go to the external process, as each image loads and before any of their records exists, then
// CaptureEnded once its relays have stopped. The external process points the devices at rings in its own memory itself,
// through the endpoints the descriptor names, and the relays write the routed records and clock sync measurements
// there.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

// protobuf/external_process.pb.h, which only the sources that build or read the messages include.
namespace tt::streaming_profiler::proto {
class CaptureDescriptor;
class FrameLayout;
class PacketFormat;
}  // namespace tt::streaming_profiler::proto

namespace tt::tt_metal::streaming_profiler::external {

namespace proto = ::tt::streaming_profiler::proto;

// The protocol version this build speaks: an external process refuses a profiled process with another.
inline constexpr uint32_t kProtocolVersion = 1;

// This build's frame and packet formats, as a capture descriptor gives them.
proto::FrameLayout this_build_frame_layout();
proto::PacketFormat this_build_packet_format();

// The profiled process's end. `detach` turns the relays' writes to the external process's rings off and returns once
// their last ones have landed, which this end does when an external process goes, so none writes to its freed rings.
class ProfiledEnd {
public:
    using Detach = std::function<void()>;
    ProfiledEnd(const proto::CaptureDescriptor& descriptor, Detach detach);
    // Closes the link.
    ~ProfiledEnd();
    // Connects to the external process listening on the socket name, and keeps trying while none is connected. Called
    // once the relays run. Records published before the external process points the relays at its rings stay in this
    // process.
    void start();
    // Tells the external process the capture has ended, once every relay has stopped and its last write has landed.
    void end_capture();
    ProfiledEnd(const ProfiledEnd&) = delete;
    ProfiledEnd& operator=(const ProfiledEnd&) = delete;

private:
    void run();
    void serve(int fd);
    // Sends the external process the call sites for it loaded since the last push, under send_mu_; false when the link
    // failed.
    bool push_call_sites();

    const std::vector<uint8_t> descriptor_;  // its Envelope, serialized
    const Detach detach_;

    std::mutex mu_;
    std::condition_variable stop_cv_;
    bool stop_ = false;
    // Every write to the link, from its thread or from a thread loading an image.
    std::mutex send_mu_;
    int fd_ = -1;  // the connected external process's socket, -1 while none is
    size_t call_sites_pushed_ = 0;
    bool capture_ended_ = false;

    std::thread thread_;
};

// The external process's socket, which profiled processes connect to.
class ExternalListener {
public:
    ExternalListener();
    ~ExternalListener();
    ExternalListener(const ExternalListener&) = delete;
    ExternalListener& operator=(const ExternalListener&) = delete;

    // Listens on the socket name; false while another process holds it.
    bool listen();
    // Waits for the next profiled process of this user and returns its connection, or -1 when the socket fails.
    int accept() const;

private:
    int fd_ = -1;
};

// The external process's end of one profiled process's link. Once started, a thread of its own reads what the Metal
// process sends until it closes the link.
class ExternalEnd {
public:
    // Reads the CaptureDescriptor from `fd`, which it then owns. descriptor() is null when the profiled process
    // closed first, sent none in time, or speaks another protocol version.
    explicit ExternalEnd(int fd);
    ~ExternalEnd();
    ExternalEnd(const ExternalEnd&) = delete;
    ExternalEnd& operator=(const ExternalEnd&) = delete;

    const proto::CaptureDescriptor* descriptor() const { return descriptor_.get(); }
    // The profiled process's pid, which messages name it by.
    int pid() const { return pid_; }
    // Starts the link's thread, which reads the call sites the profiled process pushes.
    void start();
    // Whether the profiled process has closed the link. Callable from any thread.
    bool closed() const { return closed_.load(std::memory_order_acquire); }
    // Whether the profiled process ended its capture before it closed the link, which a process that died doesn't. Set
    // before closed().
    bool capture_ended() const { return capture_ended_.load(std::memory_order_acquire); }

    // Ends the link, which has the profiled process turn its relays' writes off.
    void drop();
    // The id here of the call site the profiled process knows by `id`, which it pushed before any of the site's records
    // existed: waits for the link's thread to read it, and is nullopt when the link ended without it.
    std::optional<uint32_t> local_id(uint32_t id);

private:
    void read_thread();

    const int fd_;
    int pid_ = 0;
    std::unique_ptr<proto::CaptureDescriptor> descriptor_;
    std::atomic<bool> closed_{false};
    std::atomic<bool> capture_ended_{false};
    std::mutex sites_mu_;
    std::condition_variable sites_cv_;
    std::unordered_map<uint32_t, uint32_t> sites_;  // the profiled process's call-site ids, to this process's
    std::thread thread_;
};

}  // namespace tt::tt_metal::streaming_profiler::external
