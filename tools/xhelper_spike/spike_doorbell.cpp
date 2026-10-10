// tools/xhelper_spike/spike_doorbell.cpp - the host-only half of the Phase 1 spike.
//
// spike_cuda.cu and spike_hip.cpp need nvcc and hipcc and the two cards.  This needs neither: it is the same memfd,
// the same mapping layout and the same two doorbell words, forked into a CPU-only child that answers pings.  Run it
// on any Linux box (build with plain g++) to check the parts of Gate 1 that are not about GPUs: that memfd + mmap +
// MAP_SHARED works, that a futex without FUTEX_PRIVATE_FLAG really wakes the other process, and that a wait returns
// instead of hanging when the partner is gone.  It is also the fastest way to see whether the spin/futex choice in
// Phase 2's waits is sane.  It measures no card, so it proves nothing about Gate 1's latency numbers.
//
// Build: g++ -O2 -std=c++20 -Wall -Wextra spike_doorbell.cpp -o strata-xh-doorbell
// Run:   ./strata-xh-doorbell [--rounds N] [--spin-us N]
//
// Read the "both futex" row like this: if the cross-process wake worked, its p50 is a few microseconds.  If the wake
// never arrives, the waiter only returns from its 2 ms futex timeout and p50 jumps to about 2000 us.  That is a
// FUTEX_PRIVATE_FLAG bug, and it is exactly the mistake to make here.

#include "spike_layout.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace xhelper_spike;

static uint64_t pct(const std::vector<uint64_t>& sorted, double q) {
    if (sorted.empty()) return 0;
    return sorted[(size_t) (q * (double) (sorted.size() - 1) + 0.5)];
}

int main(int argc, char** argv) {
    uint32_t rounds = 10000;
    uint32_t spin_us = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rounds" && i + 1 < argc) rounds = (uint32_t) std::atoi(argv[++i]);
        else if (a == "--spin-us" && i + 1 < argc) spin_us = (uint32_t) std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: %s [--rounds N] [--spin-us N]\n", argv[0]);
            return 2;
        }
    }

    const int fd = (int) syscall(SYS_memfd_create, "strata-xh-doorbell", 0);
    if (fd < 0) { std::fprintf(stderr, "memfd_create: %s\n", std::strerror(errno)); return 1; }
    if (ftruncate(fd, (off_t) kTotalBytes) != 0) { std::fprintf(stderr, "ftruncate: %s\n", std::strerror(errno)); return 1; }
    void* base = mmap(nullptr, kTotalBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { std::fprintf(stderr, "mmap: %s\n", std::strerror(errno)); return 1; }
    std::memset(base, 0, kTotalBytes);
    Ctl* ctl = reinterpret_cast<Ctl*> (reinterpret_cast<char*> (base) + ctl_off());
    ctl->magic = kMagic;
    ctl->version = kVersion;
    ctl->total_bytes = kTotalBytes;
    ctl->a_off = a_off();
    ctl->a_bytes = kRegionBytes;
    ctl->b_off = b_off();
    ctl->b_bytes = kRegionBytes;
    ctl->in_off = in_off();
    ctl->in_bytes = kWorkInBytes;
    ctl->out_off = out_off();
    ctl->out_bytes = kWorkOutBytes;
    ctl->report_off = report_off();
    ctl->rounds = rounds;
    ctl->spin_us = spin_us;
    // Touch every page above the control block so the ping-pong never pays for a page fault.  Touching the control
    // block itself would wipe the magic and the offsets just written.
    for (uint64_t off = kCtlBytes; off < kTotalBytes; off += 4096) reinterpret_cast<char*> (base)[off] = 1;

    fflush(nullptr);
    const pid_t kid = fork();
    if (kid < 0) { std::fprintf(stderr, "fork: %s\n", std::strerror(errno)); return 1; }
    if (kid == 0) {
        void* cbase = mmap(nullptr, kTotalBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (cbase == MAP_FAILED) _exit(1);
        Ctl* c = reinterpret_cast<Ctl*> (reinterpret_cast<char*> (cbase) + ctl_off());
        if (c->magic != kMagic) _exit(2);
        c->state.store(SPIKE_READY);
        uint32_t last = 0;
        for (;;) {
            const uint32_t v = c->variant.load();
            const bool futex = v != SPIKE_BOTH_SPIN;
            const uint32_t spin = v == SPIKE_BOTH_SLEEP ? 0u : c->spin_us;
            const uint64_t deadline = spike_now_ns() + 60ull * 1000000000ull;
            const uint32_t s = spike_wait(c->seq, spin, futex, deadline, [&](uint32_t x) { return x != last; });
            if (s == 0) { if (getppid() == 1) _exit(0); continue; }
            last = s;
            if (c->cmd.load() == SPIKE_QUIT) _exit(0);
            c->done.store(s);
            spike_wake(c->done);
        }
    }

    for (uint32_t waited = 0; waited < 10000 && ctl->state.load() != SPIKE_READY; waited += 100) usleep(100);
    std::printf("=== doorbell check (no GPU, %llu MiB memfd, spin_us %u) ===\n",
                (unsigned long long) (kTotalBytes / kMiB), spin_us);
    if (ctl->state.load() != SPIKE_READY) { std::printf("the child never mapped the region\n"); kill(kid, SIGKILL); return 1; }

    const char* names[4] = {"both spin            ", "child futex, parent spin", "both futex           ",
                            "both sleep, no spin  "};
    uint64_t spin_p50 = 0;
    bool futex_ok = false;
    uint32_t total_timeouts = 0;
    for (uint32_t v = 0; v < 4; ++v) {
        ctl->variant.store(v);
        // With no spin budget on either side both processes are asleep before the other one answers, so this row is
        // the wake itself: 1 us-ish if FUTEX_WAKE crosses the process boundary, about 2 ms if it does not.
        const uint32_t pspin = v == SPIKE_BOTH_SLEEP ? 0u : spin_us;
        const bool pfutex = v == SPIKE_BOTH_FUTEX || v == SPIKE_BOTH_SLEEP;
        std::vector<uint64_t> us;
        us.reserve(rounds);
        uint32_t timeouts = 0;
        for (uint32_t r = 0; r < rounds; ++r) {
            const uint64_t t0 = spike_now_ns();
            const uint32_t seq = ctl->seq.load() + 1;
            ctl->cmd.store(SPIKE_PING);
            ctl->seq.store(seq);
            spike_wake(ctl->seq);
            const uint32_t got = spike_wait(ctl->done, pspin, pfutex,
                                            spike_now_ns() + 5000000ull, [&](uint32_t x) { return x == seq; });
            if (got != seq) { ++timeouts; continue; }
            us.push_back(spike_now_ns() - t0);   // nanoseconds: on one socket these round trips are under a microsecond
        }
        std::sort(us.begin(), us.end());
        const uint64_t p50 = pct(us, 0.5), p99 = pct(us, 0.99), p999 = pct(us, 0.999);
        if (v == SPIKE_BOTH_SPIN) spin_p50 = p50;
        std::printf("  %s p50 %8.2f us  p99 %8.2f us  p999 %8.2f us  timeouts %u\n", names[v], (double) p50 / 1000.0,
                    (double) p99 / 1000.0, (double) p999 / 1000.0, timeouts);
        total_timeouts += timeouts;
        if (v == SPIKE_BOTH_SLEEP) {
            // A wake that never arrives shows up as the waiter's own 2 ms futex timeout.
            futex_ok = p50 < 2000000ull;
            std::printf("      cross-process wake : %s\n",
                        futex_ok ? "works (a futex wake, not a timeout)"
                                 : "NOT working - p50 looks like the 2 ms futex timeout, check for FUTEX_PRIVATE_FLAG");
        }
    }

    // A partner that is gone must not hang the engine.
    ctl->variant.store(SPIKE_BOTH_SPIN);
    kill(kid, SIGKILL);
    int status = 0;
    waitpid(kid, &status, 0);
    const uint64_t t0 = spike_now_ns();
    const uint32_t seq = ctl->seq.load() + 1;
    ctl->cmd.store(SPIKE_PING);
    ctl->seq.store(seq);
    spike_wake(ctl->seq);
    const uint32_t got = spike_wait(ctl->done, spin_us, false, spike_now_ns() + 20000000ull,
                                    [&](uint32_t x) { return x == seq; });
    std::printf("  after SIGKILL          : %s in %.1f ms (the 20 ms patience the engine will use)\n",
                got == 0 ? "the wait gave up, as it must" : "answered?!", (double) (spike_now_ns() - t0) / 1000000.0);

    const bool ok = futex_ok && total_timeouts == 0 && got == 0;
    std::printf("\n  verdict: memfd + MAP_SHARED + cross-process futex %s; spin p50 %.1f us on this CPU, no GPU in the\n"
                "           loop, so this says nothing about Gate 1's 150 us target\n",
                ok ? "works" : "is broken", (double) spin_p50 / 1000.0);
    return ok ? 0 : 1;
}
