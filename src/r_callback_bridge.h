// Fork-owned R-callback bridge (catboost-8z4.93, P6.2).
//
// R's C API is only safe to call from R's own main thread. CatBoost's
// training runs its derivative/metric computations on TBB worker threads
// (NPar::TTbbLocalExecutor -> tbb::parallel_for), and its logger fires from
// whichever thread happens to be writing. To let an R closure serve as a
// custom objective/metric, this bridge inverts the usual arrangement:
//
//   * TrainModel()/CrossValidate()/... run on a background std::thread,
//   * R's real main thread (the .Call frame that started it) runs a drain
//     loop servicing queued requests,
//   * any thread that needs R marshals a request onto one shared queue and
//     blocks on its own condition variable until the main thread has run it.
//
// A queued request is either an opaque `std::function<void()>` executed on
// the main thread (the caller does its own R_tryEval / PROTECT inside it) or
// a log line to Rprintf. Nothing here knows anything about objectives or
// metrics: catboost-8z4.89/.90/.94 supply the marshaling closures.
//
// Mechanism confirmed empirically by the catboost-8z4.88 spike (background
// thread + main-thread R_tryEval queue, TBB parallelism preserved).
//
// This header deliberately includes no R headers: R.h's `#define length(x)`
// collides with libstdc++'s codecvt::length() unless standard headers are
// included first. Keeping it R-free makes include order irrelevant for
// consumers.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace NCatboostR {

// Thrown on a worker thread when the main thread can no longer service
// requests -- either the user pressed Ctrl-C or the drain loop has closed
// (the producer died). Unwinds CatBoost's stack instead of deadlocking.
class TRBridgeInterrupted : public std::runtime_error {
public:
    TRBridgeInterrupted()
        : std::runtime_error("catboost: interrupted while waiting for the R callback queue")
    {}
};

class TRCallbackBridge {
public:
    TRCallbackBridge() = default;
    ~TRCallbackBridge();

    TRCallbackBridge(const TRCallbackBridge&) = delete;
    TRCallbackBridge& operator=(const TRCallbackBridge&) = delete;

    // --- main thread only ---------------------------------------------------
    // Runs `work` on a background thread and drains R requests here until it
    // finishes. Rethrows whatever `work` threw (transported across the thread
    // boundary as a std::exception_ptr), so a training failure surfaces as a
    // normal C++ exception inside R_API_BEGIN's try block. Throws
    // TRBridgeInterrupted if the user interrupted.
    void Run(const std::function<void()>& work);

    // --- any other thread ---------------------------------------------------
    // Runs `action` on R's main thread; blocks until it has run. Exceptions
    // thrown by `action` are rethrown here, on the calling thread. Throws
    // immediately if called ON the main thread -- that would block the drain
    // loop against itself.
    void Call(const std::function<void()>& action);

    // Fire-and-forget: queue `len` bytes of log text for Rprintf on the main
    // thread. Never blocks.
    void Log(const char* str, size_t len);

    // True once the drain loop has seen a pending user interrupt. Safe to
    // call from any thread. The training-wiring tickets (catboost-8z4.89/.94)
    // are expected to feed this to CatBoost's own SetInterruptHandler()
    // (catboost/libs/helpers/interrupt.h) so training stops at its next
    // CheckInterrupted() rather than only at its next R callback.
    //
    // KNOWN DEVIATION (catboost-upw): detecting the interrupt consumes it.
    // The drain loop polls through R_ToplevelExec(R_CheckUserInterrupt), and
    // R clears R_interrupts_pending before longjmping, so nothing is left
    // pending afterwards. Ctrl-C consequently surfaces as an ordinary R error
    // ("interrupted while waiting for the R callback queue"), NOT as an
    // interrupt condition -- tryCatch(..., interrupt = ) will not fire on it.
    // Restoring true interrupt semantics needs R_interrupts_pending /
    // Rf_onintr(), which are not part of R's package API, so it is tracked
    // separately rather than papered over here.
    bool Interrupted() const { return Interrupted_.load(); }

    // --- instrumentation (catboost-8z4.91) ----------------------------------
    // Workers currently doing real work, i.e. inside a TActiveWorkScope and
    // not blocked on the callback queue.
    int ActiveWorkers() const { return ActiveWorkers_.load(); }
    int MaxActiveWorkers() const { return MaxActiveWorkers_.load(); }
    // Number of R_CheckUserInterrupt() polls the drain loop performed --
    // evidence that its wait is bounded, not indefinite.
    long InterruptPolls() const { return InterruptPolls_.load(); }

    void EnterWork();
    void LeaveWork();

    // The bridge whose Run() is currently executing, or nullptr. Used by the
    // logging hook, which is a captureless function pointer and so has no
    // other way to find it.
    static TRCallbackBridge* Active();

private:
    struct TRequest {
        const std::function<void()>* Action = nullptr;  // null => log line
        std::string LogLine;
        bool Owned = false;  // heap-allocated log requests are freed by the drain loop

        std::mutex M;
        std::condition_variable Cv;
        bool Done = false;
        std::exception_ptr Error;
    };

    void DrainLoop();
    void ServiceRequest(TRequest* req);

    std::mutex QueueMutex_;
    std::condition_variable QueueCv_;
    std::deque<TRequest*> Queue_;
    bool Finished_ = false;  // background thread returned; guarded by QueueMutex_
    bool Closed_ = false;    // drain loop gone; guarded by QueueMutex_

    std::atomic<bool> Interrupted_{false};
    std::exception_ptr WorkError_;

    std::atomic<int> ActiveWorkers_{0};
    std::atomic<int> MaxActiveWorkers_{0};
    std::atomic<long> InterruptPolls_{0};
};

// RAII: marks the calling thread as doing non-callback work. Call() suspends
// the mark for the duration of its wait, so ActiveWorkers() counts only
// threads that are not queue-blocked.
class TActiveWorkScope {
public:
    explicit TActiveWorkScope(TRCallbackBridge& bridge)
        : Bridge_(bridge)
    {
        Bridge_.EnterWork();
    }
    ~TActiveWorkScope() { Bridge_.LeaveWork(); }

    TActiveWorkScope(const TActiveWorkScope&) = delete;
    TActiveWorkScope& operator=(const TActiveWorkScope&) = delete;

private:
    TRCallbackBridge& Bridge_;
};

// Logging entry point for R_API_BEGIN()'s SetCustomLoggingFunction lambda.
// Rprintf()s directly when called on R's main thread or when no bridge is
// running (today's behaviour, unchanged); otherwise queues the line so the
// R API is still only ever touched by the main thread.
void LogFromAnyThread(const char* str, size_t len);

}  // namespace NCatboostR
