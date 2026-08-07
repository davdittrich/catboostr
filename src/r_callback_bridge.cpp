// Implementation of the fork-owned R-callback bridge (catboost-8z4.93, P6.2).
// See r_callback_bridge.h for the design rationale.

// Standard library headers first, before any R header: <chrono>/<locale>
// pull in libstdc++'s codecvt, whose length() member collides with R.h's
// `#define length(x) Rf_length(x)` remap if R.h is included first
// (catboost-8z4.88 finding).
#include "r_callback_bridge.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include <R.h>
#include <Rinternals.h>

#include "catboostr.h"

namespace NCatboostR {

namespace {

// The bridge whose Run() is currently on the stack, plus the thread that
// called it. Both are written once by Run() on R's main thread before the
// background thread starts and cleared after it is joined, so no other
// thread ever observes them changing.
std::atomic<TRCallbackBridge*> g_activeBridge{nullptr};
std::thread::id g_mainThreadId;

// How long the drain loop is willing to sit on the condition variable before
// coming up for air to poll for a user interrupt. Bounded wait, not an
// indefinite one -- Ctrl-C stays responsive within this many milliseconds.
constexpr int kDrainPollMs = 100;

void CheckInterruptFn(void*) {
    R_CheckUserInterrupt();
}

// R_CheckUserInterrupt() longjmps out of the current context when an
// interrupt is pending, which would skip the background thread's join and
// leave it running against a dead R stack. R_ToplevelExec catches that
// longjmp and reports it as a return value instead.
bool PendingInterrupt() {
    return R_ToplevelExec(CheckInterruptFn, nullptr) == FALSE;
}

}  // namespace

TRCallbackBridge* TRCallbackBridge::Active() {
    return g_activeBridge.load(std::memory_order_acquire);
}

TRCallbackBridge::~TRCallbackBridge() {
    if (g_activeBridge.load() == this) {
        g_activeBridge.store(nullptr);
    }
}

void TRCallbackBridge::EnterWork() {
    const int now = ActiveWorkers_.fetch_add(1) + 1;
    int seen = MaxActiveWorkers_.load();
    while (now > seen && !MaxActiveWorkers_.compare_exchange_weak(seen, now)) {
    }
}

void TRCallbackBridge::LeaveWork() {
    ActiveWorkers_.fetch_sub(1);
}

void TRCallbackBridge::Call(const std::function<void()>& action) {
    TRequest req;
    req.Action = &action;

    {
        std::lock_guard<std::mutex> lk(QueueMutex_);
        // Closed_ means the drain loop is gone (the background thread died or
        // the user interrupted): waiting would hang forever, so unwind now.
        if (Closed_ || Interrupted_.load()) {
            throw TRBridgeInterrupted();
        }
        Queue_.push_back(&req);
    }
    QueueCv_.notify_one();

    // Not counted as an active worker while parked on the queue -- that is
    // exactly the distinction catboost-8z4.91 measures.
    LeaveWork();
    {
        std::unique_lock<std::mutex> lk(req.M);
        req.Cv.wait(lk, [&req] { return req.Done; });
    }
    EnterWork();

    if (req.Error) {
        std::rethrow_exception(req.Error);
    }
}

void TRCallbackBridge::Log(const char* str, size_t len) {
    TRequest* req = new TRequest();
    req->Owned = true;
    req->LogLine.assign(str, len);

    {
        std::lock_guard<std::mutex> lk(QueueMutex_);
        if (Closed_) {
            // Nobody left to print it. Dropping a log line is the correct
            // failure mode here; the pending error is what matters.
            delete req;
            return;
        }
        Queue_.push_back(req);
    }
    QueueCv_.notify_one();
}

void TRCallbackBridge::ServiceRequest(TRequest* req) {
    if (req->Action == nullptr) {
        Rprintf("%s", req->LogLine.c_str());
        if (req->Owned) {
            delete req;
        }
        return;
    }

    std::exception_ptr err;
    if (Interrupted_.load()) {
        err = std::make_exception_ptr(TRBridgeInterrupted());
    } else {
        try {
            (*req->Action)();
        } catch (...) {
            err = std::current_exception();
        }
    }

    {
        std::lock_guard<std::mutex> lk(req->M);
        req->Error = err;
        req->Done = true;
        // Notify while still holding req->M: TRequest lives on the waiting
        // thread's stack, and that thread cannot leave wait() (and destroy it)
        // before it reacquires this mutex. Notifying after unlocking would
        // race with the request's destruction.
        req->Cv.notify_one();
    }
}

void TRCallbackBridge::DrainLoop() {
    auto lastPoll = std::chrono::steady_clock::now();

    for (;;) {
        TRequest* req = nullptr;
        {
            std::unique_lock<std::mutex> lk(QueueMutex_);
            QueueCv_.wait_for(lk, std::chrono::milliseconds(kDrainPollMs),
                              [this] { return !Queue_.empty() || Finished_; });
            if (!Queue_.empty()) {
                req = Queue_.front();
                Queue_.pop_front();
            } else if (Finished_) {
                // Producer gone and the queue is drained. Refuse further
                // requests under the same lock so a late producer throws
                // instead of blocking on a queue nobody reads.
                Closed_ = true;
                break;
            }
        }

        if (req != nullptr) {
            ServiceRequest(req);
        }

        // Bounded-interval interrupt poll, whether or not requests are
        // flowing: a busy queue must not starve Ctrl-C.
        const auto now = std::chrono::steady_clock::now();
        if (now - lastPoll >= std::chrono::milliseconds(kDrainPollMs)) {
            lastPoll = now;
            InterruptPolls_.fetch_add(1);
            if (!Interrupted_.load() && PendingInterrupt()) {
                // Best effort: CatBoost offers no cancellation handle, so the
                // signal is delivered lazily -- every subsequent Call() throws
                // TRBridgeInterrupted, unwinding the training stack. Phases
                // that never call back finish first, exactly as they do today
                // (a .Call into TrainModel is not interruptible at all now).
                Interrupted_.store(true);
                QueueCv_.notify_all();
            }
        }
    }
}

void TRCallbackBridge::Run(const std::function<void()>& work) {
    if (g_activeBridge.load() != nullptr) {
        throw std::runtime_error("catboost: R callback bridge is already running (nested use is not supported)");
    }

    {
        std::lock_guard<std::mutex> lk(QueueMutex_);
        Queue_.clear();
        Finished_ = false;
        Closed_ = false;
    }
    Interrupted_.store(false);
    WorkError_ = nullptr;
    ActiveWorkers_.store(0);
    MaxActiveWorkers_.store(0);
    InterruptPolls_.store(0);

    g_mainThreadId = std::this_thread::get_id();
    g_activeBridge.store(this, std::memory_order_release);

    std::thread background([this, &work] {
        try {
            work();
        } catch (...) {
            // A C++ throw cannot cross a thread boundary; capture it here and
            // rethrow below, on the main thread, inside R_API_BEGIN's try.
            WorkError_ = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lk(QueueMutex_);
            Finished_ = true;
        }
        QueueCv_.notify_all();
    });

    DrainLoop();
    background.join();

    g_activeBridge.store(nullptr, std::memory_order_release);

    if (WorkError_) {
        std::exception_ptr err = WorkError_;
        WorkError_ = nullptr;
        std::rethrow_exception(err);
    }
    if (Interrupted_.load()) {
        throw TRBridgeInterrupted();
    }
}

void LogFromAnyThread(const char* str, size_t len) {
    TRCallbackBridge* bridge = TRCallbackBridge::Active();
    if (bridge == nullptr || std::this_thread::get_id() == g_mainThreadId) {
        // No bridge running (every entry point except the callback-enabled
        // ones), or we already are the main thread: print directly, exactly as
        // this package did before the bridge existed.
        Rprintf("%.*s", static_cast<int>(len), str);
        return;
    }
    bridge->Log(str, len);
}

}  // namespace NCatboostR


// ---------------------------------------------------------------------------
// Smoke test entry point (catboost-8z4.93 step 5; also the hook
// catboost-8z4.91 uses to read the active-worker instrumentation).
//
// Deliberately exercises the bridge on its own, with plain std::threads
// standing in for TBB workers -- nothing here touches CatBoost, so it fails
// loudly on a bridge regression and cannot be masked by a training bug.
// ---------------------------------------------------------------------------
namespace {

// Evaluates closure(x) on R's main thread. Runs inside a queued request, so
// every R API call below happens on the main thread by construction.
double EvalClosureOnMainThread(SEXP closure, double x) {
    SEXP arg = PROTECT(Rf_ScalarReal(x));
    SEXP call = PROTECT(Rf_lang2(closure, arg));
    int errorOccurred = 0;
    SEXP result = R_tryEval(call, R_GlobalEnv, &errorOccurred);
    if (errorOccurred) {
        UNPROTECT(2);
        throw std::runtime_error("catboost: error in R callback");
    }
    const double value = Rf_asReal(result);
    UNPROTECT(2);
    return value;
}

}  // namespace

extern "C" SEXP CatBoostRCallbackBridgeSelfTest_R(
    SEXP modeParam,
    SEXP closureParam,
    SEXP nThreadsParam,
    SEXP nItemsParam
) {
    const char* mode = CHAR(STRING_ELT(modeParam, 0));
    const int nThreads = Rf_asInteger(nThreadsParam);
    const int nItems = Rf_asInteger(nItemsParam);

    // Workers write straight into this vector's payload -- plain double
    // memory, no R API call off the main thread.
    SEXP resultsVec = PROTECT(Rf_allocVector(REALSXP, nItems));
    double* results = REAL(resultsVec);
    for (int i = 0; i < nItems; ++i) {
        results[i] = NA_REAL;
    }

    char errorMessage[512];
    errorMessage[0] = '\0';
    int maxActiveWorkers = 0;
    int interruptPolls = 0;
    int activeWorkers = 0;

    // Inner scope: every C++ object with a destructor must be gone before the
    // Rf_error() longjmp below.
    {
        NCatboostR::TRCallbackBridge bridge;
        const bool logMode = std::strcmp(mode, "log") == 0;
        const bool throwMode = std::strcmp(mode, "throw") == 0;
        const bool orphanMode = std::strcmp(mode, "orphan") == 0;

        // "orphan": the background thread dies while a worker is still going
        // to want the queue. The drain loop must have closed the queue, so
        // that worker's Call() throws rather than blocking forever.
        std::thread orphan;
        std::atomic<int> orphanCallThrew{-1};

        try {
            bridge.Run([&] {
                if (std::strcmp(mode, "idle") == 0) {
                    // No requests at all: the drain loop sits on an EMPTY
                    // queue for this long. Its interrupt-poll count afterwards
                    // is what distinguishes a bounded wait_for() from an
                    // indefinite wait() -- the latter would poll zero times.
                    std::this_thread::sleep_for(std::chrono::milliseconds(nItems));
                    return;
                }

                if (orphanMode) {
                    orphan = std::thread([&] {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        try {
                            bridge.Call([] {});
                            orphanCallThrew.store(0);
                        } catch (const NCatboostR::TRBridgeInterrupted&) {
                            orphanCallThrew.store(1);
                        }
                    });
                    throw std::runtime_error("catboost: boom from the background thread");
                }

                // A worker's exception must not escape its std::thread
                // (that is std::terminate). TBB captures body exceptions and
                // rethrows them on the thread that called parallel_for; these
                // stand-in workers do the same by hand.
                std::mutex workerErrorMutex;
                std::exception_ptr workerError;

                std::vector<std::thread> workers;
                for (int t = 0; t < nThreads; ++t) {
                    workers.emplace_back([&, t] {
                        NCatboostR::TActiveWorkScope scope(bridge);
                        try {
                            for (int i = t; i < nItems; i += nThreads) {
                                if (logMode) {
                                    const std::string line =
                                        "bridge log line " + std::to_string(i) + "\n";
                                    // Same path CatBoost's own logger takes.
                                    NCatboostR::LogFromAnyThread(line.data(), line.size());
                                    results[i] = i;
                                    continue;
                                }
                                // Stand-in for non-callback training work, so
                                // MaxActiveWorkers() has something to observe.
                                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                                double value = 0.0;
                                bridge.Call([&] { value = EvalClosureOnMainThread(closureParam, i); });
                                results[i] = value;
                            }
                        } catch (...) {
                            std::lock_guard<std::mutex> lk(workerErrorMutex);
                            if (!workerError) {
                                workerError = std::current_exception();
                            }
                        }
                    });
                }
                for (auto& w : workers) {
                    w.join();
                }
                if (workerError) {
                    std::rethrow_exception(workerError);
                }
                if (throwMode) {
                    throw std::runtime_error("catboost: boom from the background thread");
                }
            });
        } catch (std::exception& e) {
            std::snprintf(errorMessage, sizeof(errorMessage), "%s", e.what());
        }

        if (orphan.joinable()) {
            // Hangs here forever if the queue-closed guard is missing -- that
            // is precisely what this mode tests.
            orphan.join();
            std::snprintf(errorMessage + std::strlen(errorMessage),
                          sizeof(errorMessage) - std::strlen(errorMessage),
                          " [orphan_call_threw=%d]", orphanCallThrew.load());
        }

        maxActiveWorkers = bridge.MaxActiveWorkers();
        interruptPolls = static_cast<int>(bridge.InterruptPolls());
        activeWorkers = bridge.ActiveWorkers();
    }

    if (errorMessage[0] != '\0') {
        UNPROTECT(1);
        Rf_error("%s", errorMessage);
    }

    SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
    SET_VECTOR_ELT(out, 0, resultsVec);
    SET_VECTOR_ELT(out, 1, Rf_ScalarInteger(maxActiveWorkers));
    SET_VECTOR_ELT(out, 2, Rf_ScalarInteger(interruptPolls));
    SET_VECTOR_ELT(out, 3, Rf_ScalarInteger(activeWorkers));

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_STRING_ELT(names, 0, Rf_mkChar("results"));
    SET_STRING_ELT(names, 1, Rf_mkChar("max_active_workers"));
    SET_STRING_ELT(names, 2, Rf_mkChar("interrupt_polls"));
    SET_STRING_ELT(names, 3, Rf_mkChar("active_workers"));
    Rf_setAttrib(out, R_NamesSymbol, names);

    UNPROTECT(3);
    return out;
}
