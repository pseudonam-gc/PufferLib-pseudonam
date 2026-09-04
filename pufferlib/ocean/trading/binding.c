#include "trading.h"
#define Env Trading
#include "../env_binding.h"

// MAX_OPTIONS/OPTION_FEATURES/etc. are compile-time constants, and
// per-episode state (underlying, heap, cash) is set up by c_reset (called
// separately via vec_reset), not here -- these are the real
// runtime-configurable parameters.
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->min_tick_spend = unpack(kwargs, "min_tick_spend");
    env->market_noise_lower = unpack(kwargs, "market_noise_lower");
    env->market_noise_upper = unpack(kwargs, "market_noise_upper");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
