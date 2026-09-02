// Shared data structures/algorithms for the trading env. Add future ones here.
#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// Width of one option's encoding in env->observations (see load_options/store_options below).
const int OPTION_FEATURES = 5;

// One tick = one trading day. Shared by anything converting between years
// (time_to_expiry, GBM dt) and ticks (ticks, expiry_tick).
const float TICKS_PER_YEAR = 365.0f;

// Box-Muller transform: turns two uniform samples into two N(0,1) samples
// Because we're freaking lazy we only take one, but we can get both.
float standard_normal(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f); // avoid log(0)
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// One exact step of Geometric Brownian Motion (the process Black-Scholes
// assumes the underlying follows) for price S, drift mu, volatility sigma.
// https://en.wikipedia.org/wiki/Geometric_Brownian_motion#Simulating_sample_paths
float gbm_step(float S, float mu, float sigma, float dt) {
    float z = standard_normal();
    return S * expf((mu - 0.5f*sigma*sigma)*dt + sigma*sqrtf(dt)*z);
}

// The simulated underlying asset that market options are priced off of.
typedef struct {
    float price;
    float volatility;
    float drift; // risk-free rate: used as both the GBM drift and Black-Scholes r
} Underlying;

void init_underlying(Underlying* u) {
    u->price = 100.0f;
    u->volatility = 0.2f;
    //u->drift = 0.01f;
    u->drift = 0.0f; // risk-free rate is 0 for simplicity
}

typedef enum {
    CALL = 0,
    PUT = 1
} OptionType;

// expiry_tick is only meaningful once a contract is held (an absolute tick,
// so the heap below stays ordered without updating every entry every step);
// it's not part of the observation wire format -- see load_options/store_options.
typedef struct {
    float time_to_expiry; // years until expiry, as generated/observed
    float strike_price;
    float price;
    OptionType type;
    bool mask; // false = padding slot, not a real option
    int expiry_tick;
} Option;

// env->observations is a flat float32 buffer, so Option can't be aliased onto
// it directly (type/mask aren't float-shaped) -- this converts explicitly.
// slot layout: [time_to_expiry, strike_price, price, type, mask]
// expiry_tick is never written here -- it's set once, at generation time
// (see gen_test_options), and isn't part of the observation wire format.
void store_options(float* observations, const Option* options, int count) {
    for (int i = 0; i < count; i++) {
        float* slot = observations + i*OPTION_FEATURES;
        slot[0] = options[i].time_to_expiry;
        slot[1] = options[i].strike_price;
        slot[2] = options[i].price;
        slot[3] = (float)options[i].type;
        slot[4] = options[i].mask ? 1.0f : 0.0f;
    }
}

// Generic array-backed min-heap over Option, ordered by expiry_tick.
typedef struct {
    Option* data;
    int count;
    int capacity;
} OwnedOptionHeap;

void heap_sift_up(OwnedOptionHeap* heap, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap->data[parent].expiry_tick <= heap->data[i].expiry_tick) {
            break;
        }
        Option tmp = heap->data[parent];
        heap->data[parent] = heap->data[i];
        heap->data[i] = tmp;
        i = parent;
    }
}

void heap_sift_down(OwnedOptionHeap* heap, int i) {
    while (true) {
        int left = 2*i + 1;
        int right = 2*i + 2;
        int smallest = i;
        if (left < heap->count && heap->data[left].expiry_tick < heap->data[smallest].expiry_tick) {
            smallest = left;
        }
        if (right < heap->count && heap->data[right].expiry_tick < heap->data[smallest].expiry_tick) {
            smallest = right;
        }
        if (smallest == i) {
            break;
        }
        Option tmp = heap->data[smallest];
        heap->data[smallest] = heap->data[i];
        heap->data[i] = tmp;
        i = smallest;
    }
}

// Returns false and drops the contract if the heap is at capacity.
bool heap_push(OwnedOptionHeap* heap, Option contract) {
    if (heap->count >= heap->capacity) {
        return false;
    }
    heap->data[heap->count] = contract;
    heap_sift_up(heap, heap->count);
    heap->count += 1;
    return true;
}

// Caller must check heap->count > 0 first.
Option heap_pop_min(OwnedOptionHeap* heap) {
    Option root = heap->data[0];
    heap->count -= 1;
    heap->data[0] = heap->data[heap->count];
    heap_sift_down(heap, 0);
    return root;
}

// Writes the k soonest-to-expire contracts into out (ascending expiry_tick,
// caller-allocated, size >= k, zero-initialized so unfilled slots read as
// padding), leaving the heap unchanged. Returns how many were written
// (min(k, heap->count)). Pops k then pushes them back, so it's O(k log n)
// rather than a full sort.
int heap_top_k(OwnedOptionHeap* heap, int k, Option* out) {
    int n = k < heap->count ? k : heap->count;
    for (int i = 0; i < n; i++) {
        out[i] = heap_pop_min(heap);
    }
    for (int i = 0; i < n; i++) {
        heap_push(heap, out[i]);
    }
    return n;
}
