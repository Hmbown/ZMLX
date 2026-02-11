// TEMPLATE_ID: t0_basic
// OP: rmsnorm
//---HEADER---
#include <metal_stdlib>
using namespace metal;
//---BODY---
constexpr uint TG_SIZE = {{TG_SIZE}};
constexpr uint VEC = {{VEC}};
constexpr uint UNROLL = {{UNROLL}};

#if {{FAST_MATH}}
inline float inv_sqrt(float x) { return rsqrt(x); }
#else
inline float inv_sqrt(float x) { return 1.0f / sqrt(x); }
#endif

uint tid = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;

uint H = (uint)x_shape[2];
uint base = row * H;

threadgroup float partial[TG_SIZE];

float sumsq = 0.0f;

// Accumulate sumsq
for (uint col0 = tid; col0 < H; col0 += TG_SIZE * UNROLL) {
    #pragma unroll
    for (uint u = 0; u < UNROLL; ++u) {
        uint col = col0 + u * TG_SIZE;
        if (col < H) {
            uint elem = base + col;
            uint loc = elem_to_loc(elem, x_shape, x_strides, x_ndim);
            float xv = (float)x[loc];
            sumsq += xv * xv;
        }
    }
}

partial[tid] = sumsq;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Reduce within threadgroup (assumes TG_SIZE is power of 2)
for (uint stride = TG_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        partial[tid] += partial[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float inv = 0.0f;
if (tid == 0) {
    float mean = partial[0] / max(1.0f, (float)H);
    float e = eps[0];
    inv = inv_sqrt(mean + e);
    partial[0] = inv;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
inv = partial[0];

// Write outputs
for (uint col0 = tid; col0 < H; col0 += TG_SIZE * UNROLL) {
    #pragma unroll
    for (uint u = 0; u < UNROLL; ++u) {
        uint col = col0 + u * TG_SIZE;
        if (col < H) {
            uint elem = base + col;
            uint loc = elem_to_loc(elem, x_shape, x_strides, x_ndim);
            float xv = (float)x[loc];
            float wv = (float)w[col];
            float outv = xv * inv * wv;
            {{COMPILE_ERROR_SNIPPET}}
            #if {{INJECT_INCORRECT}}
            outv += 0.1f;
            #endif
            y[elem] = ({{OUT_TYPE}})outv;
        }
    }
}
