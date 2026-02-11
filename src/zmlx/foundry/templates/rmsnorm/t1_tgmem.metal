// TEMPLATE_ID: t1_tgmem
// OP: rmsnorm
//---HEADER---
#include <metal_stdlib>
using namespace metal;
//---BODY---
constexpr uint TG_SIZE = {{TG_SIZE}};
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

// Cache a small tile of weights in tgmem (for demonstration)
// NOTE: This is intentionally simple and may not always help.
threadgroup float w_tile[TG_SIZE];
threadgroup float partial[TG_SIZE];

float sumsq = 0.0f;
for (uint col0 = tid; col0 < H; col0 += TG_SIZE) {
    uint elem = base + col0;
    uint loc = elem_to_loc(elem, x_shape, x_strides, x_ndim);
    float xv = (float)x[loc];
    sumsq += xv * xv;
}
partial[tid] = sumsq;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Reduce
for (uint stride = TG_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float inv = 0.0f;
if (tid == 0) {
    float mean = partial[0] / max(1.0f, (float)H);
    inv = inv_sqrt(mean + eps[0]);
    partial[0] = inv;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
inv = partial[0];

// Write output, staging weight per tile
for (uint colBase = 0; colBase < H; colBase += TG_SIZE) {
    uint col = colBase + tid;
    if (col < H) {
        w_tile[tid] = (float)w[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint u = 0; u < UNROLL; ++u) {
        uint colu = colBase + tid + u * TG_SIZE;
        if (colu < H) {
            uint elem = base + colu;
            uint loc = elem_to_loc(elem, x_shape, x_strides, x_ndim);
            float xv = (float)x[loc];
            float wv = w_tile[tid]; // simplistic; same tid
            float outv = xv * inv * wv;
            {{COMPILE_ERROR_SNIPPET}}
            #if {{INJECT_INCORRECT}}
            outv -= 0.05f;
            #endif
            y[elem] = ({{OUT_TYPE}})outv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
