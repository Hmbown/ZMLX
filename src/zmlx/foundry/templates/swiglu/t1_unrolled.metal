// TEMPLATE_ID: t1_unrolled
// OP: swiglu
//---HEADER---
#include <metal_stdlib>
using namespace metal;
//---BODY---
constexpr uint TG_SIZE = {{TG_SIZE}};
constexpr uint UNROLL = {{UNROLL}};
constexpr uint VEC = {{VEC}};

inline float sigmoid_exact(float x) { return 1.0f / (1.0f + exp(-x)); }

uint tid = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;

uint H2 = (uint)x_shape[2];
uint H = H2 / 2;
uint base_in = row * H2;
uint base_out = row * H;

// Vector-ish inner loop: process VEC contiguous cols per iteration when possible
for (uint col0 = tid * VEC; col0 < H; col0 += TG_SIZE * VEC * UNROLL) {
    #pragma unroll
    for (uint u = 0; u < UNROLL; ++u) {
        uint colu = col0 + u * TG_SIZE * VEC;
        #pragma unroll
        for (uint v = 0; v < VEC; ++v) {
            uint col = colu + v;
            if (col < H) {
                uint elem0 = base_in + col;
                uint elem1 = base_in + (H + col);
                uint loc0 = elem_to_loc(elem0, x_shape, x_strides, x_ndim);
                uint loc1 = elem_to_loc(elem1, x_shape, x_strides, x_ndim);
                float a = (float)x[loc0];
                float b = (float)x[loc1];
                float s = sigmoid_exact(a);
                float outv = (a * s) * b;
                {{COMPILE_ERROR_SNIPPET}}
                #if {{INJECT_INCORRECT}}
                outv += 0.02f;
                #endif
                y[base_out + col] = ({{OUT_TYPE}})outv;
            }
        }
    }
}
