// TEMPLATE_ID: t0_basic
// OP: swiglu
//---HEADER---
#include <metal_stdlib>
using namespace metal;
//---BODY---
constexpr uint TG_SIZE = {{TG_SIZE}};
constexpr uint UNROLL = {{UNROLL}};

inline float sigmoid_approx(float x) {
#if {{FAST_MATH}}
    float ax = fabs(x);
    float y = x / (1.0f + ax);
    return 0.5f + 0.5f * y;
#else
    return 1.0f / (1.0f + exp(-x));
#endif
}

uint tid = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;

uint H2 = (uint)x_shape[2];
uint H = H2 / 2;
uint base_in = row * H2;
uint base_out = row * H;

for (uint col0 = tid; col0 < H; col0 += TG_SIZE * UNROLL) {
    #pragma unroll
    for (uint u = 0; u < UNROLL; ++u) {
        uint col = col0 + u * TG_SIZE;
        if (col < H) {
            uint elem0 = base_in + col;
            uint elem1 = base_in + (H + col);
            uint loc0 = elem_to_loc(elem0, x_shape, x_strides, x_ndim);
            uint loc1 = elem_to_loc(elem1, x_shape, x_strides, x_ndim);
            float a = (float)x[loc0];
            float b = (float)x[loc1];
            float s = sigmoid_approx(a);
            float outv = (a * s) * b;
            {{COMPILE_ERROR_SNIPPET}}
            #if {{INJECT_INCORRECT}}
            outv *= 0.98f;
            #endif
            y[base_out + col] = ({{OUT_TYPE}})outv;
        }
    }
}
