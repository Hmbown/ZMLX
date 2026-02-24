// TEMPLATE_ID: t0_basic
// OP: moe_combine
//---HEADER---
#include <metal_stdlib>
using namespace metal;
//---BODY---
constexpr uint TG_SIZE = {{TG_SIZE}};
constexpr uint UNROLL = {{UNROLL}};

uint tid = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;

uint H = (uint)expert_y_shape[expert_y_ndim - 1];
uint P = (uint)packed_token_ids_shape[packed_token_ids_ndim - 1];

for (uint col0 = tid; col0 < H; col0 += TG_SIZE * UNROLL) {
    #pragma unroll
    for (uint u = 0; u < UNROLL; ++u) {
        uint col = col0 + u * TG_SIZE;
        if (col < H) {
            float acc = 0.0f;
            for (uint p = 0; p < P; ++p) {
                uint id_loc = elem_to_loc(
                    p, packed_token_ids_shape, packed_token_ids_strides, packed_token_ids_ndim
                );
                uint tok = (uint)packed_token_ids[id_loc];
                if (tok == row) {
                    uint w_loc = elem_to_loc(
                        p, packed_weights_shape, packed_weights_strides, packed_weights_ndim
                    );
                    float w = (float)packed_weights[w_loc];
                    uint ey_elem = p * H + col;
                    uint ey_loc = elem_to_loc(
                        ey_elem, expert_y_shape, expert_y_strides, expert_y_ndim
                    );
                    acc += ((float)expert_y[ey_loc]) * w;
                }
            }

            {{COMPILE_ERROR_SNIPPET}}
            #if {{INJECT_INCORRECT}}
            acc += 0.05f;
            #endif
            y[row * H + col] = ({{OUT_TYPE}})acc;
        }
    }
}
