// TEMPLATE_ID: t2_row_tile
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

// Cache one p-tile of (token_id, weight) in threadgroup memory.
threadgroup int tg_ids[TG_SIZE];
threadgroup float tg_w[TG_SIZE];

for (uint col0 = tid; col0 < H; col0 += TG_SIZE * UNROLL) {
    #pragma unroll
    for (uint u = 0; u < UNROLL; ++u) {
        uint col = col0 + u * TG_SIZE;
        if (col < H) {
            float acc = 0.0f;

            for (uint p0 = 0; p0 < P; p0 += TG_SIZE) {
                uint p_load = p0 + tid;
                if (p_load < P) {
                    uint id_loc = elem_to_loc(
                        p_load, packed_token_ids_shape, packed_token_ids_strides, packed_token_ids_ndim
                    );
                    uint w_loc = elem_to_loc(
                        p_load, packed_weights_shape, packed_weights_strides, packed_weights_ndim
                    );
                    tg_ids[tid] = (int)packed_token_ids[id_loc];
                    tg_w[tid] = (float)packed_weights[w_loc];
                } else {
                    tg_ids[tid] = -1;
                    tg_w[tid] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                #pragma unroll
                for (uint j = 0; j < TG_SIZE; ++j) {
                    uint p = p0 + j;
                    if (p < P && tg_ids[j] == (int)row) {
                        uint ey_elem = p * H + col;
                        uint ey_loc = elem_to_loc(
                            ey_elem, expert_y_shape, expert_y_strides, expert_y_ndim
                        );
                        float ey = (float)expert_y[ey_loc];
                        float w = tg_w[j];
                        #if {{FAST_MATH}}
                        acc = fma(ey, w, acc);
                        #else
                        acc += ey * w;
                        #endif
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            {{COMPILE_ERROR_SNIPPET}}
            #if {{INJECT_INCORRECT}}
            acc -= 0.03f;
            #endif
            y[row * H + col] = ({{OUT_TYPE}})acc;
        }
    }
}
