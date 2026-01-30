/// Metal helper snippets shared across kernels.
///
/// Zig port of `src/zmlx/msl.py`.  The `DEFAULT_HEADER` string is
/// byte-identical to the Python version so that MLX's source-based
/// compilation cache is shared across both frontends.

pub const DEFAULT_HEADER: []const u8 =
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\template <typename T>
    \\inline T kk_sigmoid(T x) {
    \\    return T(1) / (T(1) + metal::exp(-x));
    \\}
    \\
    \\template <typename T>
    \\inline T kk_silu(T x) {
    \\    return x * kk_sigmoid(x);
    \\}
    \\
    \\// tanh approximation form used in many transformer implementations
    \\template <typename T>
    \\inline T kk_gelu_tanh(T x) {
    \\    const T k0 = T(0.7978845608028654);   // sqrt(2/pi)
    \\    const T k1 = T(0.044715);
    \\    T x3 = x * x * x;
    \\    return T(0.5) * x * (T(1) + metal::tanh(k0 * (x + k1 * x3)));
    \\}
    \\
    \\// erf-based GELU is also common; Metal doesn't expose erf() in all profiles,
    \\// so we default to tanh GELU. If you need erf GELU, implement a poly approximation.
    \\
;
