/// Catalog activation kernels — Zig port of `src/zmlx/kernels/activations.py`.
///
/// Each function returns a `UnaryOp` backed by a cached Metal kernel.
/// Forward-only (VJP support requires MLX-C custom_function, not yet available).
const std = @import("std");
const Allocator = std.mem.Allocator;
const elementwise = @import("../elementwise.zig");
const metal = @import("../metal.zig");

const Array = metal.Array;
const Dtype = metal.Dtype;
const UnaryOp = elementwise.UnaryOp;

/// SiLU (Swish): x * sigmoid(x)
pub fn silu(allocator: Allocator, compute_dtype: Dtype) !UnaryOp {
    return elementwise.unary(allocator, "kk_silu(x)", "kk_silu", compute_dtype, .{});
}

/// GELU with tanh approximation (used in GPT-2, LLaMA, etc.)
pub fn geluTanh(allocator: Allocator, compute_dtype: Dtype) !UnaryOp {
    return elementwise.unary(allocator, "kk_gelu_tanh(x)", "kk_gelu_tanh", compute_dtype, .{});
}

/// ReLU: max(x, 0)
pub fn relu(allocator: Allocator, compute_dtype: Dtype) !UnaryOp {
    return elementwise.unary(allocator, "metal::max(x, (T)0)", "kk_relu", compute_dtype, .{});
}

/// Sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid(allocator: Allocator, compute_dtype: Dtype) !UnaryOp {
    return elementwise.unary(allocator, "kk_sigmoid(x)", "kk_sigmoid", compute_dtype, .{});
}

/// Exponential: exp(x)
pub fn exp(allocator: Allocator, compute_dtype: Dtype) !UnaryOp {
    return elementwise.unary(allocator, "metal::exp(x)", "kk_exp", compute_dtype, .{});
}

/// Hyperbolic tangent: tanh(x)
pub fn tanh(allocator: Allocator, compute_dtype: Dtype) !UnaryOp {
    return elementwise.unary(allocator, "metal::tanh(x)", "kk_tanh", compute_dtype, .{});
}

/// Softplus: log(exp(x) + 1)
pub fn softplus(allocator: Allocator, compute_dtype: Dtype) !UnaryOp {
    return elementwise.unary(
        allocator,
        "metal::log(metal::exp(x) + (T)1.0)",
        "kk_softplus",
        compute_dtype,
        .{},
    );
}

/// Mish: x * tanh(softplus(x))
pub fn mish(allocator: Allocator, compute_dtype: Dtype) !UnaryOp {
    return elementwise.unary(
        allocator,
        "x * metal::tanh(metal::log(metal::exp(x) + (T)1.0))",
        "kk_mish",
        compute_dtype,
        .{},
    );
}

// ---------------------------------------------------------------------------
// Tests (GPU — require Apple Silicon with Metal)
// ---------------------------------------------------------------------------

test "silu" {
    // Use page_allocator because kernelFromSpec caches globally (outlives test).
    const allocator = std.heap.page_allocator;
    const op = try silu(allocator, .float32);

    // silu(x) = x * sigmoid(x)
    // silu(-1) ≈ -0.2689, silu(0) = 0, silu(1) ≈ 0.7311, silu(2) ≈ 1.7616
    const data = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const shape = [_]i64{4};
    const a = Array.fromFloat32(&data, &shape);
    defer a.deinit();

    const result = try op.call(a);
    defer result.deinit();

    result.eval();
    const out = result.dataFloat32();
    try std.testing.expectApproxEqAbs(@as(f32, -0.2689), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), out[2], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.7616), out[3], 1e-3);
}

test "relu" {
    const allocator = std.heap.page_allocator;
    const op = try relu(allocator, .float32);

    const data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    const shape = [_]i64{5};
    const a = Array.fromFloat32(&data, &shape);
    defer a.deinit();

    const result = try op.call(a);
    defer result.deinit();

    result.eval();
    const out = result.dataFloat32();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[4], 1e-5);
}

test "sigmoid" {
    const allocator = std.heap.page_allocator;
    const op = try sigmoid(allocator, .float32);

    // sigmoid(0) = 0.5, sigmoid(large) ≈ 1, sigmoid(-large) ≈ 0
    const data = [_]f32{ -10.0, 0.0, 10.0 };
    const shape = [_]i64{3};
    const a = Array.fromFloat32(&data, &shape);
    defer a.deinit();

    const result = try op.call(a);
    defer result.deinit();

    result.eval();
    const out = result.dataFloat32();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[2], 1e-3);
}

test "exp" {
    const allocator = std.heap.page_allocator;
    const op = try exp(allocator, .float32);

    const data = [_]f32{ 0.0, 1.0, 2.0 };
    const shape = [_]i64{3};
    const a = Array.fromFloat32(&data, &shape);
    defer a.deinit();

    const result = try op.call(a);
    defer result.deinit();

    result.eval();
    const out = result.dataFloat32();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.71828), out[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 7.38906), out[2], 1e-3);
}
