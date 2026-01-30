/// Softmax kernel — Zig port of `src/zmlx/kernels/softmax.py`.
///
/// Uses the two-pass map-reduce pattern for numerical stability.
const std = @import("std");
const Allocator = std.mem.Allocator;
const rowwise = @import("../rowwise.zig");
const metal = @import("../metal.zig");

const Array = metal.Array;
const Dtype = metal.Dtype;
const MapReduceOp = rowwise.MapReduceOp;

pub const SoftmaxOpts = struct {
    threadgroup: u32 = 256,
};

/// Create a softmax kernel for the last dimension of size `d`.
///
/// Uses the two-pass map-reduce pattern:
///   Pass 1: max reduction (numerical stability)
///   Pass 2: sum of exp(x - max)
///   Write:  exp(x - max) / sum
pub fn softmaxLastDim(
    allocator: Allocator,
    d: u32,
    compute_dtype: Dtype,
    opts: SoftmaxOpts,
) !MapReduceOp {
    return rowwise.mapReduce(
        allocator,
        "kk_softmax",
        d,
        opts.threadgroup,
        // Pass 1: max reduction
        "-INFINITY",
        "max(acc1, x)",
        "max(a, b)",
        // Pass 2: sum of exp(x - max)
        "0.0f",
        "acc2 + exp(x - s1)",
        "a + b",
        // Write: exp(x - max) / sum
        "exp(x - s1) / s2",
        compute_dtype,
        .{},
    );
}

// ---------------------------------------------------------------------------
// Tests (GPU — require Apple Silicon with Metal)
// ---------------------------------------------------------------------------

test "softmax — output sums to 1" {
    // Use page_allocator because kernelFromSpec caches globally (outlives test).
    const allocator = std.heap.page_allocator;
    const d: u32 = 4;
    const op = try softmaxLastDim(allocator, d, .float32, .{});

    // Single row: [1, 2, 3, 4]
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]i64{ 1, d };
    const x = Array.fromFloat32(&data, &shape);
    defer x.deinit();

    const result = try op.call(x, 1);
    defer result.deinit();

    result.eval();
    const out = result.dataFloat32();

    // Sum should be ~1.0
    var sum: f32 = 0.0;
    for (out) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // Largest input should get largest probability
    try std.testing.expect(out[3] > out[2]);
    try std.testing.expect(out[2] > out[1]);
    try std.testing.expect(out[1] > out[0]);

    // All probabilities should be positive
    for (out) |v| {
        try std.testing.expect(v > 0.0);
    }
}

test "softmax — multiple rows" {
    // Use page_allocator because kernelFromSpec caches globally (outlives test).
    const allocator = std.heap.page_allocator;
    const d: u32 = 3;
    const rows: u32 = 2;
    const op = try softmaxLastDim(allocator, d, .float32, .{});

    // Two rows: [1, 2, 3] and [10, 20, 30]
    const data = [_]f32{ 1.0, 2.0, 3.0, 10.0, 20.0, 30.0 };
    const shape = [_]i64{ rows, d };
    const x = Array.fromFloat32(&data, &shape);
    defer x.deinit();

    const result = try op.call(x, rows);
    defer result.deinit();

    result.eval();
    const out = result.dataFloat32();

    // Each row should sum to ~1.0
    var sum0: f32 = 0.0;
    for (0..d) |j| sum0 += out[j];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum0, 1e-5);

    var sum1: f32 = 0.0;
    for (d..2 * d) |j| sum1 += out[j];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum1, 1e-5);
}
