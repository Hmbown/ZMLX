/// Normalization kernels — Zig port of `src/zmlx/kernels/norms.py`.
///
/// RMSNorm (no learned weight) using the two-pass map-reduce pattern.
const std = @import("std");
const Allocator = std.mem.Allocator;
const rowwise = @import("../rowwise.zig");
const metal = @import("../metal.zig");

const Array = metal.Array;
const Dtype = metal.Dtype;
const MapReduceOp = rowwise.MapReduceOp;

pub const RmsNormOpts = struct {
    threadgroup: u32 = 256,
};

/// Create an RMSNorm kernel (no learned weight) for the last dimension of size `d`.
///
/// Uses the map-reduce pattern:
///   Pass 1: sum of x*x
///   Pass 2: (minimal, unused — map-reduce requires two passes)
///   Write:  x * rsqrt(sum_sq / d + eps)
pub fn rmsNormNoWeight(
    allocator: Allocator,
    d: u32,
    eps: f32,
    compute_dtype: Dtype,
    opts: RmsNormOpts,
) !MapReduceOp {
    // Build write expression with baked-in constants.
    // Result: "x * rsqrt(s1 / <d>.0f + <eps>f)"
    const write_expr = try std.fmt.allocPrint(
        allocator,
        "x * rsqrt(s1 / {d}.0f + {e}f)",
        .{ d, eps },
    );
    defer allocator.free(write_expr);

    return rowwise.mapReduce(
        allocator,
        "kk_rms_norm",
        d,
        opts.threadgroup,
        // Pass 1: sum of squares
        "0.0f",
        "acc1 + x * x",
        "a + b",
        // Pass 2: unused (minimal cost — map-reduce requires two passes)
        "0.0f",
        "0.0f",
        "a + b",
        // Write: x * rsqrt(mean_sq + eps)
        write_expr,
        compute_dtype,
        .{},
    );
}

// ---------------------------------------------------------------------------
// Tests (GPU — require Apple Silicon with Metal)
// ---------------------------------------------------------------------------

test "rmsNorm — unit RMS output" {
    // Use page_allocator because kernelFromSpec caches globally (outlives test).
    const allocator = std.heap.page_allocator;
    const d: u32 = 4;
    const eps: f32 = 1e-5;
    const op = try rmsNormNoWeight(allocator, d, eps, .float32, .{});

    // Single row: [1, 2, 3, 4]
    // RMS = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) ≈ 2.7386
    // Output = x / RMS
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]i64{ 1, d };
    const x = Array.fromFloat32(&data, &shape);
    defer x.deinit();

    const result = try op.call(x, 1);
    defer result.deinit();

    result.eval();
    const out = result.dataFloat32();

    // Compute expected RMS of output: should be ~1.0
    var sum_sq: f32 = 0.0;
    for (out) |v| sum_sq += v * v;
    const rms_out = @sqrt(sum_sq / @as(f32, @floatFromInt(d)));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rms_out, 1e-3);

    // Output should preserve relative ordering
    try std.testing.expect(out[3] > out[2]);
    try std.testing.expect(out[2] > out[1]);
    try std.testing.expect(out[1] > out[0]);
}
