/// Convenience API for elementwise Metal kernels.
///
/// Reduces kernel creation from ~30 lines of boilerplate to a single
/// function call.  `unary()` and `binary()` generate MSL via codegen,
/// build a cached MetalKernel, and return a callable op struct.
const std = @import("std");
const Allocator = std.mem.Allocator;
const metal = @import("metal.zig");
const codegen = @import("codegen.zig");
const msl_mod = @import("msl.zig");

const Array = metal.Array;
const Dtype = metal.Dtype;
const MetalKernel = metal.MetalKernel;

// ---------------------------------------------------------------------------
// UnaryOp
// ---------------------------------------------------------------------------

pub const UnaryOp = struct {
    kernel: *MetalKernel,
    compute_dtype: Dtype,

    /// Launch the unary kernel on a single input array.
    pub fn call(self: *const UnaryOp, a: Array) !Array {
        const inputs = [_]Array{a};
        const tmpl_names = [_][]const u8{"T"};
        const tmpl_dtypes = [_]Dtype{self.compute_dtype};
        const outputs = try self.kernel.call(&inputs, .{
            .template_names = &tmpl_names,
            .template_dtypes = &tmpl_dtypes,
        });
        defer self.kernel.allocator.free(outputs);
        return outputs[0];
    }
};

pub const UnaryOpts = struct {
    header: []const u8 = msl_mod.DEFAULT_HEADER,
    inp: []const u8 = "inp",
    out: []const u8 = "out",
};

/// Create a cached unary elementwise kernel from a C expression.
///
/// The expression uses variable `x` (the input element).
/// Returns a `UnaryOp` backed by a kernel in the global cache.
pub fn unary(
    allocator: Allocator,
    expr: []const u8,
    name: []const u8,
    compute_dtype: Dtype,
    opts: UnaryOpts,
) !UnaryOp {
    const source = try codegen.elementwiseUnarySource(allocator, expr, opts.inp, opts.out);
    defer allocator.free(source);

    const input_names = [_][]const u8{opts.inp};
    const output_names = [_][]const u8{opts.out};

    const kernel = try metal.kernelFromSpec(allocator, .{
        .name = name,
        .input_names = &input_names,
        .output_names = &output_names,
        .source = source,
        .header = opts.header,
    }, true);

    return .{
        .kernel = kernel,
        .compute_dtype = compute_dtype,
    };
}

// ---------------------------------------------------------------------------
// BinaryOp
// ---------------------------------------------------------------------------

pub const BinaryOp = struct {
    kernel: *MetalKernel,
    compute_dtype: Dtype,

    /// Launch the binary kernel on two input arrays.
    pub fn call(self: *const BinaryOp, a: Array, b: Array) !Array {
        const inputs = [_]Array{ a, b };
        const tmpl_names = [_][]const u8{"T"};
        const tmpl_dtypes = [_]Dtype{self.compute_dtype};
        const outputs = try self.kernel.call(&inputs, .{
            .template_names = &tmpl_names,
            .template_dtypes = &tmpl_dtypes,
        });
        defer self.kernel.allocator.free(outputs);
        return outputs[0];
    }
};

pub const BinaryOpts = struct {
    header: []const u8 = msl_mod.DEFAULT_HEADER,
    lhs: []const u8 = "lhs",
    rhs: []const u8 = "rhs",
    out: []const u8 = "out",
};

/// Create a cached binary elementwise kernel from a C expression.
///
/// The expression uses variables `a` (left), `b` (right), and `x` (alias for `a`).
/// Returns a `BinaryOp` backed by a kernel in the global cache.
pub fn binary(
    allocator: Allocator,
    expr: []const u8,
    name: []const u8,
    compute_dtype: Dtype,
    opts: BinaryOpts,
) !BinaryOp {
    const source = try codegen.elementwiseBinarySource(
        allocator,
        expr,
        opts.lhs,
        opts.rhs,
        opts.out,
    );
    defer allocator.free(source);

    const input_names = [_][]const u8{ opts.lhs, opts.rhs };
    const output_names = [_][]const u8{opts.out};

    const kernel = try metal.kernelFromSpec(allocator, .{
        .name = name,
        .input_names = &input_names,
        .output_names = &output_names,
        .source = source,
        .header = opts.header,
    }, true);

    return .{
        .kernel = kernel,
        .compute_dtype = compute_dtype,
    };
}

// ---------------------------------------------------------------------------
// Tests (GPU — require Apple Silicon with Metal)
// ---------------------------------------------------------------------------

test "unary op — square" {
    // Use page_allocator because kernelFromSpec caches globally (outlives test).
    const allocator = std.heap.page_allocator;
    const op = try unary(allocator, "x * x", "test_square", .float32, .{});

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]i64{4};
    const a = Array.fromFloat32(&data, &shape);
    defer a.deinit();

    const result = try op.call(a);
    defer result.deinit();

    result.eval();
    const out = result.dataFloat32();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), out[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), out[3], 1e-5);
}

test "binary op — add" {
    const allocator = std.heap.page_allocator;
    const op = try binary(allocator, "a + b", "test_add", .float32, .{});

    const data_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const data_b = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    const shape = [_]i64{4};
    const a = Array.fromFloat32(&data_a, &shape);
    defer a.deinit();
    const b = Array.fromFloat32(&data_b, &shape);
    defer b.deinit();

    const result = try op.call(a, b);
    defer result.deinit();

    result.eval();
    const out = result.dataFloat32();
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), out[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 44.0), out[3], 1e-5);
}
