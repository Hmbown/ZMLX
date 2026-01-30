/// Convenience API for rowwise Metal kernels.
///
/// `mapReduce()` creates a two-pass map-reduce kernel (e.g. softmax,
/// layer-norm) and `parallelReduce()` creates a single-pass parallel
/// reduction kernel (e.g. row-sum, row-max).
const std = @import("std");
const Allocator = std.mem.Allocator;
const metal = @import("metal.zig");
const codegen = @import("codegen.zig");
const msl_mod = @import("msl.zig");

const Array = metal.Array;
const Dtype = metal.Dtype;
const MetalKernel = metal.MetalKernel;

// ---------------------------------------------------------------------------
// MapReduceOp  (two-pass: e.g. softmax, layer-norm)
// ---------------------------------------------------------------------------

pub const MapReduceOp = struct {
    kernel: *MetalKernel,
    d: u32,
    tg: u32,
    compute_dtype: Dtype,

    /// Launch the map-reduce kernel.
    ///
    /// `x` must have shape (rows, d) laid out contiguously.
    /// Returns an array of the same shape.
    pub fn call(self: *const MapReduceOp, x: Array, rows: u32) !Array {
        const inputs = [_]Array{x};
        const tmpl_names = [_][]const u8{"T"};
        const tmpl_dtypes = [_]Dtype{self.compute_dtype};
        const out_shape = [_]i64{ @intCast(rows), @intCast(self.d) };
        const out_shapes = [_][]const i64{&out_shape};

        const outputs = try self.kernel.call(&inputs, .{
            .grid = .{ rows * self.tg, 1, 1 },
            .threadgroup = .{ self.tg, 1, 1 },
            .template_names = &tmpl_names,
            .template_dtypes = &tmpl_dtypes,
            .output_shapes = &out_shapes,
        });
        defer self.kernel.allocator.free(outputs);
        return outputs[0];
    }
};

pub const MapReduceOpts = struct {
    header: []const u8 = msl_mod.DEFAULT_HEADER,
    inp: []const u8 = "inp",
    out: []const u8 = "out",
    scratch1: []const u8 = "buf1",
    scratch2: []const u8 = "buf2",
};

/// Create a cached two-pass map-reduce rowwise kernel.
///
/// Launch convention: grid.x = rows * tg, threadgroup.x = tg.
/// Output shape = input shape = (rows, d).
pub fn mapReduce(
    allocator: Allocator,
    name: []const u8,
    d: u32,
    tg: u32,
    pass1_init: []const u8,
    pass1_update: []const u8,
    pass1_reduce_op: []const u8,
    pass2_init: []const u8,
    pass2_update: []const u8,
    pass2_reduce_op: []const u8,
    write_expr: []const u8,
    compute_dtype: Dtype,
    opts: MapReduceOpts,
) !MapReduceOp {
    try validateThreadgroup(tg);

    const source = try codegen.rowwiseMapReduceSource(
        allocator,
        d,
        tg,
        pass1_init,
        pass1_update,
        pass1_reduce_op,
        pass2_init,
        pass2_update,
        pass2_reduce_op,
        write_expr,
        opts.inp,
        opts.out,
        opts.scratch1,
        opts.scratch2,
    );
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
        .d = d,
        .tg = tg,
        .compute_dtype = compute_dtype,
    };
}

// ---------------------------------------------------------------------------
// ParallelReductionOp  (single-pass: row-sum, row-max, etc.)
// ---------------------------------------------------------------------------

pub const ParallelReductionOp = struct {
    kernel: *MetalKernel,
    d: u32,
    tg: u32,
    compute_dtype: Dtype,

    /// Launch the parallel reduction kernel.
    ///
    /// `x` must have shape (rows, d) laid out contiguously.
    /// Returns an array of shape (rows,) — one scalar per row.
    pub fn call(self: *const ParallelReductionOp, x: Array, rows: u32) !Array {
        const inputs = [_]Array{x};
        const tmpl_names = [_][]const u8{"T"};
        const tmpl_dtypes = [_]Dtype{self.compute_dtype};
        const out_shape = [_]i64{@intCast(rows)};
        const out_shapes = [_][]const i64{&out_shape};

        const outputs = try self.kernel.call(&inputs, .{
            .grid = .{ rows * self.tg, 1, 1 },
            .threadgroup = .{ self.tg, 1, 1 },
            .template_names = &tmpl_names,
            .template_dtypes = &tmpl_dtypes,
            .output_shapes = &out_shapes,
        });
        defer self.kernel.allocator.free(outputs);
        return outputs[0];
    }
};

pub const ParallelReductionOpts = struct {
    header: []const u8 = msl_mod.DEFAULT_HEADER,
    inp: []const u8 = "inp",
    out: []const u8 = "out",
    scratch: []const u8 = "buf",
};

/// Create a cached single-pass parallel reduction kernel.
///
/// Launch convention: grid.x = rows * tg, threadgroup.x = tg.
/// Output shape = (rows,) — one scalar per row.
pub fn parallelReduce(
    allocator: Allocator,
    name: []const u8,
    d: u32,
    tg: u32,
    init_expr: []const u8,
    update_expr: []const u8,
    reduce_op: []const u8,
    finalize_expr: []const u8,
    compute_dtype: Dtype,
    opts: ParallelReductionOpts,
) !ParallelReductionOp {
    try validateThreadgroup(tg);

    const source = try codegen.rowwiseParallelReductionSource(
        allocator,
        d,
        tg,
        init_expr,
        update_expr,
        reduce_op,
        finalize_expr,
        opts.inp,
        opts.out,
        opts.scratch,
    );
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
        .d = d,
        .tg = tg,
        .compute_dtype = compute_dtype,
    };
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate that tg is a power of 2 in (0, 1024].
pub fn validateThreadgroup(tg: u32) !void {
    if (tg == 0 or tg > 1024 or (tg & (tg - 1)) != 0) {
        return error.InvalidThreadgroup;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "validateThreadgroup — valid values" {
    try validateThreadgroup(1);
    try validateThreadgroup(2);
    try validateThreadgroup(4);
    try validateThreadgroup(8);
    try validateThreadgroup(16);
    try validateThreadgroup(32);
    try validateThreadgroup(64);
    try validateThreadgroup(128);
    try validateThreadgroup(256);
    try validateThreadgroup(512);
    try validateThreadgroup(1024);
}

test "validateThreadgroup — invalid values" {
    try std.testing.expectError(error.InvalidThreadgroup, validateThreadgroup(0));
    try std.testing.expectError(error.InvalidThreadgroup, validateThreadgroup(3));
    try std.testing.expectError(error.InvalidThreadgroup, validateThreadgroup(5));
    try std.testing.expectError(error.InvalidThreadgroup, validateThreadgroup(6));
    try std.testing.expectError(error.InvalidThreadgroup, validateThreadgroup(7));
    try std.testing.expectError(error.InvalidThreadgroup, validateThreadgroup(9));
    try std.testing.expectError(error.InvalidThreadgroup, validateThreadgroup(100));
    try std.testing.expectError(error.InvalidThreadgroup, validateThreadgroup(2048));
}
