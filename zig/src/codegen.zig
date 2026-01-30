/// Tiny IR / codegen helpers — Zig port of `src/zmlx/codegen.py`.
///
/// Every function generates a Metal Shading Language (MSL) source snippet
/// that is **byte-identical** to the Python version for the same parameters.
/// This is critical because MLX caches compiled Metal by source string.
const std = @import("std");
const Allocator = std.mem.Allocator;

// ---------------------------------------------------------------------------
// Elementwise patterns
// ---------------------------------------------------------------------------

/// Generate an elementwise unary kernel body.
///
/// Matches Python: `codegen.elementwise_unary_source(expr=, inp=, out=)`
pub fn elementwiseUnarySource(
    allocator: Allocator,
    expr: []const u8,
    inp: []const u8,
    out: []const u8,
) Allocator.Error![]u8 {
    return std.fmt.allocPrint(allocator,
        "\n" ++
        "        uint elem = thread_position_in_grid.x;\n" ++
        "        T x = {s}[elem];\n" ++
        "        {s}[elem] = {s};\n" ++
        "    ", .{ inp, out, expr });
}

/// Generate an elementwise binary kernel body.
///
/// Matches Python: `codegen.elementwise_binary_source(expr=, lhs=, rhs=, out=)`
pub fn elementwiseBinarySource(
    allocator: Allocator,
    expr: []const u8,
    lhs: []const u8,
    rhs: []const u8,
    out: []const u8,
) Allocator.Error![]u8 {
    return std.fmt.allocPrint(allocator,
        "\n" ++
        "        uint elem = thread_position_in_grid.x;\n" ++
        "        T a = {s}[elem];\n" ++
        "        T b = {s}[elem];\n" ++
        "        T x = a;\n" ++
        "        {s}[elem] = {s};\n" ++
        "    ", .{ lhs, rhs, out, expr });
}

// ---------------------------------------------------------------------------
// Rowwise reduction patterns
// ---------------------------------------------------------------------------

/// Generate a simple rowwise reduction over the last dimension.
///
/// Launch convention:
///   - grid.x == rows (one thread per row)
///   - threadgroup.x can be 1 (no parallelism)
///
/// Matches Python: `codegen.rowwise_reduction_source(...)`
pub fn rowwiseReductionSource(
    allocator: Allocator,
    reduce_expr: []const u8,
    init_expr: []const u8,
    finalize_expr: []const u8,
    d: u32,
    inp: []const u8,
    out: []const u8,
) Allocator.Error![]u8 {
    return std.fmt.allocPrint(allocator,
        "\n" ++
        "        uint row = thread_position_in_grid.x;\n" ++
        "        uint base = row * {d};\n" ++
        "        T acc = {s};\n" ++
        "        for (uint j = 0; j < {d}; ++j) {{\n" ++
        "            T v = {s}[base + j];\n" ++
        "            acc = {s};\n" ++
        "        }}\n" ++
        "        {s}[row] = {s};\n" ++
        "    ", .{ d, init_expr, d, inp, reduce_expr, out, finalize_expr });
}

/// Generate a rowwise parallel reduction kernel (1 scalar pass + 1 output write).
///
/// Launch convention:
///   - threadgroup.x == TG
///   - grid.x == rows * TG
///
/// Matches Python: `codegen.rowwise_parallel_reduction_source(...)`
pub fn rowwiseParallelReductionSource(
    allocator: Allocator,
    d: u32,
    tg: u32,
    init_expr: []const u8,
    update_expr: []const u8,
    reduce_op: []const u8,
    finalize_expr: []const u8,
    inp: []const u8,
    out: []const u8,
    scratch: []const u8,
) Allocator.Error![]u8 {
    // The format string reproduces the Python f-string byte-for-byte,
    // including the trailing-whitespace line between `}}` and `if (tid == 0)`.
    return std.fmt.allocPrint(allocator,
        "\n" ++
        "        constexpr uint D = {d};\n" ++
        "        constexpr uint TG = {d};\n" ++
        "\n" ++
        "        uint gid = thread_position_in_grid.x;\n" ++
        "        uint tid = thread_position_in_threadgroup.x;\n" ++
        "        uint row = gid / TG;\n" ++
        "        uint base = row * D;\n" ++
        "\n" ++
        "        threadgroup float {s}[TG];\n" ++
        "\n" ++
        "        float acc = {s};\n" ++
        "        for (uint j = tid; j < D; j += TG) {{\n" ++
        "            float x = (float){s}[base + j];\n" ++
        "            acc = {s};\n" ++
        "        }}\n" ++
        "        {s}[tid] = acc;\n" ++
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n" ++
        "\n" ++
        "        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{\n" ++
        "            if (tid < stride) {{\n" ++
        "                float a = {s}[tid];\n" ++
        "                float b = {s}[tid + stride];\n" ++
        "                {s}[tid] = {s};\n" ++
        "            }}\n" ++
        "            threadgroup_barrier(mem_flags::mem_threadgroup);\n" ++
        "        }}\n" ++
        "        \n" ++ // trailing-whitespace line (matches Python source)
        "        if (tid == 0) {{\n" ++
        "            float s = {s}[0];\n" ++
        "            {s}[row] = (T)({s});\n" ++
        "        }}\n" ++
        "    ",
        .{
            d,     tg,          // constexpr D, TG
            scratch,             // threadgroup float {scratch}[TG]
            init_expr,           // float acc = {init_expr}
            inp,                 // (float){inp}[base + j]
            update_expr,         // acc = {update_expr}
            scratch,             // {scratch}[tid] = acc
            scratch,             // float a = {scratch}[tid]
            scratch,             // float b = {scratch}[tid + stride]
            scratch, reduce_op,  // {scratch}[tid] = {reduce_op}
            scratch,             // float s = {scratch}[0]
            out,     finalize_expr, // {out}[row] = (T)({finalize_expr})
        },
    );
}

/// Generate a rowwise map-reduce kernel (2 scalar passes + elementwise write).
///
/// This is the workhorse behind softmax, layer-norm, etc.
///
/// Launch convention:
///   - threadgroup.x == TG
///   - grid.x == rows * TG
///
/// Matches Python: `codegen.rowwise_mapreduce_source(...)`
pub fn rowwiseMapReduceSource(
    allocator: Allocator,
    d: u32,
    tg: u32,
    pass1_init: []const u8,
    pass1_update: []const u8,
    pass1_reduce_op: []const u8,
    pass2_init: []const u8,
    pass2_update: []const u8,
    pass2_reduce_op: []const u8,
    write_expr: []const u8,
    inp: []const u8,
    out: []const u8,
    scratch1: []const u8,
    scratch2: []const u8,
) Allocator.Error![]u8 {
    return std.fmt.allocPrint(allocator,
        "\n" ++
        "        constexpr uint D = {d};\n" ++
        "        constexpr uint TG = {d};\n" ++
        "\n" ++
        "        uint gid = thread_position_in_grid.x;\n" ++
        "        uint tid = thread_position_in_threadgroup.x;\n" ++
        "        uint row = gid / TG;\n" ++
        "        uint base = row * D;\n" ++
        "\n" ++
        "        threadgroup float {s}[TG];\n" ++
        "        threadgroup float {s}[TG];\n" ++
        "\n" ++
        "        // pass 1\n" ++
        "        float acc1 = {s};\n" ++
        "        for (uint j = tid; j < D; j += TG) {{\n" ++
        "            float x = (float){s}[base + j];\n" ++
        "            acc1 = {s};\n" ++
        "        }}\n" ++
        "        {s}[tid] = acc1;\n" ++
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n" ++
        "\n" ++
        "        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{\n" ++
        "            if (tid < stride) {{\n" ++
        "                float a = {s}[tid];\n" ++
        "                float b = {s}[tid + stride];\n" ++
        "                {s}[tid] = {s};\n" ++
        "            }}\n" ++
        "            threadgroup_barrier(mem_flags::mem_threadgroup);\n" ++
        "        }}\n" ++
        "        float s1 = {s}[0];\n" ++
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n" ++
        "\n" ++
        "        // pass 2\n" ++
        "        float acc2 = {s};\n" ++
        "        for (uint j = tid; j < D; j += TG) {{\n" ++
        "            float x = (float){s}[base + j];\n" ++
        "            acc2 = {s};\n" ++
        "        }}\n" ++
        "        {s}[tid] = acc2;\n" ++
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n" ++
        "\n" ++
        "        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{\n" ++
        "            if (tid < stride) {{\n" ++
        "                float a = {s}[tid];\n" ++
        "                float b = {s}[tid + stride];\n" ++
        "                {s}[tid] = {s};\n" ++
        "            }}\n" ++
        "            threadgroup_barrier(mem_flags::mem_threadgroup);\n" ++
        "        }}\n" ++
        "        float s2 = {s}[0];\n" ++
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n" ++
        "\n" ++
        "        // write outputs\n" ++
        "        for (uint j = tid; j < D; j += TG) {{\n" ++
        "            float x = (float){s}[base + j];\n" ++
        "            {s}[base + j] = (T)({s});\n" ++
        "        }}\n" ++
        "    ",
        .{
            d,          tg,              // constexpr D, TG
            scratch1,   scratch2,        // threadgroup float buf1/buf2[TG]
            // --- pass 1 ---
            pass1_init,                  // float acc1 = ...
            inp,                         // (float)inp[base+j]
            pass1_update,                // acc1 = ...
            scratch1,                    // buf1[tid] = acc1
            scratch1,                    // float a = buf1[tid]
            scratch1,                    // float b = buf1[tid+stride]
            scratch1,   pass1_reduce_op, // buf1[tid] = ...
            scratch1,                    // float s1 = buf1[0]
            // --- pass 2 ---
            pass2_init,                  // float acc2 = ...
            inp,                         // (float)inp[base+j]
            pass2_update,                // acc2 = ...
            scratch2,                    // buf2[tid] = acc2
            scratch2,                    // float a = buf2[tid]
            scratch2,                    // float b = buf2[tid+stride]
            scratch2,   pass2_reduce_op, // buf2[tid] = ...
            scratch2,                    // float s2 = buf2[0]
            // --- write ---
            inp,                         // (float)inp[base+j]
            out,        write_expr,      // out[base+j] = (T)(write_expr)
        },
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "elementwise unary source" {
    const allocator = std.testing.allocator;
    const result = try elementwiseUnarySource(allocator, "x*x", "inp", "out");
    defer allocator.free(result);

    const expected =
        "\n" ++
        "        uint elem = thread_position_in_grid.x;\n" ++
        "        T x = inp[elem];\n" ++
        "        out[elem] = x*x;\n" ++
        "    ";
    try std.testing.expectEqualStrings(expected, result);
}

test "elementwise unary source — custom names" {
    const allocator = std.testing.allocator;
    const result = try elementwiseUnarySource(allocator, "kk_sigmoid(x)", "input", "output");
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "T x = input[elem];") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "output[elem] = kk_sigmoid(x);") != null);
}

test "elementwise binary source" {
    const allocator = std.testing.allocator;
    const result = try elementwiseBinarySource(allocator, "a + b", "lhs", "rhs", "out");
    defer allocator.free(result);

    const expected =
        "\n" ++
        "        uint elem = thread_position_in_grid.x;\n" ++
        "        T a = lhs[elem];\n" ++
        "        T b = rhs[elem];\n" ++
        "        T x = a;\n" ++
        "        out[elem] = a + b;\n" ++
        "    ";
    try std.testing.expectEqualStrings(expected, result);
}

test "rowwise reduction source" {
    const allocator = std.testing.allocator;
    const result = try rowwiseReductionSource(
        allocator,
        "max(acc, v)",
        "-INFINITY",
        "acc",
        128,
        "inp",
        "out",
    );
    defer allocator.free(result);

    // Verify key fragments
    try std.testing.expect(std.mem.indexOf(u8, result, "uint base = row * 128;") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "T acc = -INFINITY;") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "for (uint j = 0; j < 128; ++j) {") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "acc = max(acc, v);") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "out[row] = acc;") != null);
}

test "rowwise parallel reduction source" {
    const allocator = std.testing.allocator;
    const result = try rowwiseParallelReductionSource(
        allocator,
        128,
        256,
        "-INFINITY",
        "max(acc, x)",
        "max(a, b)",
        "s",
        "inp",
        "out",
        "buf",
    );
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "constexpr uint D = 128;") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "constexpr uint TG = 256;") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "threadgroup float buf[TG];") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "float acc = -INFINITY;") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "acc = max(acc, x);") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "buf[tid] = max(a, b);") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "out[row] = (T)(s);") != null);
}

test "rowwise mapreduce source — softmax pattern" {
    const allocator = std.testing.allocator;
    const result = try rowwiseMapReduceSource(
        allocator,
        128,
        256,
        "-INFINITY",
        "max(acc1, x)",
        "max(a, b)",
        "0.0f",
        "acc2 + exp(x - s1)",
        "a + b",
        "exp(x - s1) / s2",
        "inp",
        "out",
        "buf1",
        "buf2",
    );
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "constexpr uint D = 128;") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "constexpr uint TG = 256;") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "// pass 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "float acc1 = -INFINITY;") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "// pass 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "float acc2 = 0.0f;") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "// write outputs") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "out[base + j] = (T)(exp(x - s1) / s2);") != null);
}
