/// ZMLX Zig frontend demo.
///
/// Creates two float32 arrays, builds an elementwise add kernel via codegen,
/// launches it through the MetalKernel wrapper, and prints the result.
const std = @import("std");
const codegen = @import("codegen.zig");
const msl = @import("msl.zig");
const metal = @import("metal.zig");

const Array = metal.Array;
const MetalKernel = metal.MetalKernel;
const Dtype = metal.Dtype;

pub fn main() !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    const allocator = std.heap.page_allocator;

    try stdout.print("=== ZMLX Zig frontend demo ===\n\n", .{});

    // 1. Create two float32 arrays.
    const data_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const data_b = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    const shape = [_]i64{4};

    const a = Array.fromFloat32(&data_a, &shape);
    defer a.deinit();
    const b = Array.fromFloat32(&data_b, &shape);
    defer b.deinit();

    try stdout.print("a = [1, 2, 3, 4]\n", .{});
    try stdout.print("b = [10, 20, 30, 40]\n\n", .{});

    // 2. Generate an elementwise add kernel using codegen.
    const source = try codegen.elementwiseBinarySource(
        allocator,
        "a + b",
        "lhs",
        "rhs",
        "out",
    );
    defer allocator.free(source);

    try stdout.print("Generated MSL source ({d} bytes):\n{s}\n\n", .{ source.len, source });

    // 3. Build and launch via MetalKernel wrapper.
    //    MetalKernel.init creates null-terminated copies of all strings
    //    for safe C interop, so plain Zig slices work here.
    const input_names = [_][]const u8{ "lhs", "rhs" };
    const output_names = [_][]const u8{"out"};

    var kern = try MetalKernel.init(allocator, .{
        .name = "vector_add",
        .input_names = &input_names,
        .output_names = &output_names,
        .source = source,
        .header = msl.DEFAULT_HEADER,
    });
    defer kern.deinit();

    const inputs = [_]Array{ a, b };
    const template_names = [_][]const u8{"T"};
    const template_dtypes = [_]Dtype{.float32};
    const outputs = try kern.call(&inputs, .{
        .template_names = &template_names,
        .template_dtypes = &template_dtypes,
    });
    defer {
        for (outputs) |o| o.deinit();
        allocator.free(outputs);
    }

    // 4. Print the result.
    outputs[0].eval();
    const result = outputs[0].dataFloat32();

    try stdout.print("result = [", .{});
    for (result, 0..) |v, i| {
        if (i > 0) try stdout.print(", ", .{});
        try stdout.print("{d:.1}", .{v});
    }
    try stdout.print("]\n", .{});
    try stdout.print("\nExpected: [11.0, 22.0, 33.0, 44.0]\n", .{});
}
