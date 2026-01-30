/// Kernel metadata export — serializes MetalKernelSpec to JSON.
///
/// Enables interop between the Zig and Python frontends: a kernel spec
/// defined in Zig can be exported to JSON and loaded by Python's
/// `metal.kernel()` factory.
const std = @import("std");
const Allocator = std.mem.Allocator;
const metal = @import("metal.zig");

const MetalKernelSpec = metal.MetalKernelSpec;

/// Serialize a MetalKernelSpec to a JSON string.
///
/// The returned string is heap-allocated; the caller must free it.
pub fn specToJson(allocator: Allocator, spec: MetalKernelSpec) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    const w = buf.writer(allocator);

    try w.writeAll("{\n");

    // name
    try w.writeAll("  \"name\": ");
    try writeJsonString(w, spec.name);
    try w.writeAll(",\n");

    // input_names
    try w.writeAll("  \"input_names\": ");
    try writeJsonStringArray(w, spec.input_names);
    try w.writeAll(",\n");

    // output_names
    try w.writeAll("  \"output_names\": ");
    try writeJsonStringArray(w, spec.output_names);
    try w.writeAll(",\n");

    // source
    try w.writeAll("  \"source\": ");
    try writeJsonString(w, spec.source);
    try w.writeAll(",\n");

    // header
    try w.writeAll("  \"header\": ");
    try writeJsonString(w, spec.header);
    try w.writeAll(",\n");

    // ensure_row_contiguous
    try std.fmt.format(w, "  \"ensure_row_contiguous\": {s},\n", .{
        if (spec.ensure_row_contiguous) "true" else "false",
    });

    // atomic_outputs
    try std.fmt.format(w, "  \"atomic_outputs\": {s}\n", .{
        if (spec.atomic_outputs) "true" else "false",
    });

    try w.writeAll("}");

    return buf.toOwnedSlice(allocator);
}

/// Write a JSON-escaped string (with surrounding quotes).
fn writeJsonString(w: anytype, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |ch| {
        switch (ch) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => {
                if (ch < 0x20) {
                    try std.fmt.format(w, "\\u{x:0>4}", .{ch});
                } else {
                    try w.writeByte(ch);
                }
            },
        }
    }
    try w.writeByte('"');
}

/// Write a JSON array of strings.
fn writeJsonStringArray(w: anytype, items: []const []const u8) !void {
    try w.writeByte('[');
    for (items, 0..) |item, i| {
        if (i > 0) try w.writeAll(", ");
        try writeJsonString(w, item);
    }
    try w.writeByte(']');
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "specToJson — basic spec" {
    const allocator = std.testing.allocator;
    const input_names = [_][]const u8{"inp"};
    const output_names = [_][]const u8{"out"};
    const json = try specToJson(allocator, .{
        .name = "test_kernel",
        .input_names = &input_names,
        .output_names = &output_names,
        .source = "x * x",
        .header = "",
    });
    defer allocator.free(json);

    // Verify key fields are present
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\": \"test_kernel\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"input_names\": [\"inp\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"output_names\": [\"out\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"source\": \"x * x\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"ensure_row_contiguous\": true") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"atomic_outputs\": false") != null);
}

test "specToJson — escapes newlines" {
    const allocator = std.testing.allocator;
    const input_names = [_][]const u8{"inp"};
    const output_names = [_][]const u8{"out"};
    const json = try specToJson(allocator, .{
        .name = "test",
        .input_names = &input_names,
        .output_names = &output_names,
        .source = "line1\nline2",
        .header = "h1\th2",
    });
    defer allocator.free(json);

    // Newline should be escaped
    try std.testing.expect(std.mem.indexOf(u8, json, "line1\\nline2") != null);
    // Tab should be escaped
    try std.testing.expect(std.mem.indexOf(u8, json, "h1\\th2") != null);
}

test "specToJson — multiple inputs/outputs" {
    const allocator = std.testing.allocator;
    const input_names = [_][]const u8{ "lhs", "rhs" };
    const output_names = [_][]const u8{ "out1", "out2" };
    const json = try specToJson(allocator, .{
        .name = "multi",
        .input_names = &input_names,
        .output_names = &output_names,
        .source = "a + b",
        .header = "",
    });
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"input_names\": [\"lhs\", \"rhs\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"output_names\": [\"out1\", \"out2\"]") != null);
}
