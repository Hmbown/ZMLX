const std = @import("std");

pub fn main() !void {
    const out = std.io.getStdOut().writer();
    try out.print("zmlx demo scaffold (wire MLX-C next)\n", .{});
}
