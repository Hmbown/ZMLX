const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zmlx_demo",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // TODO: link MLX-C here (and any shim library you add).
    // exe.addIncludePath(.{ .path = "path/to/mlx-c/include" });
    // exe.addLibraryPath(.{ .path = "path/to/mlx-c/lib" });
    // exe.linkSystemLibrary("mlx"); // placeholder

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the demo");
    run_step.dependOn(&run_cmd.step);
}
