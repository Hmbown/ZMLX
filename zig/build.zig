const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ------------------------------------------------------------------
    // MLX paths — adjust these to match your installation.
    // Homebrew default:   /opt/homebrew/include, /opt/homebrew/lib
    // CMake build tree:   path/to/mlx/build/include, path/to/mlx/build/lib
    // ------------------------------------------------------------------
    const mlx_include = b.option(
        []const u8,
        "mlx-include",
        "Path to MLX C++ headers (default: /opt/homebrew/include)",
    ) orelse "/opt/homebrew/include";
    const mlx_lib = b.option(
        []const u8,
        "mlx-lib",
        "Path to MLX library directory (default: /opt/homebrew/lib)",
    ) orelse "/opt/homebrew/lib";

    // ------------------------------------------------------------------
    // C++ shim static library
    // ------------------------------------------------------------------
    const shim = b.addLibrary(.{
        .name = "zmlx_shim",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libcpp = true,
        }),
    });
    shim.addCSourceFile(.{
        .file = b.path("shim/shim.cc"),
        .flags = &.{"-std=c++17"},
    });
    shim.addIncludePath(b.path("shim"));
    shim.addSystemIncludePath(.{ .cwd_relative = mlx_include });

    // ------------------------------------------------------------------
    // Helper: configure a compile step to link the shim + MLX + frameworks
    // ------------------------------------------------------------------
    const configureLink = struct {
        fn apply(
            step: *std.Build.Step.Compile,
            b_: *std.Build,
            shim_: *std.Build.Step.Compile,
            mlx_include_: []const u8,
            mlx_lib_: []const u8,
        ) void {
            step.addIncludePath(b_.path("shim"));
            step.addSystemIncludePath(.{ .cwd_relative = mlx_include_ });
            step.linkLibrary(shim_);
            step.addLibraryPath(.{ .cwd_relative = mlx_lib_ });
            step.linkSystemLibrary("mlx");
            step.root_module.link_libcpp = true;
            step.linkFramework("Metal");
            step.linkFramework("Foundation");
            step.linkFramework("Accelerate");
        }
    }.apply;

    // ------------------------------------------------------------------
    // Demo executable
    // ------------------------------------------------------------------
    const exe = b.addExecutable(.{
        .name = "zmlx_demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureLink(exe, b, shim, mlx_include, mlx_lib);
    b.installArtifact(exe);

    // zig build run
    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the demo");
    run_step.dependOn(&run_cmd.step);

    // ------------------------------------------------------------------
    // Unit tests — pure Zig (no shim required)
    // ------------------------------------------------------------------
    const codegen_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/codegen.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // ------------------------------------------------------------------
    // Tests requiring shim + MLX linkage
    // ------------------------------------------------------------------
    const metal_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/metal.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureLink(metal_tests, b, shim, mlx_include, mlx_lib);

    const elementwise_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/elementwise.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureLink(elementwise_tests, b, shim, mlx_include, mlx_lib);

    const rowwise_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/rowwise.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureLink(rowwise_tests, b, shim, mlx_include, mlx_lib);

    const metadata_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/metadata.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureLink(metadata_tests, b, shim, mlx_include, mlx_lib);

    // ------------------------------------------------------------------
    // Catalog kernel tests (GPU — require Apple Silicon with Metal)
    // Root at kernels.zig so sub-modules can @import("../elementwise.zig").
    // ------------------------------------------------------------------
    const catalog_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/kernels.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureLink(catalog_tests, b, shim, mlx_include, mlx_lib);

    // ------------------------------------------------------------------
    // Test steps
    // ------------------------------------------------------------------

    // zig build test  — all tests
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&b.addRunArtifact(codegen_tests).step);
    test_step.dependOn(&b.addRunArtifact(metal_tests).step);
    test_step.dependOn(&b.addRunArtifact(elementwise_tests).step);
    test_step.dependOn(&b.addRunArtifact(rowwise_tests).step);
    test_step.dependOn(&b.addRunArtifact(metadata_tests).step);
    test_step.dependOn(&b.addRunArtifact(catalog_tests).step);

    // zig build test-codegen  (fast, no MLX needed)
    const test_codegen_step = b.step("test-codegen", "Run codegen-only tests");
    test_codegen_step.dependOn(&b.addRunArtifact(codegen_tests).step);

    // zig build test-catalog  (kernel catalog GPU tests)
    const test_catalog_step = b.step("test-catalog", "Run kernel catalog tests (GPU)");
    test_catalog_step.dependOn(&b.addRunArtifact(catalog_tests).step);
}
