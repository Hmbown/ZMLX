/// Kernel catalog — re-exports all kernel modules.
///
/// Organized by domain, mirroring `src/zmlx/kernels/` in Python.
pub const activations = @import("kernels/activations.zig");
pub const softmax = @import("kernels/softmax.zig");
pub const norms = @import("kernels/norms.zig");

// Reference all sub-modules so the test runner discovers their tests.
test {
    _ = activations;
    _ = softmax;
    _ = norms;
}
