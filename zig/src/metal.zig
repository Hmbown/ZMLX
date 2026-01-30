/// MetalKernel wrapper — Zig port of `src/zmlx/metal.py` + `src/zmlx/cache.py`.
///
/// Provides:
///   - `MetalKernel` struct wrapping the C shim handle
///   - `KernelCache` keyed on source hash (SHA-256)
///   - `kernel()` factory that checks the cache first
///   - `Array` helper wrapping the shim's array operations
const std = @import("std");
const Allocator = std.mem.Allocator;

/// C bindings from zig/shim/shim.h.
pub const c = @cImport({
    @cInclude("shim.h");
});

// ---------------------------------------------------------------------------
// Dtype
// ---------------------------------------------------------------------------

pub const Dtype = enum(u32) {
    bool_ = c.ZMLX_BOOL,
    uint8 = c.ZMLX_UINT8,
    uint16 = c.ZMLX_UINT16,
    uint32 = c.ZMLX_UINT32,
    uint64 = c.ZMLX_UINT64,
    int8 = c.ZMLX_INT8,
    int16 = c.ZMLX_INT16,
    int32 = c.ZMLX_INT32,
    int64 = c.ZMLX_INT64,
    float16 = c.ZMLX_FLOAT16,
    float32 = c.ZMLX_FLOAT32,
    bfloat16 = c.ZMLX_BFLOAT16,
    complex64 = c.ZMLX_COMPLEX64,

    pub fn toC(self: Dtype) c.zmlx_dtype {
        return @intCast(@intFromEnum(self));
    }
};

// ---------------------------------------------------------------------------
// Array — thin wrapper around the shim's zmlx_array_t
// ---------------------------------------------------------------------------

pub const Array = struct {
    handle: c.zmlx_array_t,

    /// Create a float32 array from host data.
    pub fn fromFloat32(data: []const f32, shape: []const i64) Array {
        return .{
            .handle = c.zmlx_array_from_float32(
                data.ptr,
                shape.ptr,
                @intCast(shape.len),
            ),
        };
    }

    /// Create an int32 array from host data.
    pub fn fromInt32(data: []const i32, shape: []const i64) Array {
        return .{
            .handle = c.zmlx_array_from_int32(
                data.ptr,
                shape.ptr,
                @intCast(shape.len),
            ),
        };
    }

    /// Force evaluation.
    pub fn eval(self: Array) void {
        c.zmlx_eval(self.handle);
    }

    /// Return a slice over the evaluated float32 data.
    pub fn dataFloat32(self: Array) []const f32 {
        const len = c.zmlx_array_size(self.handle);
        const ptr = c.zmlx_array_data_float32(self.handle);
        return ptr[0..len];
    }

    /// Number of dimensions.
    pub fn ndim(self: Array) u32 {
        return c.zmlx_array_ndim(self.handle);
    }

    /// Size of the given axis.
    pub fn dim(self: Array, axis: u32) i64 {
        return c.zmlx_array_dim(self.handle, axis);
    }

    /// Total number of elements.
    pub fn size(self: Array) u64 {
        return c.zmlx_array_size(self.handle);
    }

    /// Return the dtype of the array.
    pub fn dtype(self: Array) Dtype {
        return @enumFromInt(c.zmlx_array_dtype(self.handle));
    }

    /// Create an array of zeros with the given shape and dtype.
    pub fn zeros(shape: []const i64, dt: Dtype) Array {
        return .{
            .handle = c.zmlx_array_zeros(
                shape.ptr,
                @intCast(shape.len),
                dt.toC(),
            ),
        };
    }

    /// Create an array of ones with the given shape and dtype.
    pub fn ones(shape: []const i64, dt: Dtype) Array {
        return .{
            .handle = c.zmlx_array_ones(
                shape.ptr,
                @intCast(shape.len),
                dt.toC(),
            ),
        };
    }

    /// Free the underlying MLX array.
    pub fn deinit(self: Array) void {
        c.zmlx_array_destroy(self.handle);
    }
};

// ---------------------------------------------------------------------------
// MetalKernelSpec
// ---------------------------------------------------------------------------

pub const MetalKernelSpec = struct {
    name: []const u8,
    input_names: []const []const u8,
    output_names: []const []const u8,
    source: []const u8,
    header: []const u8 = "",
    ensure_row_contiguous: bool = true,
    atomic_outputs: bool = false,
};

// ---------------------------------------------------------------------------
// MetalKernel
// ---------------------------------------------------------------------------

pub const MetalKernel = struct {
    handle: c.zmlx_kernel_t,
    spec: MetalKernelSpec,
    allocator: Allocator,

    // Null-terminated copies of strings passed to the C shim.
    c_name: [:0]u8,
    c_source: [:0]u8,
    c_header: [:0]u8,
    c_input_names_z: [][:0]u8, // null-terminated copies
    c_output_names_z: [][:0]u8,
    c_input_ptrs: [][*:0]const u8, // pointers into c_input_names_z
    c_output_ptrs: [][*:0]const u8,

    pub const CallOptions = struct {
        grid: [3]u32 = .{ 0, 0, 0 }, // 0 = auto
        threadgroup: [3]u32 = .{ 0, 0, 0 }, // 0 = auto
        template_names: []const []const u8 = &.{},
        template_dtypes: []const Dtype = &.{},
        output_shapes: ?[]const []const i64 = null,
        output_dtypes: ?[]const Dtype = null,
    };

    /// Create a MetalKernel from a spec.
    /// All string parameters are copied with null terminators for C interop.
    pub fn init(allocator: Allocator, spec: MetalKernelSpec) !MetalKernel {
        // Create null-terminated copies of all strings.
        const c_name = try allocator.dupeZ(u8, spec.name);
        errdefer allocator.free(c_name);
        const c_source = try allocator.dupeZ(u8, spec.source);
        errdefer allocator.free(c_source);
        const c_header = try allocator.dupeZ(u8, spec.header);
        errdefer allocator.free(c_header);

        // Input names: null-terminated copies + pointer array.
        const c_in_z = try allocator.alloc([:0]u8, spec.input_names.len);
        errdefer allocator.free(c_in_z);
        const c_in_ptrs = try allocator.alloc([*:0]const u8, spec.input_names.len);
        errdefer allocator.free(c_in_ptrs);
        for (spec.input_names, 0..) |name, i| {
            c_in_z[i] = try allocator.dupeZ(u8, name);
            c_in_ptrs[i] = c_in_z[i].ptr;
        }

        // Output names: same pattern.
        const c_out_z = try allocator.alloc([:0]u8, spec.output_names.len);
        errdefer allocator.free(c_out_z);
        const c_out_ptrs = try allocator.alloc([*:0]const u8, spec.output_names.len);
        errdefer allocator.free(c_out_ptrs);
        for (spec.output_names, 0..) |name, i| {
            c_out_z[i] = try allocator.dupeZ(u8, name);
            c_out_ptrs[i] = c_out_z[i].ptr;
        }

        const handle = c.zmlx_metal_kernel_create(
            c_name.ptr,
            @ptrCast(c_in_ptrs.ptr),
            @intCast(spec.input_names.len),
            @ptrCast(c_out_ptrs.ptr),
            @intCast(spec.output_names.len),
            c_source.ptr,
            c_header.ptr,
            spec.ensure_row_contiguous,
            spec.atomic_outputs,
        );

        return .{
            .handle = handle,
            .spec = spec,
            .allocator = allocator,
            .c_name = c_name,
            .c_source = c_source,
            .c_header = c_header,
            .c_input_names_z = c_in_z,
            .c_output_names_z = c_out_z,
            .c_input_ptrs = c_in_ptrs,
            .c_output_ptrs = c_out_ptrs,
        };
    }

    /// Launch the kernel.
    pub fn call(
        self: *const MetalKernel,
        inputs: []const Array,
        opts: CallOptions,
    ) ![]Array {
        const allocator = self.allocator;
        const num_outputs: u32 = @intCast(self.spec.output_names.len);

        // --- Build C-compatible input handle array ---
        const c_inputs = try allocator.alloc(c.zmlx_array_t, inputs.len);
        defer allocator.free(c_inputs);
        for (c_inputs, inputs) |*dst, inp| {
            dst.* = inp.handle;
        }

        // --- Grid / threadgroup (auto-size if zero) ---
        var grid = opts.grid;
        var threadgroup = opts.threadgroup;
        if (grid[0] == 0) {
            // Elementwise default: 1 thread per element of first input.
            const n: u32 = if (inputs.len > 0)
                @intCast(inputs[0].size())
            else
                1;
            grid = .{ n, 1, 1 };
        }
        if (threadgroup[0] == 0) {
            threadgroup[0] = defaultThreadgroupX(grid[0]);
            threadgroup[1] = 1;
            threadgroup[2] = 1;
        }

        // --- Output shapes ---
        // Default: each output matches the first input's shape.
        var shapes_flat_buf: [64]i64 = undefined;
        var ndims_buf: [16]u32 = undefined;
        var shapes_flat: []const i64 = undefined;
        var ndims: []const u32 = undefined;

        if (opts.output_shapes) |os| {
            // Flatten caller-provided shapes.
            var flat_len: usize = 0;
            for (os, 0..) |s, i| {
                ndims_buf[i] = @intCast(s.len);
                for (s) |d| {
                    shapes_flat_buf[flat_len] = d;
                    flat_len += 1;
                }
            }
            shapes_flat = shapes_flat_buf[0..flat_len];
            ndims = ndims_buf[0..os.len];
        } else {
            // Default: copy first input's shape for each output.
            const first = inputs[0];
            const nd = first.ndim();
            var flat_len: usize = 0;
            for (0..num_outputs) |i| {
                ndims_buf[i] = nd;
                for (0..nd) |ax| {
                    shapes_flat_buf[flat_len] = first.dim(@intCast(ax));
                    flat_len += 1;
                }
            }
            shapes_flat = shapes_flat_buf[0..flat_len];
            ndims = ndims_buf[0..num_outputs];
        }

        // --- Output dtypes ---
        var dtype_buf: [16]c.zmlx_dtype = undefined;
        const c_dtypes: [*]const c.zmlx_dtype = blk: {
            if (opts.output_dtypes) |od| {
                for (od, 0..) |dt, i| {
                    dtype_buf[i] = dt.toC();
                }
                break :blk &dtype_buf;
            } else {
                // Default: float32 for each output.
                for (0..num_outputs) |i| {
                    dtype_buf[i] = Dtype.float32.toC();
                }
                break :blk &dtype_buf;
            }
        };

        // --- Template args (must be null-terminated for C) ---
        var c_tmpl_names_z: [16][:0]u8 = undefined;
        var c_tmpl_names: [16][*:0]const u8 = undefined;
        var c_tmpl_dtypes: [16]c.zmlx_dtype = undefined;
        const num_templates: u32 = @intCast(opts.template_names.len);
        for (opts.template_names, 0..) |tn, i| {
            c_tmpl_names_z[i] = try allocator.dupeZ(u8, tn);
            c_tmpl_names[i] = c_tmpl_names_z[i].ptr;
        }
        defer for (0..num_templates) |i| {
            allocator.free(c_tmpl_names_z[i]);
        };
        for (opts.template_dtypes, 0..) |td, i| {
            c_tmpl_dtypes[i] = td.toC();
        }

        // --- Allocate output handles ---
        const out_handles = try allocator.alloc(c.zmlx_array_t, num_outputs);
        defer allocator.free(out_handles);

        // --- Call the shim ---
        // Use @ptrCast on all pointer args to bridge Zig ↔ C pointer types.
        const rc = c.zmlx_metal_kernel_call(
            self.handle,
            @ptrCast(c_inputs.ptr),
            @intCast(inputs.len),
            &grid,
            &threadgroup,
            @ptrCast(shapes_flat.ptr),
            @ptrCast(ndims.ptr),
            num_outputs,
            @ptrCast(c_dtypes),
            @ptrCast(&c_tmpl_names),
            num_templates,
            @ptrCast(&c_tmpl_dtypes),
            @ptrCast(out_handles.ptr),
        );

        if (rc != 0) return error.KernelCallFailed;

        // --- Wrap output handles ---
        const outputs = try allocator.alloc(Array, num_outputs);
        for (outputs, out_handles) |*dst, h| {
            dst.* = .{ .handle = h };
        }
        return outputs;
    }

    /// Free the kernel handle and all null-terminated string copies.
    pub fn deinit(self: *MetalKernel) void {
        c.zmlx_metal_kernel_destroy(self.handle);
        for (self.c_input_names_z) |z| self.allocator.free(z);
        for (self.c_output_names_z) |z| self.allocator.free(z);
        self.allocator.free(self.c_input_names_z);
        self.allocator.free(self.c_output_names_z);
        self.allocator.free(self.c_input_ptrs);
        self.allocator.free(self.c_output_ptrs);
        self.allocator.free(self.c_name);
        self.allocator.free(self.c_source);
        self.allocator.free(self.c_header);
    }
};

// ---------------------------------------------------------------------------
// Default threadgroup heuristic  (matches Python _default_threadgroup_x)
// ---------------------------------------------------------------------------

pub fn defaultThreadgroupX(n_threads: u32) u32 {
    const candidates = [_]u32{ 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 };
    const limit = if (n_threads > 0) n_threads else 1;
    for (candidates) |cand| {
        if (cand <= limit) return cand;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// KernelCache  (matches Python KernelCache / GLOBAL_KERNEL_CACHE)
// ---------------------------------------------------------------------------

/// SHA-256 hex digest of a byte slice (matches Python's `_sha256`).
fn sha256Hex(data: []const u8) [64]u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}

/// Cache key — mirrors Python's `KernelCacheKey`.
pub const KernelCacheKey = struct {
    name: []const u8,
    source_hash: [64]u8,
    header_hash: [64]u8,

    pub fn fromSpec(spec: MetalKernelSpec) KernelCacheKey {
        return .{
            .name = spec.name,
            .source_hash = sha256Hex(spec.source),
            .header_hash = sha256Hex(if (spec.header.len > 0) spec.header else ""),
        };
    }
};

/// In-process kernel cache keyed on (name, source_hash, header_hash).
pub const KernelCache = struct {
    const Map = std.StringHashMap(MetalKernel);

    map: Map,
    allocator: Allocator,

    pub fn init(allocator: Allocator) KernelCache {
        return .{
            .map = Map.init(allocator),
            .allocator = allocator,
        };
    }

    /// Build a composite key string for the hash map.
    fn makeKey(allocator: Allocator, ck: KernelCacheKey) ![]u8 {
        return std.fmt.allocPrint(allocator, "{s}:{s}:{s}", .{
            ck.name,
            &ck.source_hash,
            &ck.header_hash,
        });
    }

    /// Get or create a kernel from the cache.
    pub fn getOrCreate(self: *KernelCache, spec: MetalKernelSpec) !*MetalKernel {
        const ck = KernelCacheKey.fromSpec(spec);
        const key = try makeKey(self.allocator, ck);

        const gop = try self.map.getOrPut(key);
        if (gop.found_existing) {
            self.allocator.free(key);
            return gop.value_ptr;
        }

        gop.value_ptr.* = try MetalKernel.init(self.allocator, spec);
        return gop.value_ptr;
    }

    pub fn size(self: *const KernelCache) usize {
        return self.map.count();
    }

    pub fn clear(self: *KernelCache) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.map.clearAndFree();
    }

    pub fn deinit(self: *KernelCache) void {
        self.clear();
        self.map.deinit();
    }
};

/// Global kernel cache — module-level singleton.
var global_cache_backing: ?KernelCache = null;

pub fn globalCache(allocator: Allocator) *KernelCache {
    if (global_cache_backing == null) {
        global_cache_backing = KernelCache.init(allocator);
    }
    return &global_cache_backing.?;
}

/// Convenience factory: build or retrieve a cached MetalKernel.
/// Mirrors Python's `metal.kernel(...)` function.
pub fn kernelFromSpec(allocator: Allocator, spec: MetalKernelSpec, use_cache: bool) !*MetalKernel {
    if (!use_cache) {
        const ptr = try allocator.create(MetalKernel);
        ptr.* = try MetalKernel.init(allocator, spec);
        return ptr;
    }
    return globalCache(allocator).getOrCreate(spec);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "defaultThreadgroupX" {
    try std.testing.expectEqual(@as(u32, 512), defaultThreadgroupX(1024));
    try std.testing.expectEqual(@as(u32, 512), defaultThreadgroupX(512));
    try std.testing.expectEqual(@as(u32, 256), defaultThreadgroupX(511));
    try std.testing.expectEqual(@as(u32, 256), defaultThreadgroupX(256));
    try std.testing.expectEqual(@as(u32, 1), defaultThreadgroupX(1));
    try std.testing.expectEqual(@as(u32, 1), defaultThreadgroupX(0));
}

test "sha256 hex" {
    const hex = sha256Hex("hello");
    const expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
    try std.testing.expectEqualStrings(expected, &hex);
}

test "KernelCacheKey from spec" {
    const spec = MetalKernelSpec{
        .name = "test",
        .input_names = &.{"inp"},
        .output_names = &.{"out"},
        .source = "x*x",
        .header = "",
    };
    const ck = KernelCacheKey.fromSpec(spec);
    try std.testing.expectEqualStrings("test", ck.name);
    // source_hash should be SHA-256 of "x*x"
    const expected_src = sha256Hex("x*x");
    try std.testing.expectEqualStrings(&expected_src, &ck.source_hash);
}
