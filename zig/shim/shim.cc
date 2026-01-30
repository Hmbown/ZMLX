// C++ implementation of the ZMLX shim.
//
// This file links against the MLX C++ library (libmlx) and exposes a
// minimal C ABI so that Zig (or any C-compatible caller) can create and
// launch custom Metal kernels.
//
// Build: compiled as a static library and linked into the Zig executable
// via build.zig.

#include "shim.h"

#include <mlx/mlx.h>

#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace mx = mlx::core;

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct zmlx_kernel_s {
    std::string              name;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string              source;
    std::string              header;
    bool                     ensure_row_contiguous;
    bool                     atomic_outputs;
};

struct zmlx_array_s {
    mx::array arr;
    explicit zmlx_array_s(mx::array a) : arr(std::move(a)) {}
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static mx::Dtype to_mlx_dtype(zmlx_dtype dt) {
    switch (dt) {
        case ZMLX_BOOL:      return mx::bool_;
        case ZMLX_UINT8:     return mx::uint8;
        case ZMLX_UINT16:    return mx::uint16;
        case ZMLX_UINT32:    return mx::uint32;
        case ZMLX_UINT64:    return mx::uint64;
        case ZMLX_INT8:      return mx::int8;
        case ZMLX_INT16:     return mx::int16;
        case ZMLX_INT32:     return mx::int32;
        case ZMLX_INT64:     return mx::int64;
        case ZMLX_FLOAT16:   return mx::float16;
        case ZMLX_FLOAT32:   return mx::float32;
        case ZMLX_BFLOAT16:  return mx::bfloat16;
        case ZMLX_COMPLEX64: return mx::complex64;
        default:             return mx::float32;
    }
}

// ---------------------------------------------------------------------------
// Kernel lifecycle
// ---------------------------------------------------------------------------

extern "C" zmlx_kernel_t zmlx_metal_kernel_create(
    const char*        name,
    const char* const* input_names,   uint32_t num_inputs,
    const char* const* output_names,  uint32_t num_outputs,
    const char*        source,
    const char*        header,
    bool               ensure_row_contiguous,
    bool               atomic_outputs
) {
    auto* k = new zmlx_kernel_s();
    k->name   = name;
    k->source = source;
    k->header = header ? header : "";
    k->ensure_row_contiguous = ensure_row_contiguous;
    k->atomic_outputs        = atomic_outputs;
    k->input_names.reserve(num_inputs);
    for (uint32_t i = 0; i < num_inputs; ++i)
        k->input_names.emplace_back(input_names[i]);
    k->output_names.reserve(num_outputs);
    for (uint32_t i = 0; i < num_outputs; ++i)
        k->output_names.emplace_back(output_names[i]);
    return k;
}

extern "C" void zmlx_metal_kernel_destroy(zmlx_kernel_t kernel) {
    delete kernel;
}

extern "C" int zmlx_metal_kernel_call(
    zmlx_kernel_t         kernel,
    const zmlx_array_t*   inputs,             uint32_t num_inputs,
    const uint32_t        grid[3],
    const uint32_t        threadgroup[3],
    const int64_t*        output_shapes_flat,
    const uint32_t*       output_ndims,
    uint32_t              num_outputs,
    const zmlx_dtype*     output_dtypes,
    const char* const*    template_names,      uint32_t num_templates,
    const zmlx_dtype*     template_dtypes,
    zmlx_array_t*         outputs
) {
    try {
        // --- Inputs ---
        std::vector<mx::array> inp_vec;
        inp_vec.reserve(num_inputs);
        for (uint32_t i = 0; i < num_inputs; ++i)
            inp_vec.push_back(inputs[i]->arr);

        // --- Output shapes (unflatten) ---
        std::vector<mx::Shape> out_shapes;
        out_shapes.reserve(num_outputs);
        uint32_t off = 0;
        for (uint32_t i = 0; i < num_outputs; ++i) {
            mx::Shape sh;
            for (uint32_t j = 0; j < output_ndims[i]; ++j)
                sh.push_back(static_cast<int>(output_shapes_flat[off + j]));
            out_shapes.push_back(std::move(sh));
            off += output_ndims[i];
        }

        // --- Output dtypes ---
        std::vector<mx::Dtype> out_dtypes;
        out_dtypes.reserve(num_outputs);
        for (uint32_t i = 0; i < num_outputs; ++i)
            out_dtypes.push_back(to_mlx_dtype(output_dtypes[i]));

        // --- Template args ---
        // MLX 0.30+: template args use variant<int, bool, Dtype>
        using TemplateArg = std::variant<int, bool, mx::Dtype>;
        std::vector<std::pair<std::string, TemplateArg>> tmpl_args;
        tmpl_args.reserve(num_templates);
        for (uint32_t i = 0; i < num_templates; ++i)
            tmpl_args.emplace_back(template_names[i],
                                   TemplateArg{to_mlx_dtype(template_dtypes[i])});

        // --- Grid / threadgroup ---
        auto g  = std::make_tuple(static_cast<int>(grid[0]),
                                  static_cast<int>(grid[1]),
                                  static_cast<int>(grid[2]));
        auto tg = std::make_tuple(static_cast<int>(threadgroup[0]),
                                  static_cast<int>(threadgroup[1]),
                                  static_cast<int>(threadgroup[2]));

        // --- Launch ---
        // Step 1: Create the kernel callable (MLX caches compiled Metal
        //         by source string, so repeated calls are cheap).
        // Step 2: Invoke it with call-time parameters.
        //
        // NOTE: The exact C++ API may vary across MLX versions.
        // Adjust the calls below to match your installed MLX header
        // (<mlx/fast.h> or <mlx/mlx.h>).  Two common patterns:
        //
        //   (A) Two-step: auto fn = metal_kernel(name, ...);
        //                 auto result = fn({inputs, shapes, ...});
        //
        //   (B) Single:   auto result = metal_kernel(name, ..., inputs, ...);
        //
        // We use pattern (A) here.  If your MLX version uses (B), merge
        // the two calls.
        auto kernel_fn = mx::fast::metal_kernel(
            kernel->name,
            kernel->input_names,
            kernel->output_names,
            kernel->source,
            kernel->header,
            kernel->ensure_row_contiguous,
            kernel->atomic_outputs
        );

        auto result = kernel_fn(
            inp_vec,
            out_shapes,
            out_dtypes,
            g,
            tg,
            tmpl_args,
            std::nullopt,   // init_value
            false,          // verbose
            mx::default_stream(mx::default_device())
        );

        // --- Copy outputs ---
        for (uint32_t i = 0; i < num_outputs && i < result.size(); ++i)
            outputs[i] = new zmlx_array_s(std::move(result[i]));

        return 0;
    } catch (...) {
        return -1;
    }
}

// ---------------------------------------------------------------------------
// Array helpers
// ---------------------------------------------------------------------------

extern "C" zmlx_array_t zmlx_array_from_float32(
    const float* data, const int64_t* shape, uint32_t ndim
) {
    mx::Shape sh;
    for (uint32_t i = 0; i < ndim; ++i)
        sh.push_back(static_cast<int>(shape[i]));
    return new zmlx_array_s(mx::array(data, sh, mx::float32));
}

extern "C" zmlx_array_t zmlx_array_from_int32(
    const int32_t* data, const int64_t* shape, uint32_t ndim
) {
    mx::Shape sh;
    for (uint32_t i = 0; i < ndim; ++i)
        sh.push_back(static_cast<int>(shape[i]));
    return new zmlx_array_s(mx::array(data, sh, mx::int32));
}

extern "C" void zmlx_array_destroy(zmlx_array_t arr) {
    delete arr;
}

extern "C" void zmlx_eval(zmlx_array_t arr) {
    mx::eval(arr->arr);
}

extern "C" const float* zmlx_array_data_float32(zmlx_array_t arr) {
    mx::eval(arr->arr);
    return arr->arr.data<float>();
}

extern "C" uint32_t zmlx_array_ndim(zmlx_array_t arr) {
    return static_cast<uint32_t>(arr->arr.ndim());
}

extern "C" int64_t zmlx_array_dim(zmlx_array_t arr, uint32_t axis) {
    return static_cast<int64_t>(arr->arr.shape(static_cast<int>(axis)));
}

extern "C" uint64_t zmlx_array_size(zmlx_array_t arr) {
    return static_cast<uint64_t>(arr->arr.size());
}

static zmlx_dtype from_mlx_dtype(mx::Dtype dt) {
    if (dt == mx::bool_)     return ZMLX_BOOL;
    if (dt == mx::uint8)     return ZMLX_UINT8;
    if (dt == mx::uint16)    return ZMLX_UINT16;
    if (dt == mx::uint32)    return ZMLX_UINT32;
    if (dt == mx::uint64)    return ZMLX_UINT64;
    if (dt == mx::int8)      return ZMLX_INT8;
    if (dt == mx::int16)     return ZMLX_INT16;
    if (dt == mx::int32)     return ZMLX_INT32;
    if (dt == mx::int64)     return ZMLX_INT64;
    if (dt == mx::float16)   return ZMLX_FLOAT16;
    if (dt == mx::float32)   return ZMLX_FLOAT32;
    if (dt == mx::bfloat16)  return ZMLX_BFLOAT16;
    if (dt == mx::complex64) return ZMLX_COMPLEX64;
    return ZMLX_FLOAT32;
}

extern "C" zmlx_dtype zmlx_array_dtype(zmlx_array_t arr) {
    return from_mlx_dtype(arr->arr.dtype());
}

extern "C" zmlx_array_t zmlx_array_zeros(
    const int64_t* shape, uint32_t ndim, zmlx_dtype dtype
) {
    mx::Shape sh;
    for (uint32_t i = 0; i < ndim; ++i)
        sh.push_back(static_cast<int>(shape[i]));
    return new zmlx_array_s(mx::zeros(sh, to_mlx_dtype(dtype)));
}

extern "C" zmlx_array_t zmlx_array_ones(
    const int64_t* shape, uint32_t ndim, zmlx_dtype dtype
) {
    mx::Shape sh;
    for (uint32_t i = 0; i < ndim; ++i)
        sh.push_back(static_cast<int>(shape[i]));
    return new zmlx_array_s(mx::ones(sh, to_mlx_dtype(dtype)));
}

extern "C" const void* zmlx_array_data_ptr(zmlx_array_t arr) {
    mx::eval(arr->arr);
    return static_cast<const void*>(arr->arr.data<uint8_t>());
}
