// Minimal C ABI shim for MLX metal_kernel functionality.
// Wraps the MLX C++ API so Zig (or any C-compatible language) can create
// and launch custom Metal kernels.
//
// This shim exists because MLX-C does not yet expose mx.fast.metal_kernel.
// When MLX-C gains that API, this file can be retired.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------
typedef struct zmlx_kernel_s* zmlx_kernel_t;
typedef struct zmlx_array_s*  zmlx_array_t;

// ---------------------------------------------------------------------------
// Dtype enum — values must stay in sync with mlx::core::Dtype::Val
// ---------------------------------------------------------------------------
typedef enum {
    ZMLX_BOOL     = 0,
    ZMLX_UINT8    = 1,
    ZMLX_UINT16   = 2,
    ZMLX_UINT32   = 3,
    ZMLX_UINT64   = 4,
    ZMLX_INT8     = 5,
    ZMLX_INT16    = 6,
    ZMLX_INT32    = 7,
    ZMLX_INT64    = 8,
    ZMLX_FLOAT16  = 9,
    ZMLX_FLOAT32  = 10,
    ZMLX_BFLOAT16 = 11,
    ZMLX_COMPLEX64 = 12,
} zmlx_dtype;

// ---------------------------------------------------------------------------
// Kernel lifecycle
// ---------------------------------------------------------------------------

/// Create a kernel handle (stores the spec; Metal compilation is deferred to
/// the first call, and then cached by MLX internally).
zmlx_kernel_t zmlx_metal_kernel_create(
    const char*        name,
    const char* const* input_names,   uint32_t num_inputs,
    const char* const* output_names,  uint32_t num_outputs,
    const char*        source,
    const char*        header,
    bool               ensure_row_contiguous,
    bool               atomic_outputs
);

/// Destroy a previously created kernel handle.
void zmlx_metal_kernel_destroy(zmlx_kernel_t kernel);

/// Launch the kernel.
///
/// output_shapes_flat : all output shapes concatenated into one flat array.
/// output_ndims       : number of dimensions for each output (used to split
///                      the flat array).
/// outputs            : caller-allocated array of size num_outputs; filled on
///                      return with newly created zmlx_array_t handles.
///
/// Returns 0 on success, -1 on error.
int zmlx_metal_kernel_call(
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
);

// ---------------------------------------------------------------------------
// Array helpers  (only the subset needed for the demo / tests)
// ---------------------------------------------------------------------------

/// Create a float32 array from host data.  The data is copied.
zmlx_array_t  zmlx_array_from_float32(const float*   data,
                                       const int64_t* shape,
                                       uint32_t       ndim);

/// Create an int32 array from host data.  The data is copied.
zmlx_array_t  zmlx_array_from_int32(const int32_t* data,
                                     const int64_t* shape,
                                     uint32_t       ndim);

/// Destroy an array handle.
void          zmlx_array_destroy(zmlx_array_t arr);

/// Evaluate an array (force computation).
void          zmlx_eval(zmlx_array_t arr);

/// Return a pointer to the evaluated float32 data.
/// The array must have dtype float32 and must be evaluated first.
const float*  zmlx_array_data_float32(zmlx_array_t arr);

/// Number of dimensions.
uint32_t      zmlx_array_ndim(zmlx_array_t arr);

/// Size of the given axis.
int64_t       zmlx_array_dim(zmlx_array_t arr, uint32_t axis);

/// Total number of elements.
uint64_t      zmlx_array_size(zmlx_array_t arr);

/// Return the dtype of an array.
zmlx_dtype    zmlx_array_dtype(zmlx_array_t arr);

/// Create an array of zeros with the given shape and dtype.
zmlx_array_t  zmlx_array_zeros(const int64_t* shape, uint32_t ndim, zmlx_dtype dtype);

/// Create an array of ones with the given shape and dtype.
zmlx_array_t  zmlx_array_ones(const int64_t* shape, uint32_t ndim, zmlx_dtype dtype);

/// Generic data pointer (array must be evaluated first).
const void*   zmlx_array_data_ptr(zmlx_array_t arr);

#ifdef __cplusplus
}
#endif
