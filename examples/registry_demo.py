import mlx.core as mx

from zmlx import elementwise, msl, registry


def run_registry_demo():
    print("--- ZMLX Registry Demo ---")
    
    # 1. Create some kernels
    exp_fast = elementwise.unary(
        name="kk_exp_demo",
        expr="metal::exp(x)",
        compute_dtype=mx.float32,
        header=msl.DEFAULT_HEADER,
    )
    relu_fast = elementwise.unary(
        name="kk_relu_demo",
        expr="metal::max(x, (T)0)",
        compute_dtype=mx.float32,
        header=msl.DEFAULT_HEADER,
    )
    
    x = mx.random.normal((1024,))
    
    # Trigger compilation
    exp_fast(x)
    relu_fast(x)
    
    # 2. List kernels
    kernels = registry.list_kernels()
    print(f"Cached kernels: {kernels}")
    
    # 3. Check stats
    # We can retrieve the kernel object from the global cache if we really want, 
    # but usually we just want to see the count.
    print(f"Total kernels in cache: {registry.cache_size()}")
    
    # 4. Clear cache
    print("Clearing cache...")
    registry.clear_cache()
    print(f"Cache size after clearing: {registry.cache_size()}")

if __name__ == "__main__":
    run_registry_demo()
