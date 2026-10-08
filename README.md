<p align="center">
  <img src="https://raw.githubusercontent.com/coccinella-labs/kernels/main/.github/assets/thumbnail.png" alt="kernels" width="100%">
</p>

**Metal Compute Kernels for Swift**

| Package | Target | Platform |
|---|---|---|
| MetalKernels | executable | macOS 12+ |

`Package.swift` declares the package and target as `MetalKernels` and the minimum platform as macOS 12. There is no iOS target; the only iOS mention in the demo is a future item in its closing banner.

A Swift-based framework for GPU compute on Apple Silicon using Apple’s Metal API.

Metal Compute Kernels enables writing high-performance GPU kernels directly in Swift and provides direct CUDA-to-Metal execution model translations.

The CUDA examples below are worked translations, not an automated translator. `Sources/MetalKernels/kernels.metal` ships 23 Metal kernels, and `Sources/MetalKernels/main.swift` runs them alongside the equivalent CUDA source and prints both for comparison. Porting a kernel means editing the Metal version by hand.

Not every kernel has a Swift wrapper. `tiled_matrix_multiply` is shader-only, and `vector_add` is used for the side-by-side comparison rather than being called. The demo asserts its results against values computed in Swift and prints `FAILED` with the mismatch when one occurs, rather than assuming success.

Two wrappers are limited to a single 32-element threadgroup, because they use shared memory without a cross-group pass. `exclusiveScan` requires 32 elements or fewer, and `sumReduction` requires a length that is a multiple of 32 and returns one partial sum per group.

CUDA kernel example:

```cpp
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}
```

Metal translation:

```metal
kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}
```

Key differences include the replacement of `blockIdx`/`threadIdx` with `thread_position_in_grid`, explicit `device` address space qualifiers with buffer bindings, and dispatch-controlled grid sizing instead of in-kernel bounds checks.

[Swift API quick reference](QUICK_REFERENCE.md)
