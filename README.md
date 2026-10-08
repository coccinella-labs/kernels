<p align="center">
  <img src="https://raw.githubusercontent.com/coccinella-labs/kernels/main/.github/assets/thumbnail.png" alt="kernels" width="100%">
</p>

# Metal Compute Kernels for Swift

A runnable set of 23 Metal compute kernels with a demo that dispatches 16 of them,
checks the results against values computed in Swift, and prints measured timings.

| Package | Target | Platform |
|---|---|---|
| MetalKernels | executable | macOS 12+ |

This is a demonstration package, not a library. There is no product you import. The
value is the source: read `Sources/MetalKernels/kernels.metal`, then run the demo to
see each kernel's output and timing on your own GPU.

## Run it

```bash
git clone https://github.com/coccinella-labs/kernels
cd kernels
swift build -c release
./.build/release/MetalKernels
```

Requires macOS 12 or later on Apple Silicon. The demo exits non-zero if any
correctness check fails, so it works as a CI gate. No Metal device, no GPU work.

## What it demonstrates

Eleven sections, in roughly the order you would build them:

1. Elementwise array add, the whole point of a GPU in one line
2. A worked CUDA to Metal translation
3. Shared memory and `threadgroup_barrier`
4. 2D convolution
5. Fused multiply-add
6. Tiled matrix multiply
7. CPU versus GPU timing
8. Neural network layers: sigmoid, tanh, GELU
9. Timing distribution
10. Threadgroup size tuning at 32, 64, 128, 256
11. Batched matmul

## Correctness

The demo computes expected values in Swift and compares them, printing `FAILED` with
both numbers on a mismatch. Five checks run: sigmoid, tanh, gelu, depthwise output
size, and batched matmul size.

Failures are counted. If any check fails the summary banner reports the failure
instead of success and the process exits 1.

Two kernels are limited by their implementation, and the demo says so at runtime:

- `sumReduction` reduces within a single 32-element threadgroup with no cross-group
  pass, so its input length must be a multiple of 32 and it returns one partial sum
  per group.
- `exclusiveScan` requires 32 elements or fewer.

## Kernels in this repository

23 kernels are declared in `Sources/MetalKernels/kernels.metal`. The demo dispatches
16 of them by name. These 7 are shader-only: they are compiled but never called, so
their Swift wrappers do not exist.

| Kernel | Why it is not dispatched |
|---|---|
| `vector_add` | shown for CUDA comparison only; `add_arrays` does the same work |
| `tiled_matrix_multiply` | shader written, no Swift wrapper |
| `batch_norm` | shader written, no Swift wrapper |
| `layer_norm` | shader written, no Swift wrapper |
| `conv2d_batch` | shader written, no Swift wrapper |
| `gaussian_blur_x` | shader written, no Swift wrapper |
| `matrix_vector_multiply` | shader written, no Swift wrapper |

That table is checked by the test suite. Adding a wrapper for one of these, or
removing a kernel, will fail CI until the README is updated.

## CUDA to Metal translation

The CUDA examples are worked translations by hand. Nothing in this repository
translates code automatically, and porting a kernel means writing the Metal version
yourself.

CUDA:

```cpp
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}
```

Metal, from `Sources/MetalKernels/kernels.metal`:

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

The differences worth remembering:

- `blockIdx` and `threadIdx` collapse into `thread_position_in_grid`.
- Every pointer carries an address space qualifier, `device` for GPU memory, and
  arrives through a `[[buffer(n)]]` binding.
- Grid size comes from the dispatch call rather than an in-kernel bounds check.

[Swift API quick reference](QUICK_REFERENCE.md) covers the wrapper surface.

## Measured timings

MacBook Air, Apple M1, 8 GB, macOS 27.0.1, Swift 6.4 release build. These are wall
clock on an idle machine and will differ on your hardware.

```
Array add, 10000 elements   GPU 0.27 ms    CPU 0.02 ms
64x64 matmul                GPU 0.31 ms    CPU 0.93 ms
```

Two things to be clear about. These are CPU-side wall clock around a dispatch, not
GPU time from a timestamp query, so they include submission overhead and say nothing
about how busy the GPU was. The demo does not measure utilization; use Xcode's Metal
Debugger or `MTLCounter` for that.

The CPU beats the GPU on a 10000-element elementwise add, by a wide margin. That is
the expected result and it is worth internalizing: dispatch overhead dominates at
this size, and the crossover sits well above it. The demo prints its own measurements
in the summary banner rather than fixed text, so the numbers you see are from your run.

Timings vary run to run. Across repeated runs the array-add benchmark ranged from
0.235 ms to 0.577 ms average, a spread of over 150 percent, which is why the demo
reports a standard deviation instead of a single figure.

## Layout

```
Sources/MetalKernels/kernels.metal   all 23 kernels
Sources/MetalKernels/main.swift      demo, checks, and timings
Tests/MetalKernelsTests/             README drift guard
```

## Development

```bash
swift build -c release
swift test
```

`Tests/MetalKernelsTests/ReadmeDriftTests.swift` keeps this file honest: the kernel
count, the shader-only table, the code samples, and the platform claims are all
verified against the sources, and known-false claims cannot come back. It reads files
as text and creates no Metal device, so it runs on CI runners without a GPU.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and
[RELEASE.md](RELEASE.md) for the release process. [CHANGELOG.md](CHANGELOG.md)
records entries up to 1.0.0.

## License

MIT. See [LICENSE](LICENSE).
