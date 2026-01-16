# NVGPU APIs

This document explains the Python APIs for CuTe DSL NVIDIA GPU-specific operations. These APIs provide access to hardware-specific features like tensor cores, async copy operations, and warp-level instructions.

## Overview

NVGPU APIs expose NVIDIA GPU hardware features including:
- Copy operations (universal, async, warp-level)
- MMA operations (tensor cores)
- Warp-level matrix load operations

## nvgpu.CopyUniversalOp

Universal copy operation for general-purpose data movement.

### Purpose

A general-purpose copy operation that works across different memory spaces (global, shared, register). Used for synchronous copies.

### Python API Usage

```python
# Create copy atom with universal op
copy_atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(),
    cutlass.Float32
)

# Use for synchronous copy
cute.copy(copy_atom, src, dst)
```

### When to Use

- Synchronous copies between any memory spaces
- When you don't need async behavior
- General-purpose data movement

## nvgpu.cpasync.CopyG2SOp

Asynchronous copy operation from global to shared memory.

### Purpose

Hardware-accelerated async copy from global memory to shared memory using `cp.async` instructions. Enables pipelining of memory transfers with computation.

### Python API Usage

```python
# Create async copy atom
async_copy_atom = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(
        cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
    ),
    cutlass.Float16
)

# Create tiled copy
tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, thr_layout, val_layout)

# Async copy
cute.copy(tiled_copy, tSgA, tDsA)
cute.arch.cp_async_commit_group()
```

### Cache Modes

- `LoadCacheMode.GLOBAL`: Cache in global cache
- Other cache modes available depending on architecture

### Key Points

- Only works for global → shared memory copies
- Requires `cp_async_commit_group()` and `cp_async_wait_group()`
- Enables pipelining for better performance

## nvgpu.warp.LdMatrix8x8x16bOp

Warp-level matrix load operation for tensor cores.

### Purpose

Loads matrices from shared memory to registers in the format expected by tensor core MMA operations. This is the standard way to feed data into tensor cores.

### Python API Usage

```python
# Create load matrix atom
ldmatrix_atom = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(
        transpose=False,      # Whether to transpose during load
        num_matrices=4        # Number of matrices (1, 2, or 4)
    ),
    cutlass.Float16
)

# Create tiled copy aligned with MMA
tiled_copy = cute.make_tiled_copy_A(ldmatrix_atom, tiled_mma)

# Load from shared memory to register
cute.copy(tiled_copy, tCsA, tCrA)
```

### Parameters

- `transpose`: If `True`, transposes the matrix during load
- `num_matrices`: Number of 8x8 matrices to load (1, 2, or 4)

### When to Use

- Loading data for tensor core MMA operations
- Shared memory → register transfers for MMA operands
- Typically used with `make_tiled_copy_A` or `make_tiled_copy_B`

## nvgpu.warp.MmaF16BF16Op

Warp-level MMA operation for FP16/BF16 inputs.

### Purpose

Creates an MMA atom for FP16/BF16 matrix multiplication using tensor cores.

### Python API Usage

```python
# Create MMA operation
mma_op = cute.nvgpu.warp.MmaF16BF16Op()

# Create MMA atom
mma_atom = cute.make_mma_atom(mma_op)

# Create tiled MMA
tiled_mma = cute.make_tiled_mma(mma_atom, thr_layout_mnk)
```

### Key Points

- Optimized for FP16/BF16 precision
- Uses tensor cores for high throughput
- Typically accumulates in FP32

## nvgpu.MmaUniversalOp

Universal MMA operation.

### Purpose

General-purpose MMA operation that works across different precisions and architectures.

### Python API Usage

```python
# Create universal MMA operation
mma_op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)

# Create MMA atom
mma_atom = cute.make_mma_atom(mma_op)

# Create tiled MMA
tiled_mma = cute.make_tiled_mma(mma_atom, thr_layout_mnk)
```

## Usage Patterns

### Pattern 1: Async Global to Shared Copy

```python
# Create async copy atom
async_atom = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(),
    dtype
)

# Create tiled copy
tiled_copy = cute.make_tiled_copy_tv(async_atom, thr_layout, val_layout)

# Async copy
thr_copy = tiled_copy.get_slice(tidx)
cute.copy(tiled_copy, thr_copy.partition_S(gA), thr_copy.partition_D(sA))
cute.arch.cp_async_commit_group()
```

### Pattern 2: Load Matrix for Tensor Cores

```python
# Create load matrix atom
ldmatrix_atom = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
    dtype
)

# Create MMA-aligned copy
tiled_copy_A = cute.make_tiled_copy_A(ldmatrix_atom, tiled_mma)
thr_copy_A = tiled_copy_A.get_slice(tidx)

# Load from shared memory
tCsA = thr_copy_A.partition_S(sA)
tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
tCrA_copy_view = thr_copy_A.retile(tCrA)

cute.copy(tiled_copy_A, tCsA, tCrA_copy_view)
```

### Pattern 3: Tensor Core GEMM Setup

```python
# Create MMA operation
mma_op = cute.nvgpu.warp.MmaF16BF16Op()

# Create tiled MMA
tiled_mma = cute.make_tiled_mma(
    cute.make_mma_atom(mma_op),
    thr_layout_mnk
)

# Use in GEMM
cute.gemm(tiled_mma, tCrA, tCrB, acc)
```

## Architecture-Specific Notes

### Ampere (SM 8.x)

- Supports `LdMatrix8x8x16bOp` with num_matrices up to 4
- `cp.async` available for async global→shared copies
- Tensor cores support FP16/BF16/FP32/TF32

### Hopper (SM 9.x)

- Enhanced tensor core capabilities
- Improved async copy operations
- Additional precision support

### Blackwell (SM 10.x)

- New tensor core instructions
- Enhanced async operations
- CTA-wide MMA operations available

