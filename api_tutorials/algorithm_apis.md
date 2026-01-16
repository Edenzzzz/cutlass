# Algorithm APIs

This document explains the Python APIs for CuTe DSL Algorithm operations (copy and gemm). Source code explanations refer to `include/cute/algorithm/`.

## Overview

Algorithm APIs provide high-level operations for data movement (copy) and matrix multiplication (gemm). These operations work with partitioned tensors and handle the coordination of copy atoms and MMA operations.

## copy

Performs a copy operation between source and destination tensors using a copy atom or tiled copy.

### Purpose

Executes data movement between tensors, potentially using hardware-accelerated copy operations. Can use either a copy atom directly or a tiled copy for thread-cooperative operations.

### C++ Source

The copy function dispatches to the appropriate copy implementation based on the copy atom/tiled copy type and tensor layouts.

### Python API Usage

```python
# Copy using copy atom
cute.copy(copy_atom, src_tensor, dst_tensor)

# Copy using tiled copy (thread-cooperative)
cute.copy(tiled_copy, src_tensor, dst_tensor)

# Copy with predicate
cute.copy(copy_atom, src_tensor, dst_tensor, pred=predicate_tensor)

# Copy from global to shared memory (async)
cute.copy(tiled_copy, tSgA, tDsA)
cute.arch.cp_async_commit_group()

# Copy from shared to register memory
cute.copy(tiled_copy, tCsA, tCrA)
```

### Key Points

- Works with both `CopyAtom` and `TiledCopy`
- Supports predication for boundary handling
- Can be synchronous or asynchronous (with `cp_async_commit_group`)
- Automatically handles vectorization and coalescing when layouts align

## gemm

Performs matrix multiplication using tiled MMA operations.

### Purpose

Executes GEMM (General Matrix Multiply) operations: C = A × B + C, using tiled MMA operations across threads.

### C++ Source

The gemm function coordinates multiple MMA operations across threads to compute the matrix product.

### Python API Usage

```python
# Basic GEMM
cute.gemm(tiled_mma, tCrA, tCrB, tCrC)

# GEMM with different accumulator
cute.gemm(tiled_mma, tCrA, tCrB, tCrC, tCrD)

# In a pipeline loop
for k_tile in range(k_tile_count):
    # Load A, B
    cute.copy(tiled_copy_A, tCsA, tCrA)
    cute.copy(tiled_copy_B, tCsB, tCrB)
    
    # Compute
    cute.gemm(tiled_mma, tCrA, tCrB, tCrC)
```

### Key Points

- Operates on partitioned tensors (thread views)
- Accumulates into C tensor (C = A × B + C)
- Coordinates MMA operations across all threads in the tiled MMA
- Typically used in a loop over K dimension

## clear

Sets all elements of a tensor to zero.

### Purpose

Initializes a tensor (typically an accumulator) to zero.

### Python API Usage

```python
# Clear accumulator
cute.clear(acc)

# Equivalent to
acc.fill(0.0)
```

## fill

Sets all elements of a tensor to a specific value.

### Purpose

Initializes a tensor to a constant value.

### Python API Usage

```python
# Fill with zero
tensor.fill(0.0)

# Fill with specific value
tensor.fill(1.0)
```

## Usage Patterns

### Pattern 1: Global to Shared Memory Copy

```python
# Partition tensors
tSgA = thr_copy.partition_S(gA)  # Source from global
tDsA = thr_copy.partition_D(sA)  # Destination in shared

# Async copy
cute.copy(tiled_copy, tSgA, tDsA)
cute.arch.cp_async_commit_group()

# Wait for completion
cute.arch.cp_async_wait_group(0)
cute.arch.sync_threads()
```

### Pattern 2: Shared to Register Memory Copy

```python
# Partition tensors
tCsA = thr_copy.partition_S(sA)  # Source from shared
tCrA = thr_copy.retile(frgA)     # Destination in register

# Copy
cute.copy(tiled_copy, tCsA, tCrA)
```

### Pattern 3: GEMM Mainloop

```python
# Initialize accumulator
cute.clear(acc)

# Mainloop over K dimension
for k_tile in range(k_tile_count):
    # Load next tile
    cute.copy(tiled_copy_A, tCsA[k_tile], tCrA)
    cute.copy(tiled_copy_B, tCsB[k_tile], tCrB)
    
    # Compute
    cute.gemm(tiled_mma, tCrA, tCrB, acc)
```

### Pattern 4: Predicated Copy

```python
# Create predicate
pred = cute.make_rmem_tensor(pred_layout, cutlass.Boolean)
# ... set predicate values ...

# Copy with predicate
cute.copy(copy_atom, src, dst, pred=pred)
```

