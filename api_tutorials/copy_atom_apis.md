# Copy Atom APIs

This document explains the Python APIs for CuTe DSL Copy Atom operations. Source code explanations refer to `include/cute/atom/copy_atom.hpp`.

## Overview

Copy Atoms are the fundamental building blocks for data movement operations in CuTe. They represent hardware copy instructions (like `cp.async`, `ldmatrix`, etc.) and are tiled across threads to enable efficient parallel data transfers.

## make_copy_atom

Creates a `CopyAtom` from a copy operation and element type.

### Purpose

Wraps a copy operation (e.g., `CopyUniversalOp`, `CopyG2SOp`, `LdMatrix8x8x16bOp`) with an element type to create a reusable copy atom.

### C++ Source

```cpp
template <class CopyOperation, class CopyInternalType>
struct Copy_Atom<CopyOperation, CopyInternalType> : Copy_Atom<Copy_Traits<CopyOperation>, CopyInternalType>
{};
```

### Python API Usage

```python
# Universal copy atom for general memory operations
copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)

# Async copy from global to shared memory
async_copy_atom = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(),
    cutlass.Float16
)

# Load matrix operation for tensor cores
ldmatrix_atom = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
    cutlass.Float16
)
```

### Common Copy Operations

- **`cute.nvgpu.CopyUniversalOp()`**: General-purpose copy operation
- **`cute.nvgpu.cpasync.CopyG2SOp()`**: Asynchronous copy from global to shared memory
- **`cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose, num_matrices)`**: Load matrix operation for tensor cores
  - `transpose`: Whether to transpose during load
  - `num_matrices`: Number of matrices to load (typically 1, 2, or 4)

## make_tiled_copy_tv

Creates a `TiledCopy` from a copy atom and thread-value (TV) layout.

### Purpose

Tiles a copy atom across multiple threads using a specified thread-value layout. The TV layout defines how threads and values are mapped to logical coordinates.

### C++ Source

The tiled copy is created by combining the copy atom with thread and value layouts to create a hierarchical structure that distributes the copy operation across threads.

### Python API Usage

```python
# Create thread and value layouts
thr_layout = cute.make_layout((4, 32), stride=(32, 1))  # Thread layout
val_layout = cute.make_layout((4, 4), stride=(4, 1))    # Value layout

# Create tiled copy with TV layout
tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
```

### Key Points

- The thread layout defines how threads are distributed across the tile
- The value layout defines how values are distributed per thread
- Together they enable vectorized, coalesced memory accesses

## make_tiled_copy_A/B

Creates a `TiledCopy` that matches the thread-value layout expected by a `TiledMMA` for operand A or B.

### Purpose

Automatically creates a tiled copy with a TV layout that aligns with the MMA's expected layout for efficient data movement from shared memory to registers.

### C++ Source

These functions extract the appropriate layout from the `TiledMMA` and create a matching `TiledCopy`.

### Python API Usage

```python
# Create copy atom for shared memory to register
smem_copy_atom = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
    dtype
)

# Create tiled copy aligned with MMA operand A
tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma)

# Create tiled copy aligned with MMA operand B
tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
```

### When to Use

- Use `make_tiled_copy_A` when copying data that will be used as operand A in MMA
- Use `make_tiled_copy_B` when copying data that will be used as operand B in MMA
- This ensures optimal layout alignment for MMA operations

## TiledCopy.get_slice

Gets a thread-specific view of the tiled copy.

### Purpose

Returns a `ThrCopy` object that represents a single thread's portion of the tiled copy operation.

### C++ Source

```cpp
template <class ThrIdx>
CUTE_HOST_DEVICE static
auto
get_slice(ThrIdx const& thr_idx)
{
  return ThrCopy<TiledCopy, ThrIdx>(thr_idx);
}
```

### Python API Usage

```python
# Get thread index
tidx, _, _ = cute.arch.thread_idx()

# Get thread's view of the tiled copy
thr_copy = tiled_copy.get_slice(tidx)
```

## partition_S and partition_D

Partition source and destination tensors for a thread's copy operation.

### Purpose

`partition_S` extracts the portion of the source tensor that this thread should copy. `partition_D` extracts the portion of the destination tensor where this thread should write.

### C++ Source

```cpp
template <class STensor>
CUTE_HOST_DEVICE
auto
partition_S(STensor&& stensor) const {
  auto thr_tensor = make_tensor(static_cast<STensor&&>(stensor).data(), 
                                TiledCopy::tidfrg_S(stensor.layout()));
  return thr_tensor(thr_idx_, _, repeat<rank_v<STensor>>(_));
}

template <class DTensor>
CUTE_HOST_DEVICE
auto
partition_D(DTensor&& dtensor) const {
  auto thr_tensor = make_tensor(static_cast<DTensor&&>(dtensor).data(), 
                                TiledCopy::tidfrg_D(dtensor.layout()));
  return thr_tensor(thr_idx_, _, repeat<rank_v<DTensor>>(_));
}
```

### Python API Usage

```python
# Partition source tensor (e.g., from global memory)
tSgA = thr_copy.partition_S(gA)  # Get this thread's slice of source

# Partition destination tensor (e.g., to shared memory)
tDsA = thr_copy.partition_D(sA)  # Get this thread's slice of destination

# Use in copy operation
cute.copy(copy_atom, tSgA, tDsA)
```

### Key Points

- `partition_S` is used for the source (where data comes from)
- `partition_D` is used for the destination (where data goes to)
- Both return views that represent only this thread's portion

## retile

Retiles a tensor to match the layout expected by a copy operation.

### Purpose

Transforms a tensor's layout to match the copy operation's expected layout without changing the underlying data. This is useful when you have a tensor in one layout (e.g., from MMA fragment) but need to copy it using a different layout.

### C++ Source

```cpp
template <class STensor>
CUTE_HOST_DEVICE static
auto
retile_S(STensor&& stensor) {
  return make_tensor(static_cast<STensor&&>(stensor).data(), 
                     TiledCopy::retile(stensor.layout()));
}
```

### Python API Usage

```python
# Get fragment from MMA (has MMA layout)
tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))

# Get copy view for shared memory (has copy layout)
tCsA = thr_copy.partition_S(sA)

# Retile the fragment to match copy layout
tCrA_copy_view = thr_copy.retile(tCrA)

# Now can copy using the retiled view
cute.copy(tiled_copy, tCsA, tCrA_copy_view)
```

### When to Use

- When you need to copy data between layouts that don't match
- Common pattern: MMA fragment → retile → copy from shared memory
- The retile operation creates a view, not a copy of the data

## Usage Pattern

A typical pattern for using copy atoms:

```python
# 1. Create copy atom
copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)

# 2. Create tiled copy
thr_layout = cute.make_layout((4, 32), stride=(32, 1))
val_layout = cute.make_layout((4, 4), stride=(4, 1))
tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

# 3. Get thread's view
tidx, _, _ = cute.arch.thread_idx()
thr_copy = tiled_copy.get_slice(tidx)

# 4. Partition tensors
tSgA = thr_copy.partition_S(gA)  # Source from global memory
tDsA = thr_copy.partition_D(sA)  # Destination in shared memory

# 5. Perform copy
cute.copy(copy_atom, tSgA, tDsA)
```

## Integration with MMA

Copy atoms are often used to feed data into MMA operations:

```python
# Create MMA-aligned copy for operand A
tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma)
thr_copy_A = tiled_copy_A.get_slice(tidx)

# Partition shared memory tensor
tCsA = thr_copy_A.partition_S(sA)

# Get MMA fragment
tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))

# Retile fragment to match copy layout
tCrA_copy_view = thr_copy_A.retile(tCrA)

# Copy from shared memory to register
cute.copy(tiled_copy_A, tCsA, tCrA_copy_view)
```

This pattern ensures optimal data movement from shared memory to registers in the layout expected by the MMA operation.

