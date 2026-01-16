# Tensor APIs

This document explains the Python APIs for CuTe DSL Tensor operations. Source code explanations refer to `include/cute/tensor_impl.hpp`.

## Overview

Tensors in CuTe are multi-dimensional arrays with associated layouts that define how logical coordinates map to memory addresses. Tensor APIs provide operations for creating, manipulating, and accessing tensors.

## make_tensor

Creates a tensor from a pointer/iterator and a layout.

### Purpose

Constructs a tensor view over existing memory with a specified layout. The tensor doesn't own the memory; it provides a view with a specific layout.

### C++ Source

```cpp
template <class Engine, class Layout>
struct Tensor {
  Engine engine_;
  Layout layout_;
  
  CUTE_HOST_DEVICE constexpr
  Tensor(Engine const& engine, Layout const& layout)
    : engine_(engine), layout_(layout) {}
};
```

### Python API Usage

```python
# Create tensor from pointer with layout
ptr = cute.runtime.make_ptr(data_ptr, dtype)
layout = cute.make_layout((M, N), stride=(N, 1))
tensor = cute.make_tensor(ptr, layout)

# Create tensor from existing tensor's iterator with new layout
new_tensor = cute.make_tensor(old_tensor.iterator, new_layout)
```

### Key Points

- The tensor is a view; it doesn't copy data
- The layout defines how coordinates map to memory addresses
- The engine (pointer/iterator) provides access to the underlying memory

## make_rmem_tensor

Creates a register memory tensor with a specified shape and dtype.

### Purpose

Allocates a tensor in register memory (thread-local storage) with a given shape and data type. This is used for thread-local accumulators, predicates, and temporary values.

### C++ Source

Register memory tensors are allocated on the stack and provide fast thread-local storage.

### Python API Usage

```python
# Create accumulator in register memory
acc_shape = (MMA, MMA_M, MMA_N)
acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
acc.fill(0.0)

# Create predicate tensor
pred_shape = (rest_v, CPY_M, CPY_K) # CPY_M, CPY_K are tiling degrees of the copy atom shape along M and K dimensions
pred = cute.make_rmem_tensor(pred_shape, cutlass.Boolean)

# Create tensor with explicit layout
pred_layout = cute.make_layout(
    (rest_v, CPY_M, CPY_K),
    stride=(CPY_K, 0, 1)
)
pred = cute.make_rmem_tensor(pred_layout, cutlass.Boolean)
```

### Example: Accumulator Allocation with MMA

```python
# Compute accumulator shape from MMA tiler
acc_shape = thr_mma.partition_shape_C((M, N))

# Allocate in register memory with explicit dtype
acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
acc.fill(0.0)

# Use in GEMM mainloop
for k_tile in range(k_tile_count):
    cute.gemm(tiled_mma, tCrA, tCrB, acc)
```

### When to Use

- **Accumulators**: Store intermediate results in GEMM operations
- **Predicates**: Store boolean masks for predicated operations
- **Temporary values**: Thread-local scratch space
- **Explicit dtype control**: When you need a specific dtype (e.g., Float32 for accumulation even if inputs are Float16)

## make_fragment_like

Creates a tensor with the same shape and layout as another tensor, but with a new allocation.

### Purpose

Allocates a new tensor (typically in register memory) that matches the shape and layout of an existing tensor. This is commonly used to create register fragments for copy operations.

### C++ Source

```cpp
template <class Tensor>
CUTE_HOST_DEVICE constexpr
auto
make_fragment_like(Tensor const& tensor) {
  return make_tensor<ValueType>(shape(tensor));
}
```

### Python API Usage

```python
# Partition source tensor
thrA = thr_copy.partition_S(blkA)

# Create register fragment matching the partitioned tensor
frgA = cute.make_fragment_like(thrA)

# Copy from global memory to register
cute.copy(copy_atom, thrA, frgA)
```

### Key Points

- Creates a new tensor with matching shape and layout
- Typically allocates in register memory
- Used to create destination tensors for copy operations

## make_identity_tensor

Creates an identity tensor where each element contains its own logical coordinate.

### Purpose

Creates a tensor where `tensor[i, j, ...] = (i, j, ...)`. This is useful for predication and coordinate-based operations.

### C++ Source

Identity tensors use identity layouts where each coordinate maps to itself.

### Python API Usage

```python
# Create identity tensor for predication
mcA = cute.make_identity_tensor(mA.shape)
cA = cute.local_tile(mcA, tiler, coord)

# Partition identity tensor to get coordinates
tAcA = thr_copy.partition_S(cA)

# Use coordinates for predication
for rest_v in range(tApA.shape[0]):
    for m in range(tApA.shape[1]):
        coord = tAcA[(0, rest_v), m, 0, 0][0]
        tApA[rest_v, m, 0] = cute.elem_less(coord, mA.shape[0])
```

### When to Use

- **Predication**: Determine which elements are in-bounds
- **Coordinate mapping**: Track logical coordinates through transformations
- **Boundary checking**: Identify elements that need special handling

## local_tile

Extracts a local tile from a tensor based on a tiler and coordinate.

### Purpose

Slices a tensor to get a specific tile based on block coordinates. This is the primary way to get per-block views of global tensors. Internally uses `zipped_divide` (a layout operation) to divide the tensor, then selects the tile at the given coordinate.

### C++ Source

```cpp
template <class Tensor, class Tiler, class Coord>
CUTE_HOST_DEVICE constexpr
auto
local_tile(Tensor const& tensor, Tiler const& tiler, Coord const& coord) {
  // Divides tensor by tiler and selects the tile at coord
  return zipped_divide(tensor, tiler)(coord, _);
}
```

### Python API Usage

```python
# Get tile for this block
bidx, bidy, bidz = cute.arch.block_idx()
tiler_coord = (bidx, bidy, None)

# Extract local tiles
gA = cute.local_tile(mA, tiler=(BLK_M, BLK_K), coord=(bidx, None, None))
gB = cute.local_tile(mB, tiler=(BLK_N, BLK_K), coord=(None, bidy, None))
gC = cute.local_tile(mC, tiler=(BLK_M, BLK_N), coord=(bidx, bidy, None))

# With projection (select specific modes)
gA = cute.local_tile(mA, tiler=cta_tiler, coord=tiler_coord, proj=(1, None, 1))
gB = cute.local_tile(mB, tiler=cta_tiler, coord=tiler_coord, proj=(None, 1, 1))
```

### Example: Complete Tiling and Partitioning Flow

```python
# Option 1: Using zipped_divide (layout operation) then indexing
gA = cute.zipped_divide(mA, tiler_mn)  # Creates ((TileM, TileN), (RestM, RestN))
bidx, _, _ = cute.arch.block_idx()
blkA = gA[((None, None), bidx)]  # Extract tile at bidx

# Option 2: Using local_tile (convenience wrapper)
bidx, _, _ = cute.arch.block_idx()
blkA = cute.local_tile(mA, tiler=(TileM, TileN), coord=(bidx, None))

# 3. Partition for this thread
tidx, _, _ = cute.arch.thread_idx()
thr_copy = tiled_copy.get_slice(tidx)
thrA = thr_copy.partition_S(blkA)  # Get this thread's portion
```

### Key Points

- `local_tile` is a convenience wrapper that uses `zipped_divide` (see Layout APIs)
- `zipped_divide` operates on layouts; the tensor version applies it to the tensor's layout
- Both approaches achieve the same result

- `tiler`: Defines the tile shape
- `coord`: Specifies which tile to extract (use `None` for modes not tiled)
- `proj`: Optional projection to select specific modes from the tiler

## domain_offset

Shifts a tensor's domain by an offset.

### Purpose

Moves the logical coordinate space of a tensor without changing the underlying memory. Useful for handling irregular tiles or boundary conditions.

### C++ Source

```cpp
template <class Tensor, class Offset>
CUTE_HOST_DEVICE constexpr
auto
domain_offset(Offset const& offset, Tensor const& tensor) {
  // Shifts the coordinate domain
}
```

### Python API Usage

```python
# Handle irregular K dimension
residue_k = mA.shape[1] - self._bK * gA.shape[2]
gA = cute.domain_offset((0, residue_k, 0), gA)
gB = cute.domain_offset((0, residue_k, 0), gB)

# This makes the first tile (instead of last) irregular
# when k is not a multiple of tile size
```

### When to Use

- **Irregular tiles**: Handle cases where problem size isn't a multiple of tile size
- **Boundary handling**: Shift coordinate space to simplify boundary checks
- **Residue handling**: Process remainder tiles separately

## composition

Composes a tensor (or a layout) with a layout to create a new logical view.

### Purpose

Applies a layout transformation to a tensor, creating a new view with a different layout but the same underlying data.

### Python API Usage

```python
# Transpose view of V tensor
sVt = cute.composition(
    sV,
    cute.make_layout(
        (head_dim, n_block_size),
        stride=(n_block_size, 1)
    )
)
```

### Key Points

- Creates a view with a new layout
- Doesn't copy data
- Useful for transpose operations and layout transformations

## select

Selects specific modes from a tensor or shape.

### Purpose

Extracts or reorders dimensions from a tensor, shape, or layout.

### Python API Usage

```python
# Select specific modes from shape
mA = buffer_a.to_tensor(cute.select(mnkl, mode=[3, 0, 2]))  # Select L, M, K

# Select modes from layout
new_layout = cute.select(layout, mode=[1, 2, 0])  # Reorder dimensions
```

### Key Points

- `mode` parameter specifies which dimensions to select and in what order
- Can be used on shapes, layouts, or tensors
- Useful for reshaping and reordering operations

## size and shape

Get the size or shape of a tensor or layout.

### Purpose

`size` returns the size of a specific mode or the total size. `shape` returns the shape tuple.

### Python API Usage

```python
# Get shape
tensor_shape = tensor.shape
layout_shape = layout.shape

# Get size of specific mode
m_size = cute.size(tensor, mode=[0])  # Size of first mode
k_size = cute.size(tensor, mode=[2])  # Size of third mode

# Get nested size
atom_v_size = cute.size(tAsA, mode=[0, 0])  # Size of nested mode
```

### Key Points

- `shape`: Returns the full shape tuple
- `size`: Returns size of specific mode(s) or total size
- Supports nested mode access with multiple indices

## fill and clear

Initialize tensor values.

### Purpose

`fill` sets all elements to a value. `clear` sets all elements to zero.

### Python API Usage

```python
# Fill accumulator with zeros
acc.fill(0.0)

# Clear accumulator
cute.clear(acc)

# Fill with specific value
tensor.fill(1.0)
```

## store and load

Store/load tensor values (typically for register memory).

### Purpose

Explicit store/load operations for register memory tensors.

### Python API Usage

```python
# Store result to global memory
thrC.store(result)

# Load from global memory
thrC.load(source)
```

## SharedStorage and get_tensor

SharedStorage provides a structured way to allocate and access shared memory buffers.

### Purpose

`SharedStorage` is a struct (decorated with `@cute.struct`) that defines the layout of shared memory buffers. It allows you to:
- Define multiple named buffers in shared memory
- Specify alignment requirements
- Create tensor views from memory ranges using `get_tensor`


### Python API Usage

#### Defining SharedStorage

```python
@cute.struct
class SharedStorage:
    # Memory range with automatic size calculation
    sQ: cute.struct.Align[
        cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
    ]
    sK: cute.struct.Align[
        cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
    ]
    sV: cute.struct.Align[
        cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
    ]
```

#### Allocating SharedStorage

```python
# Create shared memory allocator
smem = cutlass.utils.SmemAllocator()

# Allocate the SharedStorage struct
storage = smem.allocate(SharedStorage)
```

#### Creating Tensors from Storage Fields

```python
# Get tensor view from storage field with specified layout, by composing the layout with the underlying data pointer.
sQ = storage.sQ.get_tensor(sQ_layout)
sK = storage.sK.get_tensor(sKV_layout)
sV = storage.sV.get_tensor(sKV_layout)
```

### Key Components

1. **`@cute.struct`**: Decorator that marks a class as a CuTe struct
2. **`cute.struct.MemRange[dtype, size]`**: Defines a memory range
   - `dtype`: Element type
   - `size`: Number of elements (can use `cute.cosize(layout)` to compute from layout)
3. **`cute.struct.Align[MemRange[...], alignment]`**: Aligns a memory range
   - `alignment`: Byte alignment (e.g., 1024 for optimal memory access)
4. **`cute.cosize(layout)`**: Computes the codomain size (in elements) needed for a layout
   - Returns the minimum number of elements needed to store all possible offsets generated by the layout
   - Calculated as: maximum offset + 1, where maximum offset is computed from shape and stride
   - Example: For layout `(4,(3,2)):(2,(8,1))`, cosize = 24
5. **`get_tensor(layout)`**: Creates a tensor view from a `MemRange` field
   - Takes a layout that matches the size specified in `MemRange`
   - Returns a tensor view over the memory range

### When to Use

- **Multiple buffers**: When you need several named shared memory buffers
- **Alignment control**: When you need specific alignment for performance
- **Type safety**: When you want compile-time checking of buffer sizes
- **Complex layouts**: When buffer sizes depend on layouts computed at compile time

### Example: Flash Attention

```python
# Define storage structure
@cute.struct
class SharedStorage:
    sQ: cute.struct.Align[
        cute.struct.MemRange[dtype, cute.cosize(sQ_layout)], 1024
    ]
    sK: cute.struct.Align[
        cute.struct.MemRange[dtype, cute.cosize(sKV_layout)], 1024
    ]
    sV: cute.struct.Align[
        cute.struct.MemRange[dtype, cute.cosize(sKV_layout)], 1024
    ]

# In kernel
smem = cutlass.utils.SmemAllocator()
storage = smem.allocate(SharedStorage)

# Create tensor views
sQ = storage.sQ.get_tensor(sQ_layout)
sK = storage.sK.get_tensor(sKV_layout)
sV = storage.sV.get_tensor(sKV_layout)
```

### Key Points

- `cute.cosize(layout)` computes the codomain size (number of elements) needed for a layout
  - This is the minimum buffer size needed to hold all data accessed through the layout
  - Accounts for stride patterns, not just shape
- Alignment (e.g., 1024) ensures optimal memory access patterns and may be required for certain operations
- `get_tensor` creates a view; it doesn't copy data
- The layout passed to `get_tensor` should have `cosize` matching the size used in `MemRange`
- `SmemAllocator` manages shared memory allocation and ensures proper alignment

