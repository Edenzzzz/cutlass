# Layout APIs

This document explains the Python APIs for CuTe DSL Layout operations. Source code explanations refer to `include/cute/layout.hpp`.

## Overview

Layouts in CuTe define how logical coordinates map to indices (memory addresses). They consist of a shape (dimensions) and stride (how to traverse dimensions). Layout APIs provide functions for creating and manipulating layouts.

## make_layout

Creates a layout from shape and optional stride.

### Purpose

Constructs a layout that maps logical coordinates to linear indices. If stride is not provided, defaults to a row-major layout.

### C++ Source

```cpp
template <class Shape, class Stride = LayoutLeft::Apply<Shape>>
struct Layout {
  Shape shape_;
  Stride stride_;
  
  CUTE_HOST_DEVICE constexpr
  Layout(Shape const& shape = {}, Stride const& stride = {})
    : shape_(shape), stride_(stride) {}
};
```

### Python API Usage

```python
# Row-major layout (default)
layout = cute.make_layout((M, N))  # Stride: (N, 1)

# Explicit stride
layout = cute.make_layout((M, N), stride=(N, 1))

# Column-major layout
layout = cute.make_layout((M, N), stride=(1, M))

# Complex nested layout
layout = cute.make_layout(
    ((32, 4), (4, 4)),
    stride=((4, 512), (1, 128))
)
```

### Key Points

- Default stride is row-major (rightmost dimension has stride 1)
- Stride defines how to compute linear index from coordinates
- Supports nested shapes and strides for hierarchical layouts

## make_ordered_layout

Creates a layout with strides ordered by a specified dimension order.

### Purpose

Automatically generates strides based on a dimension ordering, useful for creating layouts that match specific memory access patterns.

### Python API Usage

```python
# Order dimensions: (1, 0) means dimension 1 is fastest
layout = cute.make_ordered_layout((4, 64), order=(1, 0))
# Results in stride: (64, 1) - column-major

# Order dimensions: (0, 1) means dimension 0 is fastest  
layout = cute.make_ordered_layout((4, 64), order=(0, 1))
# Results in stride: (1, 4) - row-major
```

### When to Use

- When you know the desired dimension ordering but not exact strides
- For creating layouts that match input tensor memory layouts
- Simplifies layout creation for common patterns

## make_layout_tv

Creates a tiler and TV layout from thread and value layouts.

### Purpose

Creates a layout that maps thread and value coordinates to an index in a memory space (e.g. smem, gmem). Returns both the tiler (tile shape) and the TV layout.

### Python API Usage

```python
# Define thread and value layouts
thr_layout = cute.make_layout((4, 32), stride=(32, 1))
val_layout = cute.make_layout((4, 4), stride=(4, 1))

# Create tiler and TV layout
# (thread_idx, value_idx) -> (M,N)
tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

# Use with tiled copy
tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
```

### Key Points

- Combines thread and value layouts into a single TV layout
- Returns the tiler shape that matches the TV layout
- Essential for creating efficient tiled copy operations

## Layout Properties

### shape

Get the shape of a layout.

```python
layout_shape = layout.shape  # Returns shape tuple
```

### stride

Get the stride of a layout.

```python
layout_stride = layout.stride  # Returns stride tuple
```

## logical_divide

Divides a layout by a tiler, creating a hierarchical structure.

### Purpose

Splits a layout into tiles and residuals. For each mode, creates a pair `(Tile, Rest)` where `Tile` is the tiled portion and `Rest` is the remainder. Extra dimensions are preserved in their original positions.

### Shape Transformation

**Input shape**: `(M, N, L, ...)`  
**Tiler shape**: `(TileM, TileN)` (applies to first 2 modes, use `None` to skip modes)  
**Output shape**: `((TileM, RestM), (TileN, RestN), L, ...)`



### C++ Source

```cpp
template <class Layout, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
logical_divide(Layout const& layout, Tiler const& tiler) {
  // Divides each mode: ((TileM, RestM), (TileN, RestN), ...)
  return composition(layout, make_layout(tiler, complement(tiler, shape(layout))));
}
```

### Python API Usage

```python
# Example 1: Basic 2D division
# Input shape: (M, N)
# Tiler shape: (TileM, TileN)
# Output shape: ((TileM, RestM), (TileN, RestN))
layout = cute.make_layout((M, N), stride=(N, 1))
tiler = (TileM, TileN)
divided = cute.logical_divide(layout, tiler)

# Example 2: With extra dimensions
# Input shape: (M, N, L, K)
# Tiler shape: (TileM, TileN)
# Output shape: ((TileM, RestM), (TileN, RestN), L, K)
layout = cute.make_layout((M, N, L, K), stride=(N*L*K, L*K, K, 1))
tiler = (TileM, TileN)
divided = cute.logical_divide(layout, tiler)
# Extra dimensions L and K are preserved

# Example 3: Divide specific mode with None for others
# Input shape: (4, MMA_M, MMA_N)
# Tiler shape: (None, None, 2)
# Output shape: (4, MMA_M, (2, RestN))
layout = cute.make_layout((4, MMA_M, MMA_N))
divided = cute.logical_divide(layout, (None, None, 2))
# Only the last mode is divided, first two modes unchanged

# Use with tensor layout
rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
# Then create new layout from divided result
rP_mma_view = cute.make_layout(
    (
        (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
        rP_layout_divided.shape[1],
        rP_layout_divided.shape[2][1],
    ),
    stride=(
        (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
        rP_layout_divided.stride[1],
        rP_layout_divided.stride[2][1],
    ),
)
```

### Key Points

- Divides each mode independently: `(Tile, Rest)` pairs
- Use `None` in tiler to skip modes
- Result preserves the hierarchical structure of tiles and residuals
- Useful for reshaping layouts to match operation requirements (e.g., MMA instruction shapes)

## zipped_divide

Divides a layout by a tiler, gathering tiles and residuals into separate modes.

### Purpose

Similar to `logical_divide`, but gathers all tile modes together and all residual modes together. Extra dimensions are gathered into the residual mode. This is more convenient for accessing tiles by index.

### Shape Transformation

**Input shape**: `(M, N, L, ...)`  
**Tiler shape**: `(TileM, TileN)` (applies to first 2 modes)  
**Output shape**: `((TileM, TileN), (RestM, RestN, L, ...))`

Where:
- First mode `(TileM, TileN)` contains all tile dimensions
- Second mode `(RestM, RestN, L, ...)` contains all residual dimensions plus extra dimensions
- `RestM = ceil(M / TileM)` - number of tiles in M dimension
- `RestN = ceil(N / TileN)` - number of tiles in N dimension

### C++ Source

```cpp
template <class Layout, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
zipped_divide(Layout const& layout, Tiler const& tiler) {
  // Gathers tiles and residuals: ((TileM, TileN), (RestM, RestN, ...))
  return tile_unzip(logical_divide(layout, tiler), tiler);
}
```

### Python API Usage

```python
# Example 1: Basic 2D division
# Input shape: (M, N)
# Tiler shape: (TileM, TileN)
# Output shape: ((TileM, TileN), (RestM, RestN))
layout = cute.make_layout((M, N), stride=(N, 1))
tiler = (TileM, TileN)
zipped = cute.zipped_divide(layout, tiler)

# Example 2: With extra dimensions
# Input shape: (M, N, L, K)
# Tiler shape: (TileM, TileN)
# Output shape: ((TileM, TileN), (RestM, RestN, L, K))
layout = cute.make_layout((M, N, L, K), stride=(N*L*K, L*K, K, 1))
tiler = (TileM, TileN)
zipped = cute.zipped_divide(layout, tiler)
# Extra dimensions L and K are gathered into the second mode

# Example 3: Works with tensors (applies to tensor's layout)
# Input tensor shape: (M, N, Tiles_K)
# Tiler shape: (TileM, TileN)
# Output tensor shape: ((TileM, TileN), (RestM, RestN, Tiles_K))
gA = cute.zipped_divide(mA, tiler_mn)

# Access specific tile
bidx, _, _ = cute.arch.block_idx()
blkA = gA[((None, None), bidx)]  # Get tile at bidx
# blkA has shape: (TileM, TileN, RestM, RestN, Tiles_K) after indexing

# Example 4: Divide identity tensor for predication
# Input shape: (M, N, L)
# Tiler shape: (TileM, TileN)
# Output shape: ((TileM, TileN), (RestM, RestN, L))
idC = cute.make_identity_tensor(mC.shape)
cC = cute.zipped_divide(idC, tiler=tiler_mn)
```

### Key Points

- Gathers tiles into first mode, residuals into second mode
- More convenient than `logical_divide` for accessing tiles by index
- Works on both layouts and tensors (tensor version applies to layout)
- Common pattern: `zipped_divide` → `local_tile` → partition

## tiled_divide

Divides a layout by a tiler, unpacking the residual mode.

### Purpose

Similar to `zipped_divide`, but unpacks the second mode (residuals) so that residual dimensions are separate rather than grouped. Extra dimensions remain unpacked in their original positions.

### Shape Transformation

**Input shape**: `(M, N, L, ...)`  
**Tiler shape**: `(TileM, TileN)` (applies to first 2 modes)  
**Output shape**: `((TileM, TileN), RestM, RestN, L, ...)`

Where:
- First mode `(TileM, TileN)` contains all tile dimensions (same as `zipped_divide`)
- Residual dimensions `RestM, RestN` are unpacked (not grouped)
- Extra dimensions `L, ...` remain unpacked in their original positions
- `RestM = ceil(M / TileM)` - number of tiles in M dimension
- `RestN = ceil(N / TileN)` - number of tiles in N dimension

### C++ Source

```cpp
template <class Layout, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
tiled_divide(Layout const& layout, Tiler const& tiler) {
  auto result = zipped_divide(layout, tiler);
  auto R1 = rank<1>(result);
  return result(_, repeat<R1>(_));  // Unpack second mode
}
```

### Python API Usage

```python
# Example 1: Basic 2D division
# Input shape: (M, N)
# Tiler shape: (TileM, TileN)
# Output shape: ((TileM, TileN), RestM, RestN)
layout = cute.make_layout((M, N), stride=(N, 1))
tiler = (TileM, TileN)
tiled = cute.tiled_divide(layout, tiler)

# Example 2: With extra dimensions
# Input shape: (M, N, L, K)
# Tiler shape: (TileM, TileN)
# Output shape: ((TileM, TileN), RestM, RestN, L, K)
layout = cute.make_layout((M, N, L, K), stride=(N*L*K, L*K, K, 1))
tiler = (TileM, TileN)
tiled = cute.tiled_divide(layout, tiler)
# Residual dimensions and extra dimensions are unpacked

# Example 3: Cluster layout for thread mapping
# Input shape: (M, N, K)
# Tiler shape: (thr_id.shape,)
# Output shape: ((thr_id.shape), RestM, RestN, RestK)
cluster_shape_mnk = (M, N, K)
cluster_layout_vmnk = cute.tiled_divide(
    cute.make_layout(cluster_shape_mnk),
    (tiled_mma.thr_id.shape,)
)
# Used to map cluster coordinates to thread coordinates
```

### Key Points

- Tiles are grouped in first mode (same as `zipped_divide`)
- Residual dimensions are unpacked (unlike `zipped_divide`)
- Extra dimensions remain unpacked
- Useful when you need to access residual dimensions individually

## flat_divide

Divides a layout by a tiler, completely flattening the result.

### Purpose

Similar to `zipped_divide`, but unpacks both tile and residual modes completely, creating a flat structure. This is the most unpacked form of division.

### Shape Transformation

**Input shape**: `(M, N, L, ...)`  
**Tiler shape**: `(TileM, TileN)` (applies to first 2 modes)  
**Output shape**: `(TileM, TileN, RestM, RestN, L, ...)`

Where:
- All dimensions are unpacked into a flat sequence
- Tile dimensions come first: `TileM, TileN`
- Residual dimensions follow: `RestM, RestN`
- Extra dimensions `L, ...` remain at the end
- `RestM = ceil(M / TileM)` - number of tiles in M dimension
- `RestN = ceil(N / TileN)` - number of tiles in N dimension

### C++ Source

```cpp
template <class Layout, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
flat_divide(Layout const& layout, Tiler const& tiler) {
  auto result = zipped_divide(layout, tiler);
  auto R0 = rank<0>(result);
  auto R1 = rank<1>(result);
  return result(repeat<R0>(_), repeat<R1>(_));  // Unpack both modes
}
```

### Python API Usage

```python
# Example 1: Basic 2D division
# Input shape: (M, N)
# Tiler shape: (TileM, TileN)
# Output shape: (TileM, TileN, RestM, RestN)
layout = cute.make_layout((M, N), stride=(N, 1))
tiler = (TileM, TileN)
flat = cute.flat_divide(layout, tiler)

# Example 2: With extra dimensions
# Input shape: (M, N, L, K)
# Tiler shape: (TileM, TileN)
# Output shape: (TileM, TileN, RestM, RestN, L, K)
layout = cute.make_layout((M, N, L, K), stride=(N*L*K, L*K, K, 1))
tiler = (TileM, TileN)
flat = cute.flat_divide(layout, tiler)
# All dimensions are unpacked into a flat sequence

# Example 3: Partitioning tensors for TMA operations
# Input tensor shape: (seqlen_q, head_dim, num_heads)
# Tiler shape: (mma_tile_m, mma_tile_k) from qk_mma_tiler
# Output shape: (mma_tile_m, mma_tile_k, RestM, RestK, num_heads)
gQ_qdl = cute.flat_divide(
    mQ_qdl,
    cute.select(self.qk_mma_tiler, mode=[0, 2])  # Select M and K modes
)
# Used for TMA partitioning where flat structure is needed

# Example 4: Partitioning K tensor
# Input tensor shape: (seqlen_k, head_dim, num_heads)
# Tiler shape: (mma_tile_n, mma_tile_k)
# Output shape: (mma_tile_n, mma_tile_k, RestN, RestK, num_heads)
gK_kdl = cute.flat_divide(
    mK_kdl,
    cute.select(self.qk_mma_tiler, mode=[1, 2])  # Select N and K modes
)
```

### Key Points

- Completely flattens the hierarchical structure
- All dimensions are unpacked into a single sequence
- Useful for operations that require flat indexing
- Common in TMA (Tensor Memory Accelerator) partitioning

### Comparison of Divide Operations

**Example with extra dimensions:**

Input shape: `(M, N, L, K)`  
Tiler shape: `(TileM, TileN)`

- **`logical_divide`**: 
  - Output shape: `((TileM, RestM), (TileN, RestN), L, K)`
  - Preserves mode structure and extra dimensions in their original positions
  - Use when you need to access tile/rest pairs per mode
  - Better for reshaping layouts to match specific operation shapes
  
- **`zipped_divide`**: 
  - Output shape: `((TileM, TileN), (RestM, RestN, L, K))`
  - Gathers tiles into first mode, residuals and extra dimensions into second mode
  - Use when you want to access tiles by a single index
  - More convenient for tiling workflows

- **`tiled_divide`**: 
  - Output shape: `((TileM, TileN), RestM, RestN, L, K)`
  - Tiles grouped, residuals unpacked, extra dimensions unpacked
  - Use when you need grouped tiles but individual residual access
  - Common for cluster/thread coordinate mapping

- **`flat_divide`**: 
  - Output shape: `(TileM, TileN, RestM, RestN, L, K)`
  - Completely flat structure, all dimensions unpacked
  - Use when you need flat indexing for operations like TMA
  - Simplest structure for sequential access

