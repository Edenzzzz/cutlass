## This tutorial explains the Python APIs for CuTe DSL MMA atom in detail. Source code explanations refer to include/cute/atom/mma_atom.hpp, as many of these APIs do not expose implementations at the Python level.

# TiledMMA
TiledMMA is a template class that tiles a base MMA_Atom across multiple threads for parallel matrix multiplication.
### Purpose
Tiles a single MMA instruction into a larger operation distributed across many threads, enabling larger GEMM tiles.
### Key Components
1. Template parameters:
  - MMA_Atom: The base MMA instruction/atom
  - AtomLayoutMNK: How to tile across M, N, K dimensions
  - PermutationMNK: Optional permutations for each MNK mode
1. Thread layout:
  - ThrLayoutVMNK: Maps threads to (V, M, N, K) coordinates
  - V: Threads within a single MMA atom
  - M, N, K: Thread tiling across the larger tile
1. Core functionality:
  - thrfrg_A/B/C: Partition tensors into thread-fragment structures
  - get_slice(thr_idx): Get a ThrMMA view for a specific thread
  - get_layoutA_TV/B_TV/C_TV: Get thread-value layouts for each operand
### How it works
  - Takes a base MMA_Atom (e.g., a single warp-level MMA)
  - Tiles it across a thread block using AtomLayoutMNK
  - Provides partitioning functions (thrfrg_A/B/C) to distribute data
  - Each thread gets a ThrMMA slice via get_slice() to access its portion
In essence, it scales a single MMA operation into a larger, multi-threaded GEMM tile.

## thrfrg_A/B/C
thrfrg_A is a member of TiledMMA that partitions an A tensor (M×K) for thread-level access in Hopper/Ampere style mma instructions (exception: Blackwell tcgen05.mma is CTA-wide, and instead uses mma-level access). thrfrg_C/thrfrg_B work similarly but on the B and C matrix in A x B = C. 

**Input**: Tensor with shape `(M, K, ...)`  
**Output**: Tensor with shape `((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))`
### Purpose
  Partitions tensor A so each thread knows which data it handles in the tiled matrix multiplication.
### Transformation Steps
  The function performs four transformations:
  1. Permutation 
```cpp
auto t_tile = make_tile(permutation_mnk<0>(), permutation_mnk<2>());
auto t_tensor = logical_divide(atensor, t_tile);  // (PermM,PermK,...)
```
  - Applies M and K permutations to align with the tiled MMA layout
  - Additional dimensions (...) are preserved through this step
  2. Atom Tiling
```cpp
auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                           make_layout(size<2>(AtomShape_MNK{})));
auto a_tensor = zipped_divide(t_tensor, a_tile);  // ((AtomM,AtomK),(RestM,RestK,...))
```
  - Tiles into atom-sized chunks (the basic MMA unit)
  - Separates atom dimensions from remaining dimensions
  - Additional dimensions (...) are preserved in RestM and RestK
  3. Thread-Value Transformation 
``` cpp
auto tv_tensor = a_tensor.compose(AtomLayoutA_TV{},_);  // ((ThrV,FrgV),(RestM,RestK,...))
```
  - Maps from (M,K) to (Thread, Value) using AtomLayoutA_TV
  - ThrV: threads within an MMA
  - FrgV: values/fragments within an MMA
  - Additional dimensions (...) are preserved along with RestM and RestK
  4. Thread Tiling
```cpp
auto thr_tile = make_tile(_,
                             make_tile(make_layout(size<1>(thr_layout_vmnk_)),
                                       make_layout(size<3>(thr_layout_vmnk_))));
auto thr_tensor = zipped_divide(tv_tensor, thr_tile);  // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))
```
  - Tiles across threads in M and K using thr_layout_vmnk_
  - Final structure: `((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))`
### Output Structure
  The result has this hierarchical shape: `((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))`
  - ThrV: Threads local to an MMA (from layout<0>(ThrLayoutVMNK))
  - ThrM: Threads tiled in M (from layout<1>(ThrLayoutVMNK)), e.g. Support ThrV is the size of a warp (32), this tiles ThrM warps to handle a ThrM times larger M dim
  - ThrK: Threads tiled in K (from layout<3>(ThrLayoutVMNK))
  - FrgV: Values local to an MMA
  - RestM: MMA tiling degree in M
  - RestK: MMA tiling degree in K
  - Additional dimensions (...): Any extra dimensions such as L.
### Usage
  Used by partition_A to extract the slice of tensor A for a specific thread. This enables each thread to load its portion of A for the MMA operation. This mirrors thrfrg_C (for C) and thrfrg_B (for B), but operates on the M-K dimensions of tensor A.
  
### thrfrg_B (Input tensor B)
    - Input shape: `(N,K,...)`
    - Permutations: permutation_mnk<1>(), permutation_mnk<2>() → `(PermN, PermK, ...)`
    - Atom tiling: Uses size<1> and size<2> of AtomShape_MNK → `(AtomN, AtomK)`
    - Layout: AtomLayoutB_TV
    - Thread tiling: Uses size<2> and size<3> of thr_layout_vmnk_ → `(ThrN, ThrK)`
    - Output: `((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK,...)))`

### thrfrg_C (Output tensor C)
    - Input shape: `(M,N,...)`
    - Permutations: permutation_mnk<0>(), permutation_mnk<1>() → `(PermM, PermN, ...)`
    - Atom tiling: Uses size<0> and size<1> of AtomShape_MNK → `(AtomM, AtomN)`
    - Layout: AtomLayoutC_TV
    - Thread tiling: Uses size<1> and size<2> of thr_layout_vmnk_ → `(ThrM, ThrN)`
    - Output: `((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN,...)))`

## ThrMMA

ThrMMA is a thread-specific view of a TiledMMA operation. It represents a single thread's portion of a tiled matrix multiplication.

### Purpose
ThrMMA provides a thread-local interface to access and operate on the portion of the tiled GEMM that belongs to a specific thread, enabling parallel execution across a thread block.

### Key Components

1. **Structure**:
   - Inherits from `TiledMMA` to access tiling information
   - Stores `thr_vmnk_`: this thread's coordinates in (V, M, N, K) space
     - `V`: Thread within an MMA atom
     - `M, N, K`: Thread's position in the tiled operation

2. **Core Functions**:
   - **`partition_A/B/C`**: Extract this thread's slice of tensors A, B, or C
     - Uses `thrfrg_A/B/C` to get the tiled layout
     - Slices using this thread's coordinates (`thr_vmk`, `thr_vnk`, or `thr_vmn`)
     - Returns a view of the data this thread should process
   
   - **`partition_fragment_A/B/C`**: Same as above, but also creates the appropriate fragment tensor (register allocation) for MMA operations

### Usage Pattern (Python API)

```python
# Create a tiled MMA operation
tiled_mma = cute.make_tiled_mma(mma_atom, thr_layout_mnk)

# Get this thread's view of the tiled MMA
thr_mma = tiled_mma.get_slice(tidx)  # tidx is the thread index

# Partition tensors for this thread
# (MMA, MMA_M, MMA_K, RestM, RestK, ...)
tCgA = thr_mma.partition_A(gA)  # Get this thread's slice of A

# (MMA, MMA_N, MMA_K, RestN, RestK, ...)
tCgB = thr_mma.partition_B(gB)  # Get this thread's slice of B

# (MMA, MMA_M, MMA_N, RestM, RestN, ...)
tCgC = thr_mma.partition_C(gC)  # Get this thread's slice of C

# Create register fragments for computation
tCrA = tiled_mma.make_fragment_A(sA)  # Register fragment for A
tCrB = tiled_mma.make_fragment_B(sB)  # Register fragment for B
tCrC = tiled_mma.make_fragment_C(acc_shape)  # Accumulator fragment
```

## partition_A/B/C

`partition_A` is a member of `ThrMMA` (a specific thread's view of the tiled MMA). It extracts the slice of tensor A that this thread (or this mma/CTA in B200) should process.

### How it works

**Step 1: `thrfrg_A` transforms the layout**
- Builds a tensor view using the same data but with the layout from `thrfrg_A`, which structures the tensor as `((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))`.

**Step 2: `partition_A` slices for a specific thread**

```cpp
partition_A(ATensor&& atensor) const
{
  // 1. Create tensor with thrfrg_A layout
  auto thr_tensor = make_tensor(atensor.data(), this->thrfrg_A(atensor.layout()));

  // 2. Create coordinate from thread's VMK position
  auto thr_vmk = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<3>(thr_vmnk_)));

  // 3. Slice: select this thread's portion, keep all Rest modes
  return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
}
```

**What it does:**
- `thr_vmk` selects the thread's (V, M, K) coordinate
- `make_coord(_, repeat<...>(_))` keeps all Rest modes (RestM, RestK, and any additional dimensions ...)

### Result

Returns a view of tensor A containing only the data this thread should process, structured as `(FrgV, RestM, RestK, ...)` where:
- `FrgV`: Fragment values for this thread
- `RestM`: Remaining M dimensions
- `RestK`: Remaining K dimensions  
- `...`: Any additional dimensions from the input tensor

### Example

- **Input `gA`**: `(128, 64, 4)` - `(MmaTile_M, MmaTile_K, Tiles_K)`
- **After `partition_A`**: `((128, 16), 1, 4, 4)` - `(MmaA, NumMma_M, NumMma_K, Tiles_K)`

**Meaning:**
- `(128, 16)`: 128 values per MMA atom, 16 values per thread (from `AtomLayoutA_TV`)
- `1`: 1 MMA tile in M dimension (RestM)
- `4`: 4 MMA tiles in K dimension (RestK)
- `4`: 4 outer K tiles (Tiles_K - preserved from input)

### partition_B and partition_C

`partition_B` and `partition_C` follow the same pattern but operate on different dimensions:
- **`partition_B`**: Extracts thread's slice using `thrfrg_B` and coordinates `(V, N, K)`
- **`partition_C`**: Extracts thread's slice using `thrfrg_C` and coordinates `(V, M, N)`

## partition_shape_A/B/C

`partition_shape_A`, `partition_shape_B`, and `partition_shape_C` are standalone functions that compute the shape of a partitioned tensor without actually partitioning it.

### Purpose

These functions are useful for **static allocation** when you need to know the shape of a partitioned tensor before runtime, such as when allocating accumulator tensors or determining buffer sizes.

### How it works

1. Creates a dummy layout from the input shape
2. Applies `thrfrg_A/B/C` to get the partitioned layout structure
3. Slices to get the fragment shape (similar to `partition_A/B/C` but without actual data)
4. Returns the shape of the resulting tensor

### Usage Pattern (Python API)

```python
# Compute accumulator shape statically
acc_shape = tiled_mma.partition_shape_C(mma_tiler_mn)  # (MMA, MMA_M, MMA_N)

# Or compute shapes for A/B (less common, as they depend on layout)
shape_A = tiled_mma.partition_shape_A(shape_MK)
shape_B = tiled_mma.partition_shape_B(shape_NK)
```

### Using with make_rmem_tensor

When you need to specify a custom dtype (rather than using the default fragment type from `make_fragment_C`), you can use `make_rmem_tensor` with the shape from `partition_shape_C`:

```python
# Compute accumulator shape
acc_shape_O = thr_mma.partition_shape_C((self._m_block_size, self._head_dim_padded))

# Allocate register memory tensor with explicit dtype
acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
acc_O.fill(0.0)
```

This pattern is useful when:
- You need a specific accumulator dtype (e.g., `Float32` for accumulation even if inputs are `Float16`)
- You want explicit control over the tensor allocation
- You're initializing the accumulator before use

### Important Notes

- **`partition_shape_C`**: Commonly used for accumulator allocation since C shape can be determined statically
- **`partition_shape_A/B`**: Less commonly used because A/B fragment shapes often depend on the actual tensor layout and thread index. For dynamic cases, use `thr_mma.partition_A/B(...).shape` instead.

## make_fragment_A/B/C

`make_fragment_A`, `make_fragment_B`, and `make_fragment_C` are static methods of `TiledMMA` (inherited from `MMA_Atom`) that create fragment tensors for MMA operations.

### Purpose

These functions allocate register (or tensor memory in B200) fragments with the appropriate layout and data type for MMA operations. Fragments are the actual data structures that threads use to perform matrix multiplication.

### Key Characteristics

1. **Input Requirements**:
   - Can accept either a partitioned tensor or a shape tuple
   - For tensor input: Expects a tensor that has already been partitioned (typically via `partition_A/B/C`)
     - For `make_fragment_A`: Input must have rank >= 3 with shape `(V, M, K, ...)` where `V` matches the fragment value size
     - For `make_fragment_B`: Input must have rank >= 3 with shape `(V, N, K, ...)` where `V` matches the fragment value size
     - For `make_fragment_C`: Input must have rank >= 3 with shape `(V, M, N, ...)` where `V` matches the fragment value size
   - For shape input: Accepts a shape tuple (commonly used with `partition_shape_C`)

2. **Fragment Types**:
   - Uses `FrgTypeA`, `FrgTypeB`, or `FrgTypeC` from the MMA traits
   - May be a view type (e.g., for tensor memory descriptors) or a value type (for register allocation)
   - For view types: Creates a view of the input tensor
   - For value types: Creates a new tensor with fragment layout matching the input shape

### Usage Pattern (Python API)

```python
# Method 1: From partitioned tensors (common for A/B)
tCsA = thr_mma.partition_A(sA)
tCrA = tiled_mma.make_fragment_A(tCsA)  # Create register fragment from partitioned shared memory tensor. Ensures layout alignment for vectorized copy.

tCsB = thr_mma.partition_B(sB)
tCrB = tiled_mma.make_fragment_B(tCsB)

# Method 2: From shape (common for C accumulator, combined with partition_shape_C)
# Compute the accumulator shape statically
qk_acc_shape = thr_mma.partition_shape_C((mma_tiler_m, mma_tiler_n))

# Allocate accumulator fragment using the computed shape
acc_qk = thr_mma.make_fragment_C(qk_acc_shape)

# Method 3: Direct from tensor (when tensor is already partitioned)
tCrA = tiled_mma.make_fragment_A(sA)  # sA must already be partitioned
tCrB = tiled_mma.make_fragment_B(sB)
```

### Combined Usage Example

A common pattern is to use `partition_shape_C` with `make_fragment_C` for accumulator allocation:

```python
# Get thread's view
thr_mma = tiled_mma.get_slice(tidx)

# Compute accumulator shape from MMA tiler dimensions
qk_acc_shape = thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))

# Allocate QK accumulator fragment
acc_qk = thr_mma.make_fragment_C(qk_acc_shape)
```

This pattern is particularly useful because:
- The accumulator shape can be determined statically from the MMA tiler
- No actual tensor data is needed to compute the shape
- The fragment can be allocated before any data partitioning occurs

### Differences Between A/B/C

- **`make_fragment_A`**: Creates fragment for left-hand side operand (M×K)
- **`make_fragment_B`**: Creates fragment for right-hand side operand (N×K)
- **`make_fragment_C`**: Creates accumulator fragment (M×N), typically stored in tensor memory (TMEM) on SM100+

## partition_fragment_A/B/C

`partition_fragment_A`, `partition_fragment_B`, and `partition_fragment_C` are methods of `ThrMMA` that combine partitioning and fragment creation in a single call.

> **Note**: These methods are currently only available in C++ API. In Python, use separate calls to `partition_A/B/C` followed by `make_fragment_A/B/C`.

### Purpose

Convenience methods that perform both `partition_A/B/C` and `make_fragment_A/B/C` in one step. Equivalent to calling `make_fragment_A(partition_A(...))`.

### How it works

```cpp
partition_fragment_A(ATensor&& atensor) const
{
  return TiledMMA::make_fragment_A(partition_A(atensor));
}
```

1. Calls `partition_A/B/C` to get the thread's slice
2. Calls `make_fragment_A/B/C` on the partitioned result
3. Returns the fragment tensor ready for MMA operations

### Usage Pattern (C++ API)

```cpp
// Get thread's view
ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);

// Partition and create fragment in one call
Tensor tCrA = thr_mma.partition_fragment_A(sA);  // Equivalent to tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
Tensor tCrB = thr_mma.partition_fragment_B(sB);
Tensor tCrC = thr_mma.partition_fragment_C(gC);
```

### When to Use

- **Use `partition_fragment_*`**: When you need both partitioning and fragment creation, and the operation is thread-specific
- **Use separate calls**: When you need the partitioned tensor view separately, or when fragment creation is done at a different scope (e.g., `tiled_mma.make_fragment_C` for accumulators)

### Example (C++)

```cpp
// Method 1: Combined (common for A/B)
Tensor tCrA = thr_mma.partition_fragment_A(sA);
Tensor tCrB = thr_mma.partition_fragment_B(sB);

// Method 2: Separate (common for C accumulator)
// Using partition_shape_C + make_fragment_C
auto acc_shape = thr_mma.partition_shape_C(make_shape(mma_tiler_m, mma_tiler_n));
Tensor tCrC = thr_mma.make_fragment_C(acc_shape);

// Method 3: Alternative - partition then make_fragment
Tensor tCgC = thr_mma.partition_C(gC);
Tensor tCrC = tiled_mma.make_fragment_C(tCgC);
```

### Python Equivalent

In Python, achieve the same result using separate calls:

```python
# Equivalent to partition_fragment_A in C++
tCsA = thr_mma.partition_A(sA)
tCrA = tiled_mma.make_fragment_A(tCsA)

# Equivalent to partition_fragment_B in C++
tCsB = thr_mma.partition_B(sB)
tCrB = tiled_mma.make_fragment_B(tCsB)

# Equivalent to partition_fragment_C in C++
tCgC = thr_mma.partition_C(gC)
tCrC = tiled_mma.make_fragment_C(tCgC)
```