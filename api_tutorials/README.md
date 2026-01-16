# CuTe DSL API Documentation

This directory contains comprehensive documentation for the CuTe DSL Python APIs, organized by functional category. The documentation explains both the C++ source code (in `include/cute/`) and Python API usage patterns found in `examples/python/CuTeDSL/ampere/`.

## How this tutorial is generated
Only the MMA atom part is reviewed by Wenxuan so far. Other parts are generated with the following prompt in the same dialogue referencing the MMA atom tutorial:
"great! the mma atom part is mostly complete. Now scan all the cute APIs under examples/python/CuTeDSL/ampere/ used inside cute.jit and cute.kernel regions. Group the unmentioned ones into categories (e.g. mma atom methods are mostly from include/cute/atom/mma_atom.hpp, and copy atoms methods are from include/cute/atom/copy_atom.hpp)
Write a README for each of these categories, outlining the the main functions used and explain their c++ source code and python API usage"

## Documentation Index

### Core APIs

1. **[MMA Atom APIs](mma_atom_apis.md)** - Matrix Multiply-Accumulate operations
   - `TiledMMA`, `ThrMMA`
   - `thrfrg_A/B/C`, `partition_A/B/C`
   - `make_fragment_A/B/C`, `partition_shape_A/B/C`

2. **[Copy Atom APIs](copy_atom_apis.md)** - Data movement operations
   - `make_copy_atom`, `make_tiled_copy_tv`
   - `make_tiled_copy_A/B`
   - `partition_S/D`, `retile`
   - `get_slice`

3. **[Tensor APIs](tensor_apis.md)** - Tensor creation and manipulation
   - `make_tensor`, `make_rmem_tensor`
   - `make_fragment_like`, `make_identity_tensor`
   - `local_tile`, `domain_offset`
   - `composition`, `select`, `size`, `shape`
   - `fill`, `clear`, `store`, `load`

4. **[Layout APIs](layout_apis.md)** - Layout creation and manipulation
   - `make_layout`, `make_ordered_layout`
   - `make_layout_tv`
   - `logical_divide`, `zipped_divide` `tiled_divide` `flat_divide`
   - Layout properties (shape, stride)

5. **[Algorithm APIs](algorithm_apis.md)** - High-level operations
   - `copy` - Data movement
   - `gemm` - Matrix multiplication
   - `clear`, `fill` - Tensor initialization

6. **[Architecture APIs](architecture_apis.md)** - GPU hardware access
   - `arch.thread_idx`, `arch.block_idx`
   - `arch.sync_threads`
   - `arch.cp_async_commit_group`, `arch.cp_async_wait_group`

7. **[NVGPU APIs](nvgpu_apis.md)** - NVIDIA GPU-specific features
   - `nvgpu.CopyUniversalOp`
   - `nvgpu.cpasync.CopyG2SOp`
   - `nvgpu.warp.LdMatrix8x8x16bOp`
   - `nvgpu.warp.MmaF16BF16Op`

## Quick Reference

### Common Patterns

#### Pattern 1: Tiling and Partitioning
```python
# 1. Get local tile (uses zipped_divide internally)
bidx, _, _ = cute.arch.block_idx()
blkA = cute.local_tile(mA, tiler=(TileM, TileN), coord=(bidx, None))

# 2. Partition for thread
tidx, _, _ = cute.arch.thread_idx()
thr_copy = tiled_copy.get_slice(tidx)
thrA = thr_copy.partition_S(blkA)
```

#### Pattern 2: Async Copy Pipeline
```python
# Async copy
cute.copy(tiled_copy, tSgA, tDsA)
cute.arch.cp_async_commit_group()

# Wait and sync
cute.arch.cp_async_wait_group(0)
cute.arch.sync_threads()
```

#### Pattern 3: GEMM Mainloop
```python
# Initialize
cute.clear(acc)

# Mainloop
for k_tile in range(k_tile_count):
    # Load
    cute.copy(tiled_copy_A, tCsA, tCrA)
    cute.copy(tiled_copy_B, tCsB, tCrB)
    
    # Compute
    cute.gemm(tiled_mma, tCrA, tCrB, acc)
```

## Source Code References

All documentation references C++ source code in:
- `include/cute/atom/mma_atom.hpp` - MMA operations
- `include/cute/atom/copy_atom.hpp` - Copy operations
- `include/cute/tensor_impl.hpp` - Tensor implementation
- `include/cute/layout.hpp` - Layout implementation
- `include/cute/algorithm/` - Algorithm implementations
- `include/cute/arch/` - Architecture-specific code
- `include/cute/nvgpu/` - NVIDIA GPU-specific code

## Example Code

All examples are based on real usage patterns from:
- `examples/python/CuTeDSL/ampere/` - Ampere architecture examples
- `examples/python/CuTeDSL/hopper/` - Hopper architecture examples
- `examples/python/CuTeDSL/blackwell/` - Blackwell architecture examples

## Getting Started

1. Start with **[MMA Atom APIs](mma_atom_apis.md)** to understand matrix operations
2. Read **[Copy Atom APIs](copy_atom_apis.md)** for data movement
3. Review **[Tensor APIs](tensor_apis.md)** for tensor manipulation
4. Check **[Algorithm APIs](algorithm_apis.md)** for high-level operations
5. Refer to **[Architecture APIs](architecture_apis.md)** and **[NVGPU APIs](nvgpu_apis.md)** for hardware-specific features

## Notes

- All Python API examples use `cute.jit` or `cute.kernel` decorators
- C++ source code references are provided for implementation details
- Examples are based on actual usage in the CuTe DSL codebase
- Some APIs may have architecture-specific variations

