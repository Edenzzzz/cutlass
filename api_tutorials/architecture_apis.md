# Architecture APIs

This document explains the Python APIs for CuTe DSL Architecture operations. Source code explanations refer to `include/cute/arch/`.

## Overview

Architecture APIs provide access to GPU hardware features and thread/block indexing. These are essential for coordinating work across threads and blocks, and for using hardware-specific features like async copy operations.

## arch.thread_idx

Gets the thread index within a block.

### Purpose

Returns the 3D thread index (x, y, z) for the current thread within its block.

### Python API Usage

```python
# Get thread index
tidx, tidy, tidz = cute.arch.thread_idx()

# Typically only x is used
tidx, _, _ = cute.arch.thread_idx()
```

### Key Points

- Returns (x, y, z) thread coordinates
- `tidx` (x) is most commonly used
- Used for partitioning operations and thread-specific data access

## arch.block_idx

Gets the block index within the grid.

### Purpose

Returns the 3D block index (x, y, z) for the current block within the grid.

### Python API Usage

```python
# Get block index
bidx, bidy, bidz = cute.arch.block_idx()

# For 2D tiling
bidx, bidy, _ = cute.arch.block_idx()
tiler_coord = (bidx, bidy, None)
```

### Key Points

- Returns (x, y, z) block coordinates
- Used to determine which tile this block processes
- Combined with `local_tile` to get block-specific data

## arch.sync_threads

Synchronizes all threads in a block.

### Purpose

Waits until all threads in the block reach this point. Essential for ensuring data is ready before use (e.g., after async copies to shared memory).

### Python API Usage

```python
# Wait for async copy to complete
cute.arch.cp_async_wait_group(0)
cute.arch.sync_threads()

# Ensure shared memory is ready
cute.arch.sync_threads()
```

### When to Use

- After async copies to shared memory
- Before reading shared memory written by other threads
- To ensure all threads are at the same point in execution

## arch.cp_async_commit_group

Commits a group of async copy operations.

### Purpose

Marks a group of `cp.async` operations as ready to execute. Multiple copy operations can be batched into a group for better performance.

### Python API Usage

```python
# Issue async copies
cute.copy(tiled_copy_A, tSgA, tDsA)
cute.copy(tiled_copy_B, tSgB, tDsB)

# Commit the group
cute.arch.cp_async_commit_group()
```

### Key Points

- Batches multiple async operations
- Improves instruction throughput
- Must be followed by `cp_async_wait_group` to ensure completion

## arch.cp_async_wait_group

Waits for async copy operations to complete.

### Purpose

Waits until the number of pending async copy groups is less than or equal to the specified value. Used to manage the async copy pipeline.

### Python API Usage

```python
# Wait for all pending copies to complete
cute.arch.cp_async_wait_group(0)

# Wait until at most 1 copy is pending (for pipelining)
cute.arch.cp_async_wait_group(num_smem_stages - 2)
```

### Pipeline Pattern

```python
# Prefetch phase
for i in range(num_smem_stages - 1):
    cute.copy(tiled_copy, tSgA[i], tDsA[i])
    cute.arch.cp_async_commit_group()

# Mainloop
for k_tile in range(k_tile_count):
    # Wait for next tile to be ready
    cute.arch.cp_async_wait_group(num_smem_stages - 2)
    cute.arch.sync_threads()
    
    # Use current tile
    # ... compute ...
    
    # Start next copy
    cute.copy(tiled_copy, tSgA[next], tDsA[next])
    cute.arch.cp_async_commit_group()
```

### Key Points

- `wait_group(0)`: Wait for all copies to complete
- `wait_group(N)`: Wait until ≤N copies are pending
- Used for pipelining: overlap computation with memory transfers

## Usage Patterns

### Pattern 1: Basic Thread/Block Indexing

```python
# Get indices
tidx, _, _ = cute.arch.thread_idx()
bidx, bidy, _ = cute.arch.block_idx()

# Use for partitioning
thr_copy = tiled_copy.get_slice(tidx)
gA = cute.local_tile(mA, tiler, coord=(bidx, bidy, None))
```

### Pattern 2: Async Copy Pipeline

```python
# Prefetch
for i in range(num_stages - 1):
    cute.copy(tiled_copy, tSgA[i], tDsA[i])
    cute.arch.cp_async_commit_group()

# Mainloop
for k in range(k_tiles):
    # Wait for next tile
    cute.arch.cp_async_wait_group(num_stages - 2)
    cute.arch.sync_threads()
    
    # Compute with current tile
    cute.gemm(tiled_mma, tCrA, tCrB, acc)
    
    # Start next copy
    cute.copy(tiled_copy, tSgA[next], tDsA[next])
    cute.arch.cp_async_commit_group()
```

### Pattern 3: Synchronization Points

```python
# After async copy
cute.arch.cp_async_wait_group(0)
cute.arch.sync_threads()

# Before using shared memory
cute.arch.sync_threads()
tCsA = thr_copy.partition_S(sA)
```

