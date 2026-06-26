# CUDA Matrix Multiply

A small, self-contained CUDA example that computes `C = A * B` for square
matrices, with both a naive and a tiled (shared-memory) kernel, verified
against a CPU reference.

## What's inside

- **`matmul_naive`** — one thread per output element, reads straight from
  global memory.
- **`matmul_tiled`** — loads `TILE x TILE` blocks into shared memory to cut
  global-memory traffic.
- **CPU reference + verification** — results are checked against a plain
  triple-loop implementation.

## Requirements

- NVIDIA GPU + CUDA Toolkit (`nvcc`)
- CMake 3.18+ (optional; a plain `Makefile` is also provided)

## Build & run

### With Make

```bash
make
./matmul 512        # N = 512 (default)
```

### With CMake

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build
./build/matmul 1024
```

The program prints the matrix size and whether the GPU result matches the
CPU reference:

```
Matrix multiply C = A * B for N = 512
Verification: PASSED
```

## Notes

- `TILE` is set to 16; adjust in `src/matmul.cu` to experiment with block
  sizes.
- Pass the matrix dimension `N` as the first CLI argument.

## License

MIT
