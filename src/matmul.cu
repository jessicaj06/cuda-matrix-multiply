// Simple CUDA matrix multiplication: C = A * B
//
// Includes a naive global-memory kernel and a tiled shared-memory kernel,
// verified against a CPU reference implementation.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 16

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Naive kernel: each thread computes one element of C.
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled kernel using shared memory to reduce global-memory traffic.
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int tiledCol = t * TILE + threadIdx.x;
        int tiledRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < N && tiledCol < N) ? A[row * N + tiledCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (tiledRow < N && col < N) ? B[tiledRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// CPU reference for verification.
static void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static bool verify(const float* ref, const float* got, int N) {
    const float eps = 1e-2f;
    for (int i = 0; i < N * N; ++i) {
        if (fabsf(ref[i] - got[i]) > eps) {
            fprintf(stderr, "Mismatch at %d: ref=%f got=%f\n", i, ref[i],
                    got[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    printf("Matrix multiply C = A * B for N = %d\n", N);

    size_t bytes = (size_t)N * N * sizeof(float);
    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    float* hC = (float*)malloc(bytes);
    float* hRef = (float*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        hA[i] = (float)(rand() % 10);
        hB[i] = (float)(rand() % 10);
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    matmul_naive<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    matmul_cpu(hA, hB, hRef, N);
    printf("Verification: %s\n", verify(hRef, hC, N) ? "PASSED" : "FAILED");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);
    free(hRef);
    return 0;
}
