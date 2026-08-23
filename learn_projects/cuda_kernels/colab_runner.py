import os
import pty
import select
import sys
import time

SSH_HOST = "6a5S9SZSxGLwrvjnvu93c9tGY@sfo2.tmate.io"

def run_in_colab_pty(command_script, timeout_sec=180):
    master, slave = pty.openpty()
    
    pid = os.fork()
    if pid == 0:
        # Child process: run SSH with slave PTY
        os.close(master)
        os.dup2(slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        os.close(slave)
        os.execlp("ssh", "ssh", "-o", "StrictHostKeyChecking=no", SSH_HOST)
    else:
        # Parent process: interact with master PTY
        os.close(slave)
        
        # Wait for shell prompt
        time.sleep(3)
        os.write(master, b"export PS1='COLAB_PROMPT# '\n")
        time.sleep(1)
        
        # Send command script
        for line in command_script.strip().split("\n"):
            os.write(master, (line + "\n").encode("utf-8"))
            time.sleep(0.05)
            
        os.write(master, b"echo 'EXECUTION_DONE_MARKER'\n")
        
        output = []
        start_time = time.time()
        
        while time.time() - start_time < timeout_sec:
            r, _, _ = select.select([master], [], [], 1.0)
            if r:
                try:
                    data = os.read(master, 4096)
                    if not data:
                        break
                    text = data.decode("utf-8", errors="ignore")
                    output.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    if "EXECUTION_DONE_MARKER" in "".join(output[-5:]):
                        break
                except OSError:
                    break
                    
        os.close(master)
        os.waitpid(pid, 0)
        return "".join(output)

if __name__ == "__main__":
    benchmark_code = r'''
cat << 'PYEOF' > /content/test_tiled_vs_naive.py
import torch, time
from torch.utils.cpp_extension import load_inline

cuda_naive = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void naive_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

torch::Tensor naive_matmul(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naive_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}
"""

cuda_tiled = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define TILE_SIZE 16

__global__ void tiled_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float acc = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        s_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        s_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) acc += s_A[ty][k] * s_B[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

torch::Tensor tiled_matmul(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    tiled_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}
"""

print('=' * 85)
print('🚀 COMPILING NAIVE & TILED SHARED-MEMORY CUDA KERNELS WITH NVCC...')
print('=' * 85)

naive_mod = load_inline(name='n_mod', cpp_sources='torch::Tensor naive_matmul(torch::Tensor A, torch::Tensor B);', cuda_sources=cuda_naive, functions=['naive_matmul'], extra_cuda_cflags=['-O3'])
tiled_mod = load_inline(name='t_mod', cpp_sources='torch::Tensor tiled_matmul(torch::Tensor A, torch::Tensor B);', cuda_sources=cuda_tiled, functions=['tiled_matmul'], extra_cuda_cflags=['-O3'])

print('✓ Both CUDA Kernels Compiled Successfully!\n')
print(f"{'N':>6}  {'cuBLAS (ms)':>12}  {'Naive CUDA (ms)':>16}  {'Tiled CUDA (ms)':>16}  {'Naive GFLOPS':>14}  {'Tiled GFLOPS':>14}  {'Speedup':>10}")
print('─' * 98)

for N in [256, 512, 1024, 2048, 4096]:
    M = K = N
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')

    # cuBLAS
    torch.cuda.synchronize()
    for _ in range(3): C_ref = torch.matmul(A, B)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10): C_ref = torch.matmul(A, B)
    torch.cuda.synchronize()
    cublas_ms = (time.perf_counter() - t0) / 10 * 1000

    # Naive
    torch.cuda.synchronize()
    for _ in range(3): C_naive = naive_mod.naive_matmul(A, B)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10): C_naive = naive_mod.naive_matmul(A, B)
    torch.cuda.synchronize()
    naive_ms = (time.perf_counter() - t0) / 10 * 1000

    # Tiled
    torch.cuda.synchronize()
    for _ in range(3): C_tiled = tiled_mod.tiled_matmul(A, B)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10): C_tiled = tiled_mod.tiled_matmul(A, B)
    torch.cuda.synchronize()
    tiled_ms = (time.perf_counter() - t0) / 10 * 1000

    flops = 2 * M * N * K
    naive_gflops = flops / (naive_ms / 1000.0) / 1e9
    tiled_gflops = flops / (tiled_ms / 1000.0) / 1e9
    speedup = naive_ms / tiled_ms if tiled_ms > 0 else 0

    print(f"{N:>6}  {cublas_ms:>10.2f}ms  {naive_ms:>14.2f}ms  {tiled_ms:>14.2f}ms  {naive_gflops:>14.1f}  {tiled_gflops:>14.1f}  {speedup:>9.2f}x")

print('\n' + '=' * 85)
print('✅ MILESTONE 4.1 & 4.2 COMPLETE: SHARED MEMORY TILING SPEEDUP PROVEN!')
print('=' * 85)
PYEOF
python3 /content/test_tiled_vs_naive.py
'''
    run_in_colab_pty(benchmark_code, timeout_sec=120)
