#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

// Keccak-256 常量
__constant__ uint64_t KECCAK_ROUND_CONSTANTS[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// 循环左移
__device__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] 置换函数
__device__ void keccak_f1600(uint64_t state[25]) {
    for (int round = 0; round < 24; round++) {
        uint64_t C[5], D[5];

        // Theta 步骤
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                state[x + 5 * y] ^= D[x];
            }
        }

        // Rho 和 Pi 步骤
        uint64_t B[25];
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int r[5][5] = {
                    {0, 36, 3, 41, 18},
                    {1, 44, 10, 45, 2},
                    {62, 6, 43, 15, 61},
                    {28, 55, 25, 21, 56},
                    {27, 20, 39, 8, 14}
                };
                B[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64(state[x + 5 * y], r[x][y]);
            }
        }

        // Chi 步骤
        for (int y = 0; y < 5; y++) {
            uint64_t T[5];
            for (int x = 0; x < 5; x++) {
                T[x] = B[x + 5 * y];
            }
            for (int x = 0; x < 5; x++) {
                state[x + 5 * y] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
            }
        }

        // Iota 步骤
        state[0] ^= KECCAK_ROUND_CONSTANTS[round];
    }
}

// Keccak-256 哈希函数
__device__ void keccak256(const uint8_t *input, size_t input_len, uint8_t *output) {
    uint64_t state[25] = {0};
    const size_t rate = 136; // 1088 bits = 136 bytes (for Keccak-256)

    size_t block_size = 0;
    uint8_t *state_bytes = (uint8_t *)state;

    // 吸收阶段
    for (size_t i = 0; i < input_len; i++) {
        state_bytes[block_size++] ^= input[i];
        if (block_size == rate) {
            keccak_f1600(state);
            block_size = 0;
        }
    }

    // 填充
    state_bytes[block_size] ^= 0x01;
    state_bytes[rate - 1] ^= 0x80;
    keccak_f1600(state);

    // 输出
    for (int i = 0; i < 32; i++) {
        output[i] = state_bytes[i];
    }
}

// GPU核函数：批量计算Keccak-256
__global__ void batch_keccak256(const uint8_t *inputs, size_t input_size, uint8_t *outputs, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        const uint8_t *input = inputs + idx * input_size;
        uint8_t *output = outputs + idx * 32;
        keccak256(input, input_size, output);
    }
}

// 主机函数：包装GPU调用
extern "C" {
    void cuda_batch_keccak256(const uint8_t *inputs_host, size_t input_size, uint8_t *outputs_host, int count) {
        size_t input_total_size = count * input_size;
        size_t output_total_size = count * 32;

        uint8_t *d_inputs, *d_outputs;

        cudaMalloc(&d_inputs, input_total_size);
        cudaMalloc(&d_outputs, output_total_size);

        cudaMemcpy(d_inputs, inputs_host, input_total_size, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

        batch_keccak256<<<blocksPerGrid, threadsPerBlock>>>(d_inputs, input_size, d_outputs, count);

        cudaMemcpy(outputs_host, d_outputs, output_total_size, cudaMemcpyDeviceToHost);

        cudaFree(d_inputs);
        cudaFree(d_outputs);
    }
}
