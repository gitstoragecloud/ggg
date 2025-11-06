#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// 从secp256k1_gpu.cu导入
extern "C" void cuda_generate_public_keys(const uint32_t *private_keys_host, uint32_t *public_keys_host, int count);

// 从keccak256_gpu.cu导入
extern "C" void cuda_batch_keccak256(const uint8_t *inputs_host, size_t input_size, uint8_t *outputs_host, int count);

typedef struct {
    uint32_t d[8];
} uint256_t;

typedef struct {
    uint256_t x;
    uint256_t y;
    int infinity;
} ec_point_t;

// GPU核函数：生成随机私钥
__global__ void generate_random_private_keys(uint256_t *private_keys, int count, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        for (int i = 0; i < 8; i++) {
            private_keys[idx].d[i] = curand(&state);
        }

        // 确保私钥在有效范围内 (小于secp256k1的n)
        // 简化处理：确保最高位不要太大
        if (private_keys[idx].d[7] == 0 && private_keys[idx].d[6] == 0 &&
            private_keys[idx].d[5] == 0 && private_keys[idx].d[4] == 0 &&
            private_keys[idx].d[3] == 0 && private_keys[idx].d[2] == 0 &&
            private_keys[idx].d[1] == 0 && private_keys[idx].d[0] == 0) {
            // 全零，重新生成
            private_keys[idx].d[0] = curand(&state) | 1;
        }
    }
}

// 将字节数组转换为十六进制字符串
void bytes_to_hex(const uint8_t *bytes, size_t len, char *hex) {
    const char hex_chars[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        hex[i * 2] = hex_chars[(bytes[i] >> 4) & 0xF];
        hex[i * 2 + 1] = hex_chars[bytes[i] & 0xF];
    }
    hex[len * 2] = '\0';
}

// 将uint256转换为字节数组 (big-endian)
void uint256_to_bytes(const uint256_t *num, uint8_t *bytes) {
    for (int i = 0; i < 8; i++) {
        uint32_t val = num->d[7 - i];
        bytes[i * 4 + 0] = (val >> 24) & 0xFF;
        bytes[i * 4 + 1] = (val >> 16) & 0xFF;
        bytes[i * 4 + 2] = (val >> 8) & 0xFF;
        bytes[i * 4 + 3] = val & 0xFF;
    }
}

// 主函数
int main(int argc, char **argv) {
    int count = 1000; // 默认生成1000个密钥对

    if (argc > 1) {
        count = atoi(argv[1]);
        if (count <= 0 || count > 10000000) {
            printf("请输入有效的数量 (1-10000000)\n");
            return 1;
        }
    }

    printf("正在生成 %d 个以太坊密钥对...\n", count);
    clock_t start = clock();

    // 分配内存
    size_t priv_size = count * sizeof(uint256_t);
    size_t pub_size = count * sizeof(ec_point_t);

    uint256_t *h_private_keys = (uint256_t *)malloc(priv_size);
    ec_point_t *h_public_keys = (ec_point_t *)malloc(pub_size);

    // 步骤1：在GPU上生成随机私钥
    uint256_t *d_private_keys;
    cudaMalloc(&d_private_keys, priv_size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    unsigned long long seed = time(NULL);
    generate_random_private_keys<<<blocksPerGrid, threadsPerBlock>>>(d_private_keys, count, seed);
    cudaMemcpy(h_private_keys, d_private_keys, priv_size, cudaMemcpyDeviceToHost);
    cudaFree(d_private_keys);

    clock_t after_privkey = clock();
    printf("私钥生成完成: %.3f 秒\n", (double)(after_privkey - start) / CLOCKS_PER_SEC);

    // 步骤2：生成公钥
    cuda_generate_public_keys((uint32_t *)h_private_keys, (uint32_t *)h_public_keys, count);

    clock_t after_pubkey = clock();
    printf("公钥生成完成: %.3f 秒\n", (double)(after_pubkey - after_privkey) / CLOCKS_PER_SEC);

    // 步骤3：计算以太坊地址
    // 公钥格式：去掉0x04前缀，连接x和y坐标 (各32字节)
    uint8_t *pubkey_bytes = (uint8_t *)malloc(count * 64);
    for (int i = 0; i < count; i++) {
        uint256_to_bytes(&h_public_keys[i].x, pubkey_bytes + i * 64);
        uint256_to_bytes(&h_public_keys[i].y, pubkey_bytes + i * 64 + 32);
    }

    // 对公钥进行Keccak-256哈希
    uint8_t *address_hashes = (uint8_t *)malloc(count * 32);
    cuda_batch_keccak256(pubkey_bytes, 64, address_hashes, count);

    clock_t after_keccak = clock();
    printf("地址计算完成: %.3f 秒\n", (double)(after_keccak - after_pubkey) / CLOCKS_PER_SEC);

    // 输出前10个结果作为示例
    printf("\n前10个密钥对示例:\n");
    printf("================================================\n");

    for (int i = 0; i < (count < 10 ? count : 10); i++) {
        // 私钥
        uint8_t priv_bytes[32];
        uint256_to_bytes(&h_private_keys[i], priv_bytes);
        char priv_hex[65];
        bytes_to_hex(priv_bytes, 32, priv_hex);

        // 公钥 (x坐标)
        char pubx_hex[65];
        bytes_to_hex(pubkey_bytes + i * 64, 32, pubx_hex);

        // 公钥 (y坐标)
        char puby_hex[65];
        bytes_to_hex(pubkey_bytes + i * 64 + 32, 32, puby_hex);

        // 地址 (Keccak-256哈希的后20字节)
        char addr_hex[41];
        bytes_to_hex(address_hashes + i * 32 + 12, 20, addr_hex);

        printf("\n密钥对 #%d:\n", i + 1);
        printf("私钥: 0x%s\n", priv_hex);
        printf("公钥X: 0x%s\n", pubx_hex);
        printf("公钥Y: 0x%s\n", puby_hex);
        printf("地址: 0x%s\n", addr_hex);
    }

    clock_t end = clock();
    double total_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n================================================\n");
    printf("总耗时: %.3f 秒\n", total_time);
    printf("速度: %.0f 个密钥对/秒\n", count / total_time);

    // 保存到文件
    FILE *f = fopen("eth_keys.txt", "w");
    if (f) {
        fprintf(f, "# 以太坊密钥对\n");
        fprintf(f, "# 生成时间: %s", ctime(&(time_t){time(NULL)}));
        fprintf(f, "# 总数: %d\n\n", count);

        for (int i = 0; i < count; i++) {
            uint8_t priv_bytes[32];
            uint256_to_bytes(&h_private_keys[i], priv_bytes);
            char priv_hex[65];
            bytes_to_hex(priv_bytes, 32, priv_hex);

            char addr_hex[41];
            bytes_to_hex(address_hashes + i * 32 + 12, 20, addr_hex);

            fprintf(f, "%s,0x%s\n", priv_hex, addr_hex);
        }
        fclose(f);
        printf("\n所有密钥已保存到 eth_keys.txt\n");
    }

    // 清理
    free(h_private_keys);
    free(h_public_keys);
    free(pubkey_bytes);
    free(address_hashes);

    return 0;
}
