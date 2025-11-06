#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

// secp256k1 椭圆曲线参数
// y^2 = x^3 + 7 (mod p)

// 256位大整数用8个32位整数表示 (little-endian)
typedef struct {
    uint32_t d[8];
} uint256_t;

// 椭圆曲线点
typedef struct {
    uint256_t x;
    uint256_t y;
    int infinity;  // 无穷远点标志
} ec_point_t;

// secp256k1 素数 p
__constant__ uint256_t SECP256K1_P = {
    {0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
     0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}
};

// secp256k1 基点 G
__constant__ ec_point_t SECP256K1_G = {
    {{0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
      0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E}},
    {{0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
      0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77}},
    0
};

// 256位整数比较: a > b 返回1, a < b 返回-1, a == b 返回0
__device__ int uint256_cmp(const uint256_t *a, const uint256_t *b) {
    for (int i = 7; i >= 0; i--) {
        if (a->d[i] > b->d[i]) return 1;
        if (a->d[i] < b->d[i]) return -1;
    }
    return 0;
}

// 256位加法: r = a + b
__device__ void uint256_add(uint256_t *r, const uint256_t *a, const uint256_t *b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a->d[i] + b->d[i] + carry;
        r->d[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

// 256位减法: r = a - b
__device__ void uint256_sub(uint256_t *r, const uint256_t *a, const uint256_t *b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a->d[i] - b->d[i] - borrow;
        r->d[i] = (uint32_t)diff;
        borrow = (diff >> 32) & 1;
    }
}

// 模加法: r = (a + b) mod p
__device__ void mod_add(uint256_t *r, const uint256_t *a, const uint256_t *b, const uint256_t *p) {
    uint256_add(r, a, b);
    if (uint256_cmp(r, p) >= 0) {
        uint256_sub(r, r, p);
    }
}

// 模减法: r = (a - b) mod p
__device__ void mod_sub(uint256_t *r, const uint256_t *a, const uint256_t *b, const uint256_t *p) {
    if (uint256_cmp(a, b) >= 0) {
        uint256_sub(r, a, b);
    } else {
        uint256_t temp;
        uint256_sub(&temp, p, b);
        uint256_add(r, a, &temp);
        if (uint256_cmp(r, p) >= 0) {
            uint256_sub(r, r, p);
        }
    }
}

// 简化的模乘法 (使用蒙哥马利或其他优化算法会更快，这里使用基础实现)
__device__ void mod_mul(uint256_t *r, const uint256_t *a, const uint256_t *b, const uint256_t *p) {
    uint256_t result = {{0}};
    uint256_t temp = *a;

    for (int i = 0; i < 256; i++) {
        // 如果b的第i位是1
        int word_idx = i / 32;
        int bit_idx = i % 32;
        if ((b->d[word_idx] >> bit_idx) & 1) {
            mod_add(&result, &result, &temp, p);
        }

        // temp = temp * 2 mod p
        uint256_t temp2;
        uint256_add(&temp2, &temp, &temp);
        if (uint256_cmp(&temp2, p) >= 0) {
            uint256_sub(&temp, &temp2, p);
        } else {
            temp = temp2;
        }
    }

    *r = result;
}

// 模逆: r = a^(-1) mod p (使用扩展欧几里得算法)
__device__ void mod_inv(uint256_t *r, const uint256_t *a, const uint256_t *p) {
    // 费马小定理: a^(-1) = a^(p-2) mod p (当p是素数)
    // 使用快速幂算法
    uint256_t exp = *p;
    // exp = p - 2
    if (exp.d[0] >= 2) {
        exp.d[0] -= 2;
    } else {
        exp.d[0] = 0xFFFFFFFE;
        int i = 1;
        while (i < 8 && exp.d[i] == 0) {
            exp.d[i] = 0xFFFFFFFF;
            i++;
        }
        if (i < 8) exp.d[i]--;
    }

    uint256_t result = {{1, 0, 0, 0, 0, 0, 0, 0}};
    uint256_t base = *a;

    for (int i = 0; i < 256; i++) {
        int word_idx = i / 32;
        int bit_idx = i % 32;
        if ((exp.d[word_idx] >> bit_idx) & 1) {
            mod_mul(&result, &result, &base, p);
        }
        mod_mul(&base, &base, &base, p);
    }

    *r = result;
}

// 椭圆曲线点加法: R = P + Q
__device__ void ec_add(ec_point_t *r, const ec_point_t *p, const ec_point_t *q) {
    if (p->infinity) {
        *r = *q;
        return;
    }
    if (q->infinity) {
        *r = *p;
        return;
    }

    // 如果P == Q，使用点倍乘
    if (uint256_cmp(&p->x, &q->x) == 0 && uint256_cmp(&p->y, &q->y) == 0) {
        // 点倍乘公式
        uint256_t s, temp1, temp2;

        // s = (3 * x^2) / (2 * y) mod p
        mod_mul(&temp1, &p->x, &p->x, &SECP256K1_P);  // x^2
        mod_add(&temp2, &temp1, &temp1, &SECP256K1_P); // 2*x^2
        mod_add(&temp1, &temp2, &temp1, &SECP256K1_P); // 3*x^2

        uint256_t two_y;
        uint256_add(&two_y, &p->y, &p->y);
        if (uint256_cmp(&two_y, &SECP256K1_P) >= 0) {
            uint256_sub(&two_y, &two_y, &SECP256K1_P);
        }

        uint256_t inv;
        mod_inv(&inv, &two_y, &SECP256K1_P);
        mod_mul(&s, &temp1, &inv, &SECP256K1_P);

        // x3 = s^2 - 2*x mod p
        mod_mul(&temp1, &s, &s, &SECP256K1_P);
        uint256_t two_x;
        uint256_add(&two_x, &p->x, &p->x);
        if (uint256_cmp(&two_x, &SECP256K1_P) >= 0) {
            uint256_sub(&two_x, &two_x, &SECP256K1_P);
        }
        mod_sub(&r->x, &temp1, &two_x, &SECP256K1_P);

        // y3 = s*(x - x3) - y mod p
        mod_sub(&temp1, &p->x, &r->x, &SECP256K1_P);
        mod_mul(&temp2, &s, &temp1, &SECP256K1_P);
        mod_sub(&r->y, &temp2, &p->y, &SECP256K1_P);

        r->infinity = 0;
        return;
    }

    // 一般点加法
    uint256_t s, temp1, temp2;

    // s = (y2 - y1) / (x2 - x1) mod p
    mod_sub(&temp1, &q->y, &p->y, &SECP256K1_P);
    mod_sub(&temp2, &q->x, &p->x, &SECP256K1_P);

    uint256_t inv;
    mod_inv(&inv, &temp2, &SECP256K1_P);
    mod_mul(&s, &temp1, &inv, &SECP256K1_P);

    // x3 = s^2 - x1 - x2 mod p
    mod_mul(&temp1, &s, &s, &SECP256K1_P);
    mod_sub(&temp2, &temp1, &p->x, &SECP256K1_P);
    mod_sub(&r->x, &temp2, &q->x, &SECP256K1_P);

    // y3 = s*(x1 - x3) - y1 mod p
    mod_sub(&temp1, &p->x, &r->x, &SECP256K1_P);
    mod_mul(&temp2, &s, &temp1, &SECP256K1_P);
    mod_sub(&r->y, &temp2, &p->y, &SECP256K1_P);

    r->infinity = 0;
}

// 椭圆曲线标量乘法: R = k * P (使用double-and-add算法)
__device__ void ec_mul(ec_point_t *r, const uint256_t *k, const ec_point_t *p) {
    ec_point_t result;
    result.infinity = 1;  // 初始化为无穷远点

    ec_point_t temp = *p;

    for (int i = 0; i < 256; i++) {
        int word_idx = i / 32;
        int bit_idx = i % 32;

        if ((k->d[word_idx] >> bit_idx) & 1) {
            ec_add(&result, &result, &temp);
        }
        ec_add(&temp, &temp, &temp);  // 点倍乘
    }

    *r = result;
}

// GPU核函数：批量生成公钥
// private_keys: 输入的私钥数组
// public_keys: 输出的公钥数组 (x, y坐标)
// count: 要生成的密钥对数量
__global__ void generate_public_keys(const uint256_t *private_keys, ec_point_t *public_keys, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        // 计算公钥 = 私钥 * G
        ec_mul(&public_keys[idx], &private_keys[idx], &SECP256K1_G);
    }
}

// 主机函数：包装GPU调用
extern "C" {
    void cuda_generate_public_keys(const uint32_t *private_keys_host, uint32_t *public_keys_host, int count) {
        size_t priv_size = count * sizeof(uint256_t);
        size_t pub_size = count * sizeof(ec_point_t);

        uint256_t *d_private_keys;
        ec_point_t *d_public_keys;

        cudaMalloc(&d_private_keys, priv_size);
        cudaMalloc(&d_public_keys, pub_size);

        cudaMemcpy(d_private_keys, private_keys_host, priv_size, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

        generate_public_keys<<<blocksPerGrid, threadsPerBlock>>>(d_private_keys, d_public_keys, count);

        cudaMemcpy(public_keys_host, d_public_keys, pub_size, cudaMemcpyDeviceToHost);

        cudaFree(d_private_keys);
        cudaFree(d_public_keys);
    }
}
