#!/usr/bin/env python3
"""
ETH GPU密钥生成器 - Python绑定

提供Python接口调用CUDA加速的以太坊密钥生成功能
"""

import ctypes
import os
import sys
from typing import List, Dict
import numpy as np

class EthKeygenGPU:
    """以太坊密钥生成GPU加速类"""

    def __init__(self, lib_path=None):
        """
        初始化GPU密钥生成器

        Args:
            lib_path: CUDA库路径，如果为None则自动查找
        """
        if lib_path is None:
            # 尝试在多个位置查找库
            possible_paths = [
                "./libeth_keygen.so",
                "./build/libeth_keygen.so",
                "/usr/local/lib/libeth_keygen.so",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    lib_path = path
                    break

        if lib_path is None or not os.path.exists(lib_path):
            raise FileNotFoundError(
                "找不到CUDA库文件。请先编译项目:\n"
                "  mkdir build && cd build && cmake .. && make"
            )

        self.lib = ctypes.CDLL(lib_path)

        # 定义函数签名
        self.lib.cuda_generate_public_keys.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),  # private_keys
            ctypes.POINTER(ctypes.c_uint32),  # public_keys
            ctypes.c_int                      # count
        ]
        self.lib.cuda_generate_public_keys.restype = None

        self.lib.cuda_batch_keccak256.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # inputs
            ctypes.c_size_t,                  # input_size
            ctypes.POINTER(ctypes.c_uint8),   # outputs
            ctypes.c_int                      # count
        ]
        self.lib.cuda_batch_keccak256.restype = None

    def generate(self, count: int = 1000) -> List[Dict[str, str]]:
        """
        生成指定数量的以太坊密钥对

        Args:
            count: 要生成的密钥对数量

        Returns:
            包含密钥信息的字典列表，每个字典包含:
                - private_key: 私钥(hex)
                - public_key_x: 公钥X坐标(hex)
                - public_key_y: 公钥Y坐标(hex)
                - address: 以太坊地址(hex with 0x prefix)
        """
        if count <= 0 or count > 10000000:
            raise ValueError("count必须在1到10000000之间")

        # 生成随机私钥
        private_keys = np.random.randint(
            1, 2**256, size=count, dtype=object
        )

        # 转换为uint32数组(8个uint32表示一个256位整数)
        priv_array = np.zeros((count, 8), dtype=np.uint32)
        for i, pk in enumerate(private_keys):
            for j in range(8):
                priv_array[i, 7-j] = (pk >> (j * 32)) & 0xFFFFFFFF

        # 调用GPU生成公钥
        pub_array = np.zeros((count, 17), dtype=np.uint32)  # x(8) + y(8) + infinity(1)

        priv_ptr = priv_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        pub_ptr = pub_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

        self.lib.cuda_generate_public_keys(priv_ptr, pub_ptr, count)

        # 准备Keccak输入(公钥的x和y坐标，各32字节)
        keccak_input = np.zeros((count, 64), dtype=np.uint8)
        for i in range(count):
            # X坐标
            for j in range(8):
                val = pub_array[i, 7-j]
                keccak_input[i, j*4:(j+1)*4] = [
                    (val >> 24) & 0xFF,
                    (val >> 16) & 0xFF,
                    (val >> 8) & 0xFF,
                    val & 0xFF
                ]
            # Y坐标
            for j in range(8):
                val = pub_array[i, 15-j]
                keccak_input[i, 32 + j*4:32 + (j+1)*4] = [
                    (val >> 24) & 0xFF,
                    (val >> 16) & 0xFF,
                    (val >> 8) & 0xFF,
                    val & 0xFF
                ]

        # 计算Keccak-256哈希
        keccak_output = np.zeros((count, 32), dtype=np.uint8)

        keccak_in_ptr = keccak_input.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        keccak_out_ptr = keccak_output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        self.lib.cuda_batch_keccak256(keccak_in_ptr, 64, keccak_out_ptr, count)

        # 组装结果
        results = []
        for i in range(count):
            # 私钥
            priv_bytes = b''.join([
                priv_array[i, 7-j].to_bytes(4, 'big')
                for j in range(8)
            ])
            priv_hex = priv_bytes.hex()

            # 公钥X
            pubx_bytes = b''.join([
                pub_array[i, 7-j].to_bytes(4, 'big')
                for j in range(8)
            ])
            pubx_hex = pubx_bytes.hex()

            # 公钥Y
            puby_bytes = b''.join([
                pub_array[i, 15-j].to_bytes(4, 'big')
                for j in range(8)
            ])
            puby_hex = puby_bytes.hex()

            # 地址(Keccak哈希的后20字节)
            address_bytes = keccak_output[i, 12:]
            address = '0x' + address_bytes.tobytes().hex()

            results.append({
                'private_key': priv_hex,
                'public_key_x': pubx_hex,
                'public_key_y': puby_hex,
                'address': address
            })

        return results

    def generate_to_file(self, count: int, filename: str = "eth_keys.txt"):
        """
        生成密钥对并保存到文件

        Args:
            count: 要生成的密钥对数量
            filename: 输出文件名
        """
        print(f"正在生成 {count} 个以太坊密钥对...")
        keys = self.generate(count)

        with open(filename, 'w') as f:
            f.write("# 以太坊密钥对\n")
            f.write(f"# 总数: {count}\n\n")
            f.write("私钥,地址\n")

            for key in keys:
                f.write(f"{key['private_key']},{key['address']}\n")

        print(f"完成! 密钥已保存到 {filename}")
        print(f"\n前5个示例:")
        for i, key in enumerate(keys[:5], 1):
            print(f"\n#{i}")
            print(f"  私钥: 0x{key['private_key']}")
            print(f"  地址: {key['address']}")


def main():
    """命令行入口"""
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print("用法: python eth_keygen.py [数量]")
            sys.exit(1)
    else:
        count = 1000

    try:
        keygen = EthKeygenGPU()
        keygen.generate_to_file(count)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"生成密钥时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
