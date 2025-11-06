#!/usr/bin/env python3
"""
以太坊GPU密钥生成器使用示例

演示如何使用Python API生成ETH密钥对
"""

from eth_keygen import EthKeygenGPU
import time

def example_basic():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)

    # 创建生成器实例
    keygen = EthKeygenGPU()

    # 生成10个密钥对
    print("\n生成10个密钥对...")
    start = time.time()
    keys = keygen.generate(10)
    elapsed = time.time() - start

    print(f"完成! 耗时: {elapsed:.3f}秒\n")

    # 显示结果
    for i, key in enumerate(keys, 1):
        print(f"密钥对 #{i}:")
        print(f"  私钥: 0x{key['private_key']}")
        print(f"  地址: {key['address']}")
        print()


def example_batch():
    """批量生成示例"""
    print("=" * 60)
    print("示例2: 批量生成")
    print("=" * 60)

    keygen = EthKeygenGPU()

    # 生成1000个密钥对
    count = 1000
    print(f"\n生成{count}个密钥对...")
    start = time.time()
    keys = keygen.generate(count)
    elapsed = time.time() - start

    print(f"完成! 耗时: {elapsed:.3f}秒")
    print(f"速度: {count/elapsed:.0f} 个/秒\n")

    # 显示前5个
    print("前5个结果:")
    for i, key in enumerate(keys[:5], 1):
        print(f"  {i}. {key['address']}")


def example_search_vanity():
    """搜索靓号地址示例"""
    print("=" * 60)
    print("示例3: 搜索靓号地址")
    print("=" * 60)

    keygen = EthKeygenGPU()

    # 搜索以特定模式开头的地址
    pattern = "0x0000"  # 寻找以0x0000开头的地址
    print(f"\n搜索以 '{pattern}' 开头的地址...")
    print("(这可能需要一些时间...)\n")

    attempts = 0
    found = []
    batch_size = 1000

    start = time.time()

    while len(found) < 5:  # 找到5个
        keys = keygen.generate(batch_size)
        attempts += batch_size

        for key in keys:
            if key['address'].startswith(pattern):
                found.append(key)
                print(f"找到! (尝试了 {attempts} 次)")
                print(f"  地址: {key['address']}")
                print(f"  私钥: 0x{key['private_key']}\n")

                if len(found) >= 5:
                    break

    elapsed = time.time() - start
    print(f"总共尝试: {attempts} 次")
    print(f"总耗时: {elapsed:.1f}秒")
    print(f"速度: {attempts/elapsed:.0f} 次/秒")


def example_save_to_file():
    """保存到文件示例"""
    print("=" * 60)
    print("示例4: 保存到文件")
    print("=" * 60)

    keygen = EthKeygenGPU()

    # 生成并保存
    print("\n生成100个密钥对并保存到文件...")
    keygen.generate_to_file(100, "example_keys.txt")
    print("\n文件已保存为 example_keys.txt")


def main():
    """运行所有示例"""
    try:
        # 基本使用
        example_basic()
        input("\n按回车继续下一个示例...")

        # 批量生成
        example_batch()
        input("\n按回车继续下一个示例...")

        # 保存到文件
        example_save_to_file()

        # 搜索靓号 (注释掉因为可能耗时较长)
        # input("\n按回车继续下一个示例(靓号搜索，可能较慢)...")
        # example_search_vanity()

    except FileNotFoundError:
        print("\n错误: 找不到CUDA库文件")
        print("请先编译项目:")
        print("  方法1 (Makefile): make lib")
        print("  方法2 (CMake): mkdir build && cd build && cmake .. && make")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
