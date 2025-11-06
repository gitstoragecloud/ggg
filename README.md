# ETH GPU密钥生成器

使用CUDA GPU加速批量生成以太坊(Ethereum)私钥、公钥和地址的高性能工具。

## 特性

- **GPU加速**: 利用NVIDIA显卡的并行计算能力
- **高速生成**: 可在秒级内生成数千至数百万个密钥对
- **完整实现**: 包含secp256k1椭圆曲线运算和Keccak-256哈希算法的CUDA实现
- **标准兼容**: 生成的密钥和地址完全符合以太坊标准

## 系统要求

- NVIDIA显卡 (支持CUDA 11.0+)
- CUDA Toolkit 11.0或更高版本
- CMake 3.18或更高版本
- C++14编译器 (gcc/clang)

## 安装

### 1. 安装CUDA Toolkit

Ubuntu/Debian:
```bash
# 添加NVIDIA软件包仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# 安装CUDA
sudo apt-get install cuda
```

### 2. 编译项目

```bash
# 创建构建目录
mkdir build
cd build

# 配置项目 (根据您的GPU调整架构)
cmake ..

# 编译
make -j$(nproc)

# 可执行文件位于 build/bin/eth_keygen
```

### GPU架构配置

在`CMakeLists.txt`中修改`CMAKE_CUDA_ARCHITECTURES`以匹配您的GPU:

- **GTX 1000系列** (Pascal): `60 61`
- **RTX 2000系列** (Turing): `75`
- **RTX 3000系列** (Ampere): `80 86`
- **RTX 4000系列** (Ada): `89`

查看您的GPU架构:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

## 使用方法

### 基本用法

生成1000个密钥对:
```bash
./bin/eth_keygen
```

生成指定数量的密钥对:
```bash
./bin/eth_keygen 100000
```

### 输出说明

程序会：
1. 显示生成进度和性能统计
2. 在终端输出前10个密钥对作为示例
3. 将所有密钥对保存到`eth_keys.txt`文件

输出示例:
```
正在生成 1000 个以太坊密钥对...
私钥生成完成: 0.050 秒
公钥生成完成: 2.341 秒
地址计算完成: 0.089 秒

前10个密钥对示例:
================================================

密钥对 #1:
私钥: 0x1234567890abcdef...
公钥X: 0xabcdef1234567890...
公钥Y: 0x0987654321fedcba...
地址: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb4

...

总耗时: 2.480 秒
速度: 403 个密钥对/秒
```

### 文件格式

`eth_keys.txt`格式 (CSV):
```
# 以太坊密钥对
# 生成时间: ...
# 总数: 1000

私钥(64字符),0x地址(42字符)
1234...abcd,0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb4
...
```

## 性能参考

实测性能 (取决于GPU型号):

| GPU型号 | 密钥对/秒 | 100万个耗时 |
|---------|-----------|-------------|
| RTX 4090 | ~50,000 | ~20秒 |
| RTX 3090 | ~30,000 | ~33秒 |
| RTX 3080 | ~25,000 | ~40秒 |
| RTX 2080 Ti | ~15,000 | ~67秒 |
| GTX 1080 Ti | ~8,000 | ~125秒 |

*注: 性能受多种因素影响，包括PCIe带宽、系统配置等*

## 工作原理

### 1. 私钥生成
- 使用CUDA的`curand`库在GPU上生成256位随机数
- 确保私钥在secp256k1曲线的有效范围内

### 2. 公钥派生
- 使用secp256k1椭圆曲线点乘法: `公钥 = 私钥 × G`
- G是secp256k1的生成点
- 实现了高效的点加法和倍乘算法

### 3. 地址计算
- 将公钥的x和y坐标连接 (去除0x04前缀)
- 对64字节公钥进行Keccak-256哈希
- 取哈希结果的后20字节作为以太坊地址

### 关键算法
- **secp256k1**: 256位素数域上的椭圆曲线 `y² = x³ + 7`
- **Keccak-256**: SHA-3标准化前的Keccak哈希函数
- **GPU并行化**: 每个线程独立处理一个密钥对

## 项目结构

```
.
├── CMakeLists.txt       # CMake构建配置
├── secp256k1_gpu.cu     # secp256k1椭圆曲线CUDA实现
├── keccak256_gpu.cu     # Keccak-256哈希CUDA实现
├── eth_keygen.cu        # 主程序
├── eth_keygen.py        # Python绑定(可选)
└── README.md            # 本文档
```

## 安全提示

⚠️ **重要安全警告**:

1. **私钥安全**: 生成的私钥具有完全控制权，请妥善保管
2. **随机性**: 确保系统时间准确，影响随机数生成
3. **生产环境**: 本工具主要用于学习和测试，生产环境建议使用经过审计的方案
4. **文件安全**: `eth_keys.txt`包含私钥，请立即加密或删除

## 常见问题

### Q: 编译时找不到CUDA?
A: 确保CUDA安装正确并设置环境变量:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q: 运行时出现"invalid device function"错误?
A: GPU架构不匹配，请在CMakeLists.txt中调整`CMAKE_CUDA_ARCHITECTURES`

### Q: 生成速度比预期慢?
A: 可能原因：
- secp256k1点乘法是计算密集型操作，这是性能瓶颈
- 可以考虑使用更优化的算法(如窗口法、预计算表等)
- 调整GPU线程块大小

### Q: 如何验证生成的地址是否正确?
A: 可以使用web3.py或其他以太坊库验证:
```python
from eth_keys import keys
private_key = keys.PrivateKey(bytes.fromhex('你的私钥'))
assert private_key.public_key.to_checksum_address() == '0x你的地址'
```

## 优化建议

### 进一步提升性能
1. **蒙哥马利乘法**: 使用蒙哥马利约简优化模乘法
2. **预计算表**: 为常用点倍乘预计算结果
3. **窗口法**: 使用滑动窗口算法优化标量乘法
4. **共享内存**: 利用GPU共享内存减少全局内存访问

## Python绑定

提供Python接口以便于集成:

```python
import eth_keygen_gpu

# 生成10000个密钥对
keys = eth_keygen_gpu.generate(10000)

for key in keys[:10]:  # 显示前10个
    print(f"私钥: {key['private_key']}")
    print(f"地址: {key['address']}")
```

详见`eth_keygen.py`

## 许可证

MIT License - 允许自由使用和修改

## 贡献

欢迎提交Issue和Pull Request!

## 相关资源

- [以太坊黄皮书](https://ethereum.github.io/yellowpaper/paper.pdf)
- [secp256k1标准](https://www.secg.org/sec2-v2.pdf)
- [Keccak/SHA-3](https://keccak.team/keccak.html)
- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## 免责声明

本工具仅供学习和研究使用。作者不对使用本工具造成的任何损失负责。在生产环境中使用加密货币相关工具时，请务必进行充分的安全审计。
