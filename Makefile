# Makefile for ETH GPU Key Generator

# CUDA编译器
NVCC = nvcc

# CUDA架构 (根据您的GPU修改)
CUDA_ARCH = -arch=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86

# 编译选项
NVCC_FLAGS = -O3 --use_fast_math --expt-relaxed-constexpr

# 源文件
SOURCES = secp256k1_gpu.cu keccak256_gpu.cu eth_keygen.cu

# 输出文件
TARGET = eth_keygen

# 默认目标
all: $(TARGET)

# 编译可执行文件
$(TARGET): $(SOURCES)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET) -lcurand

# 编译共享库(用于Python绑定)
lib: $(SOURCES)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) -shared -Xcompiler -fPIC \
		secp256k1_gpu.cu keccak256_gpu.cu -o libeth_keygen.so -lcurand

# 清理
clean:
	rm -f $(TARGET) *.o libeth_keygen.so eth_keys.txt

# 测试运行
test: $(TARGET)
	./$(TARGET) 100

# 显示GPU信息
gpu-info:
	nvidia-smi

# 帮助
help:
	@echo "ETH GPU密钥生成器 - Makefile使用说明"
	@echo ""
	@echo "可用目标:"
	@echo "  make          - 编译可执行文件"
	@echo "  make lib      - 编译共享库(用于Python)"
	@echo "  make test     - 编译并测试(生成100个密钥)"
	@echo "  make clean    - 清理编译文件"
	@echo "  make gpu-info - 显示GPU信息"
	@echo "  make help     - 显示此帮助信息"
	@echo ""
	@echo "注意: 请根据您的GPU修改CUDA_ARCH变量"
	@echo "  GTX 1080 Ti:  -arch=sm_61"
	@echo "  RTX 2080 Ti:  -arch=sm_75"
	@echo "  RTX 3090:     -arch=sm_86"
	@echo "  RTX 4090:     -arch=sm_89"

.PHONY: all lib clean test gpu-info help
