cmake_minimum_required(VERSION 3.18)
project(eth_keygen_gpu LANGUAGES CXX CUDA)

# 设置C++和CUDA标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置CUDA架构 (根据您的GPU调整)
# 常见架构：
# - Pascal: 60, 61
# - Volta: 70
# - Turing: 75
# - Ampere: 80, 86
# - Ada: 89
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# 设置CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math")

# 源文件
set(SOURCES
    secp256k1_gpu.cu
    keccak256_gpu.cu
    eth_keygen.cu
)

# 创建可执行文件
add_executable(eth_keygen ${SOURCES})

# 链接CUDA库
target_link_libraries(eth_keygen ${CUDA_LIBRARIES} curand)

# 设置输出目录
set_target_properties(eth_keygen PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 安装目标
install(TARGETS eth_keygen DESTINATION bin)
