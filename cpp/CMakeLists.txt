cmake_minimum_required(VERSION 3.10)
project(ocr_onnx)

set(CMAKE_CXX_STANDARD 14)

# --------------------------
# ONNXRuntime 路径
# --------------------------
set(ONNXRUNTIME_DIR /Users/zhangxin/gitlab/ocr_cpp/osx/onnxruntime-osx-arm64-1.23.2)

# --------------------------
# OpenCV
# --------------------------
find_package(OpenCV REQUIRED)

# --------------------------
# 头文件路径
# --------------------------
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_DIR}/include
        ${PROJECT_SOURCE_DIR}        # 让 .h 可以互相 include
)

# --------------------------
# 库路径
# --------------------------
link_directories(
        ${ONNXRUNTIME_DIR}/lib
)

# --------------------------
# 源文件列表（推荐写法）
# --------------------------
set(SRC_FILES
        main.cpp
        text_det.cpp
        text_rec.cpp
        text_angle_cls.cpp
)

# --------------------------
# 生成可执行文件
# --------------------------
add_executable(ocr_app ${SRC_FILES})

# --------------------------
# 链接库
# --------------------------
target_link_libraries(ocr_app
        onnxruntime
        ${OpenCV_LIBS}
)
