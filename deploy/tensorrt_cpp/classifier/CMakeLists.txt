# ===================
# link other/system libraries
# ===================
include_directories(/usr/local/lib)
link_directories(/usr/lib/x86_64-linux-gnu)
#link_libraries(libglog.a)


# ==============================================================
# link cuda libraries
# ==============================================================
find_package(CUDA REQUIRED)
message(STATUS "    CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "    CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})


# ==============================================================
# find opencv libraries
# ==============================================================
find_package(OpenCV 4 REQUIRED)
message(STATUS "    OpenCV_DIR = ${OpenCV_DIR}")
message(STATUS "    OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    OpenCV_LIBS = ${OpenCV_LIBS}")
include_directories(${OPENCV_INCLUDE_DIRS})


# ==============================================================
# add tensorrt libraries
# https://blog.csdn.net/dengwanyu/article/details/86710735
# ==============================================================
include_directories(/usr/local/TensorRT/TensorRT-7.2.2.3/samples/common)
include_directories(/usr/local/TensorRT/TensorRT-7.2.2.3/include)
include_directories(/usr/local/TensorRT/TensorRT-7.2.2.3/lib)
include_directories(/usr/local/TensorRT/TensorRT-7.2.2.3/lib/stubs)

set(TENSORRT_LIB_PATH "/usr/local/TensorRT/TensorRT-7.2.2.3/lib")
set(TENSORRT_LIB_PATH_ "/usr/local/TensorRT/TensorRT-7.2.2.3/lib/stubs")
message(STATUS "    TENSORRT_LIB_PATH:${TENSORRT_LIB_PATH}")
message(STATUS "    TENSORRT_LIB_PATH_:${TENSORRT_LIB_PATH_}")
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*")
file(GLOB LIBS_ "${TENSORRT_LIB_PATH_}/*")
message(STATUS "    TENSORRT_LIBS:${LIBS}")
message(STATUS "    TENSORRT_LIBS_:${LIBS_}")

add_library(trt classification.cpp gpu_allocator.cpp)

target_link_libraries(trt ${OpenCV_LIBS}  ${LIBS} ${LIBS_} ${CUDA_LIBRARIES})