cmake_minimum_required(VERSION 3.18)
project(TorchTest VERSION 0.1.0 LANGUAGES C CXX)


set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
set(CMAKE_CXX_STANDARD 17)
add_definitions(-D_UNICODE)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)


add_executable(TorchTest main.cpp)

target_link_libraries(TorchTest ${OpenCV_LIBS} ${TORCH_LIBRARIES})