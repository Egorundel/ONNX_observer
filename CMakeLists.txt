cmake_minimum_required(VERSION 3.10)
project(ONNX_observer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find TensorRT
find_package(TensorRT REQUIRED)
message(STATUS "TensorRT Version: ${TensorRT_VERSION_STRING}")
message(STATUS "Find TensorRT include at ${TensorRT_INCLUDE_DIRS}")
message(STATUS "Find TensorRT libraries: ${TensorRT_LIBRARIES}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include 
    ${TensorRT_INCLUDE_DIRS} 
    ${TensorRT_SAMPLE_DIR}/common
)

link_directories(
    ${TensorRT_LIBRARIES}
)

# Define source files
file(GLOB My_Source_Files ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB My_Include_Files ${PROJECT_SOURCE_DIR}/include/*.h)

# Add executable
add_executable(ONNX_observer ${My_Source_Files} ${My_Include_Files})

# Include directories
target_include_directories(ONNX_observer PUBLIC 
    ${PROJECT_SOURCE_DIR}/include 
    ${TensorRT_INCLUDE_DIRS} 
    ${TensorRT_SAMPLE_DIR}/common
)

# Link libraries
target_link_libraries(ONNX_observer PUBLIC 
    ${CMAKE_THREAD_LIBS_INIT}
    ${TensorRT_LIBRARIES}
)

# Set C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors")
