PROJECT(orb_segmentation)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_BUILD_TYPE Debug)
# 设置编译选项,允许c++11标准、O3优化、多线程。match选项可避免一些cpu上的问题
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread")

find_package( OpenCV REQUIRED )

find_package( Caffe REQUIRED )
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#二进制文件输出到bin

set(build_libraries ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${BOOST_LIBRARIES})
add_executable(orb_segmentation src/orb_segmentation.cpp)
target_link_libraries(orb_segmentation ${build_libraries})
