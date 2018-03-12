PROJECT(darknet_preproccess)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
set(OpenCV_DIR "~/software/opencv/share/OpenCV")

add_executable(darknet_preproccess darknet_preproccess.cpp)
target_link_libraries(darknet_preproccess opencv_core opencv_highgui opencv_imgproc)
