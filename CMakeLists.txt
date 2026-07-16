cmake_minimum_required(VERSION 3.10)
project(kcf_tracking_failure_demo CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(kcf_demo
    src/main.cpp
    src/kcftracker.cpp
    src/TrackingFailureDetector.cpp
)

target_include_directories(kcf_demo PRIVATE ${OpenCV_INCLUDE_DIRS} src)
target_link_libraries(kcf_demo PRIVATE ${OpenCV_LIBS})
