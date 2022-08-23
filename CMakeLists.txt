cmake_minimum_required(VERSION 2.8)
project( main )
#find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(
  src
  include
)
set(EXECUTABLE_OUTPUT_PATH "../bin")


add_executable( main
 src/main.cpp
 include/Matrix.h
 src/Matrix.cpp

)
#target_link_libraries( main  ${OpenCV_LIBS} )
