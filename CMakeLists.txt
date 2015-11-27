cmake_minimum_required(VERSION 2.8)
project(test)

# Tell CMake to run moc when necessary:
  set(CMAKE_AUTOMOC ON)
  # As moc files are generated in the binary dir, tell CMake
  # to always look for includes there:
  set(CMAKE_INCLUDE_CURRENT_DIR ON)

  # Widgets finds its own dependencies (QtGui and QtCore).
  find_package(Qt5Widgets)
#  find_package(Qt5Declarative)

  # The Qt5Widgets_INCLUDES also includes the include directories for
  # dependencies QtCore and QtGui
  include_directories(${Qt5Widgets_INCLUDES})

  # We need add -DQT_WIDGETS_LIB when using QtWidgets in Qt 5.
  add_definitions(${Qt5Widgets_DEFINITIONS})

# --------------------------------------------------
#opencv should be built first!
set(OpenCV_DIR $ENV{OPENCV_OUT})
message("++++OpenCV_DIR:${OpenCV_DIR}")
FIND_PACKAGE( OpenCV REQUIRED)
if(OpenCV_FOUND)
	message("found")
	message( "OpenCV_INCLUDE_DIRS:${OpenCV_INCLUDE_DIRS}" )
	message( "OpenCV_INCLUDES:${OpenCV_INCLUDES}")
	message( "OpenCV_LIBRARIES:${OpenCV_LIBRARIES}")
	message( "OpenCV_LIBS:${OpenCV_LIBS}")
	message( "OpenCV_DEFINITIONS:${OpenCV_DEFINITIONS}")
else()
	message(FATAL_ERROR "openCV not foud")
endif()

message("*** OpenCV_LIBS:${OpenCV_LIBS}")
message("*** OpenCV_LIBRARIES:${OpenCV_LIBRARIES}")
# --------------------------------------------------
#openMP to avoid link error : omp_xxx error!
# undefined reference to `GOMP_loop_dynamic_start
if( NOT OPENMP_FOUND)
	find_package(OpenMP)
	if(OPENMP_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif()
endif( NOT OPENMP_FOUND)
# --------------------------------------------------

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src)
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")
add_executable( KCF ${sourcefiles} )

qt5_use_modules(KCF Widgets  OpenGL Test Concurrent)

target_link_libraries( KCF ${OpenCV_LIBS})




