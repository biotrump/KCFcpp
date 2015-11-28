cmake_minimum_required(VERSION 2.8)
project(KCF)

message("CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR}")
message("PROJECT_NAME : ${PROJECT_NAME}")
message("KCF_SOURCE_DIR : ${KCF_SOURCE_DIR}")
message("KCF_BINARY_DIR : ${KCF_BINARY_DIR}")

# --------------------------------------------------
# Organize output files.  On Windows this also keeps .dll files next
# to the .exe files that need them, making tests easy to run.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${KCF_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${KCF_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${KCF_BINARY_DIR}/lib)


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
#find picort
set(PICO_DIR $ENV{PICO_DIR})
message("PICO_DIR=${PICO_DIR}" )
find_path(PICO_INC picornt.h
			HINTS ${PICO_DIR}/rnt )
message("**PICO_INC=${PICO_INC}" )
if(PICO_INC)
	include_directories("${PICO_INC}")
endif(PICO_INC)

set(TARGET_ARCH $ENV{TARGET_ARCH})
message("**TARGET_ARCH=${TARGET_ARCH}")
set(PICO_OUT $ENV{PICO_OUT})
message("**PICO_OUT=${PICO_OUT} ")
find_library (PICO-RTLIB
			NAMES picort
			PATHS ${PICO_OUT}/libs/${TARGET_ARCH}
			PATH_SUFFIXES lib
			NO_DEFAULT_PATH)
message("**PICO-RTLIB=${PICO-RTLIB} ")
# --------------------------------------------------

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src)
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")
add_executable( KCF ${sourcefiles} )

qt5_use_modules(KCF Widgets  OpenGL Test Concurrent)

target_link_libraries( KCF ${PICO-RTLIB}
				${OpenCV_LIBS}
				)




