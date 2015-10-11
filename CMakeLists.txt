#Configure logger
# cmake_minimum_required (VERSION 2.8)
# project(Common)

# Extra cmake modules to find packages
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_tools)

# Require Python
find_package(PythonLibs REQUIRED)

# Initialize swig
# find_package(SWIG REQUIRED)
# include(${SWIG_USE_FILE})
# SET(CMAKE_SWIG_FLAGS -py3 -Wall -DSWIG_PYTHON)

# Eigen
# find_package(Eigen3 REQUIRED)

# Armadillo
find_package(Armadillo REQUIRED)

############################## Optimized code flags
# set(CMAKE_CXX_FLAGS " -O3 ")
# set(CMAKE_CXX_FLAGS " -O2 ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -fopenmp")
# set(CMAKE_CLANG "${CMAKE_GCC} -march=native -mtune=native -fopenmp")

# add_definitions(-DNDEBUG)	#This is for BOOST functions, and to disable assertions!
# add_definitions(-DTRACER=0)
# add_definitions(-DEIGEN_NO_DEBUG -DARMA_NO_DEBUG)

# Debug mode
# set(CMAKE_GCC "${CMAKE_GCC} -g -ggdb")
############################## End optimized code flags

############################## Compiler settings
# set(CMAKE_CXX_COMPILER g++)
# set(CMAKE_CXX_COMPILER "clang++")
# set(CMAKE_CXX_COMPILER_ARG1="clang++")

# set(CMAKE_CLANG "${CMAKE_CLANG} -pipe -Wno-unknown-warning-option -Woverloaded-virtual")
# set(CMAKE_GCC "${CMAKE_GCC} -Wno-unused-function  -ffunction-sections\
#  -fdata-sections -Wno-unused-local-typedefs -Wno-empty-body\
# -Wno-unused-but-set-variable\
# -Wno-return-local-addr -Wno-cpp \
# ")

# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pipe -fPIC -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-address")

# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_GCC}")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_GENERAL} ${CMAKE_CLANG}")
add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_4_API_VERSION)

############################## End compiler settings

# SET(CMAKE_SWIG_FLAGS -py3 -Wall -DSWIG_PYTHON)

AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/ src)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/gas src_gas)

include_directories(
  ${PYTHON_INCLUDE_PATH}
  # ${EIGEN3_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src # The source code
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gas # The source code for gas instances
  )

# The common library
add_library(common_src ${src} ${src_gas})
add_library(_common SHARED commonPYTHON_wrap.cxx)
target_link_libraries(_common common_src)
set_target_properties(_common PROPERTIES PREFIX "")

# set_source_files_properties(src/swig/common.i PROPERTIES CPLUSPLUS ON)
# set_source_files_properties(src/swig/arma_numpy.i PROPERTIES CPLUSPLUS ON)
# set_source_files_properties(src/arma_numpy.cpp  PROPERTIES COMPILE_FLAGS
#   " -Wno-#warnings")

# set(SWIG_MODULE_common_EXTRA_DEPS src/swig/arma_numpy.i)

# swig_add_module(common python src/swig/common.i)

# common should be linked to armadillo
# swig_link_libraries(common common_src
#   ${PYTHON_LIBRARIES}
#   ${ARMADILLO_LIBRARIES}
# )
set_source_files_properties( 
      PROPERTIES COMPILE_FLAGS "-Wno-missing-field-initializers -Wno-self-assign -Wno-#warnings")

