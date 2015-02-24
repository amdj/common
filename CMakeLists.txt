#Configure logger

cmake_minimum_required (VERSION 3.1)

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_SWIG_FLAGS "-Wall")



set (math_common_SRC
  math_common.cpp
  fsolve/fsolve.cpp
  arma_eigen.cpp 
  gas/gas.cpp
  gas/idealgas.cpp
  solid/solid.cpp
  special/bessel.cpp
  special/skewsine.cpp
  rottfuncs/rottfuncs.cpp
  special/cbessj.cpp
  swig_files/arma_numpy.cpp
)
# math_common should be linked to armadillo
include_directories(
  /usr/include/eigen3
  
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/swig_files
  ${CMAKE_CURRENT_SOURCE_DIR}/gas
  ${CMAKE_CURRENT_SOURCE_DIR}/special
  ${CMAKE_CURRENT_SOURCE_DIR}/solid
  ${CMAKE_CURRENT_SOURCE_DIR}/rottfuncs
  ${CMAKE_CURRENT_SOURCE_DIR}/fsolve  
  )

add_library(math_common ${math_common_SRC})
SET_TARGET_PROPERTIES(math_common PROPERTIES COMPILE_FLAGS
  "-DTRACERNAME=mathcommonlogger")
target_link_libraries(math_common armadillo)

set_source_files_properties(gas.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(solid.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(arma_numpy.i PROPERTIES CPLUSPLUS ON)

swig_add_module(gas python gas.i)
swig_add_module(solid python solid.i)

# set_source_files_properties(gas.i PROPERTIES DEPENDS swig_files/arma_numpy.i)
# set_source_files_properties(solid.i PROPERTIES DEPENDS arma_numpy.i)
set(SWIG_MODULE_gas_EXTRA_DEPS ${gas_SOURCE_DIR}/swig_files/arma_numpy.i)

set_source_files_properties(gas.i PROPERTIES SWIG_FLAGS "-py3")
set_source_files_properties(solid.i PROPERTIES SWIG_FLAGS "-py3")
swig_link_libraries(solid math_common ${PYTHON_LIBRARIES} armadillo)
swig_link_libraries(gas math_common ${PYTHON_LIBRARIES} armadillo)
