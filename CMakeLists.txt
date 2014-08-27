#Configure logger
configure_file (logger.h.in  "${CMAKE_CURRENT_SOURCE_DIR}/logger.h")
set (math_common_SRC
  math_common.cpp
  fsolve/fsolve.cpp
  arma_eigen.cpp 
  material.cpp
  gas/gas.cpp
  solid/solid.cpp
  bessel/bessel.cpp
  rottfuncs/rottfuncs.cpp
  bessel/cbessj.cpp)

add_library(math_common ${math_common_SRC})

SET_TARGET_PROPERTIES(math_common PROPERTIES COMPILE_FLAGS
  "-DLOGLEVEL=mathcommonlogger")

# math_common should be linked to armadillo
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/gas
  ${CMAKE_CURRENT_SOURCE_DIR}/bessel
  ${CMAKE_CURRENT_SOURCE_DIR}/solid
  ${CMAKE_CURRENT_SOURCE_DIR}/rottfuncs
  ${CMAKE_CURRENT_SOURCE_DIR}/fsolve  
  )

target_link_libraries(math_common armadillo)


set_source_files_properties(
  gas.pyx solid.pyx
  PROPERTIES CYTHON_IS_CXX TRUE )
set_source_files_properties(solid.pyx PROPERTIES DEPENDS solid.pxi)
set_source_files_properties(gas.pyx PROPERTIES DEPENDS gas.pxi)
cython_add_module(gas gas.pyx)
cython_add_module(solid solid.pyx)
target_link_libraries(gas math_common armadillo)
target_link_libraries(solid math_common armadillo)
SET_TARGET_PROPERTIES(gas PROPERTIES COMPILE_FLAGS "-Wno-cpp -DLOGLEVEL=gaslogger")
SET_TARGET_PROPERTIES(solid PROPERTIES COMPILE_FLAGS "-Wno-cpp -DLOGLEVEL=solidlogger")


