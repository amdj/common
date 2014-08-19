#Configure logger
configure_file (logger.h.in  "${CMAKE_CURRENT_SOURCE_DIR}/logger.h")

set (math_common_SRC math_common.cpp fsolve.cpp arma_eigen.cpp
  material.cpp gas/gas.cpp solid/solid.cpp bessel/bessel.cpp rottfuncs.cpp bessel/cbessj.cpp)

add_library(math_common ${math_common_SRC})
SET_TARGET_PROPERTIES(math_common PROPERTIES COMPILE_FLAGS
  "-DLOGLEVEL=mathcommonlogger")

# math_common should be linked to armadillo
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/gas
  ${CMAKE_CURRENT_SOURCE_DIR}/bessel
  ${CMAKE_CURRENT_SOURCE_DIR}/solid)

target_link_libraries(math_common armadillo)

