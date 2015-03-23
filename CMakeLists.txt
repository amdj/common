#Configure logger

cmake_minimum_required (VERSION 3.1)


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
  .
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/swig_files
  ${CMAKE_CURRENT_SOURCE_DIR}/gas
  ${CMAKE_CURRENT_SOURCE_DIR}/special
  ${CMAKE_CURRENT_SOURCE_DIR}/solid
  ${CMAKE_CURRENT_SOURCE_DIR}/rottfuncs
  ${CMAKE_CURRENT_SOURCE_DIR}/fsolve  
  )

add_library(math_common ${math_common_SRC})
target_link_libraries(math_common ${ARMADILLO_LIBRARIES})

SET_TARGET_PROPERTIES(math_common PROPERTIES COMPILE_FLAGS
  "-DTRACERNAME=mathcommonlogger")

set_source_files_properties(common.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(arma_numpy.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(swig_files/arma_numpy.cpp  PROPERTIES COMPILE_FLAGS
  " -Wno-#warnings")


# This does work
set(SWIG_MODULE_common_EXTRA_DEPS swig_files/arma_numpy.i)


swig_add_module(common python common.i)
swig_link_libraries(common math_common ${PYTHON_LIBRARIES} ${ARMADILLO_LIBRARIES})
set_source_files_properties( ${swig_generated_file_fullname}
      PROPERTIES COMPILE_FLAGS "-Wno-missing-field-initializers -Wno-self-assign -Wno-#warnings")

