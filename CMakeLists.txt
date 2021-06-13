cmake_minimum_required(VERSION 3.12.4 FATAL_ERROR)

project(CUTLASS LANGUAGES CXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/CUDA.cmake)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CUTLASS_NATIVE_CUDA)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --std=c++11)
endif()

include(GNUInstallDirs)

function(apply_cuda_gencode_flags TARGET)

  set(NVCC_FLAGS)
  set(CLANG_FLAGS)
  foreach(ARCH ${CUTLASS_NVCC_ARCHS_ENABLED})
    list(APPEND CLANG_FLAGS --cuda-gpu-arch=sm_${ARCH})
    set(CODES)
    list(APPEND CODES sm_${ARCH})
    list(APPEND CODES compute_${ARCH})
    list(JOIN CODES "," CODES_STR)
    list(APPEND NVCC_FLAGS -gencode=arch=compute_${ARCH},code=[${CODES_STR}])
  endforeach()

  if (CUDA_COMPILER MATCHES "[Cc]lang")
    target_compile_options(
      ${TARGET}
      PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${CLANG_FLAGS}>
      )
  else()
    target_compile_options(
      ${TARGET}
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
      )
  endif()

endfunction()

function(apply_standard_compile_options TARGET)
  target_compile_options(
    ${TARGET}
    PRIVATE
    $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:${_FLAGS}>
    )
endfunction()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(examples)
