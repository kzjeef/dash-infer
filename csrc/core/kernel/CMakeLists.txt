    set(KERNEL_INC_DIR
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${THREAD_INCLUDE}
      ${PROJECT_SOURCE_DIR}/csrc
      ${PROJECT_SOURCE_DIR}/csrc/common
      ${PROJECT_SOURCE_DIR}/csrc/device
      ${PROJECT_SOURCE_DIR}/third_party/from_source/cutlass/include
      ${CMAKE_BINARY_DIR}/csrc
    )

if(NOT ALLSPARK_CBLAS MATCHES "NONE")
  list(APPEND KERNEL_INC_DIR ${CBLAS_INCLUDE_DIR})
  list(APPEND KERNEL_3RD_LIBS ${CBLAS_LIBRARY})
endif()

# check if cross compiling
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(aarch64)|(AARCH64)|(arm64)|(ARM64)")
  list(APPEND ALLSPARK_PUBLIC_DEFINITIONS -DALLSPARK_USE_NEON_)
  list(APPEND ALLSPARK_DEFINITION "-DENABLE_ARM")
  if(ENABLE_ARM_V84_V9)
    # list(APPEND ALLSPARK_PUBLIC_DEFINITIONS -DALLSPARK_USE_NEON_HIGH_VER_)
    if(${CMAKE_CXX_COMPILER} MATCHES "clang") # arm compiler
        list(APPEND ALLSPARK_CXX_FLAGS "-march=armv8.2a+dotprod+fp16+bf16+i8mm+sve+sve2")
    else()  # gcc
        list(APPEND ALLSPARK_CXX_FLAGS "-march=armv8.2-a+dotprod+fp16+bf16+i8mm+sve+sve2")
    endif()
  else()
    if(${CMAKE_CXX_COMPILER} MATCHES "clang") # arm compiler
        list(APPEND ALLSPARK_CXX_FLAGS "-march=armv8.2a+dotprod+fp16+i8mm")
    else()  # gcc
        list(APPEND ALLSPARK_CXX_FLAGS "-march=armv8.2-a+dotprod+fp16+i8mm")
    endif()
  endif()
endif() # AARCH64

file(GLOB_RECURSE src_cpu_common cpu/*.cpp)

file(
  GLOB_RECURSE
  src_avx2
  cpu/layernorm.cpp
  cpu/mha.cpp
  cpu/rotary.cpp
)

file(
  GLOB_RECURSE
  src_avx512
  cpu/famha_x64_avx512.cpp
)

file(
  GLOB_RECURSE
  src_arm
  cpu/gemm_lowp/arm/*
)

if(ENABLE_AVX2)
    set(AVX2_FLAGS "-mavx2 -mfma")
    message("AVX2 flags: ${AVX2_FLAGS}, files: ${src_avx2}")
    set_source_files_properties(${src_avx2} PROPERTIES COMPILE_FLAGS "${AVX2_FLAGS}")
endif(ENABLE_AVX2)

if(ENABLE_AVX512)
    set(AVX512_FLAGS "-mavx512f")
    message("AVX512 flags: ${AVX512_FLAGS}, files: ${src_avx512}")
    set_source_files_properties(${src_avx512} PROPERTIES COMPILE_FLAGS "${AVX512_FLAGS}")
endif(ENABLE_AVX512)

if(NOT ENABLE_ARM_V84_V9)
    # set(ARM_FLAGS "-march=armv8.2a+dotprod+fp16+bf16+i8mm+sve+sve2")
    # message("ARM Instr flags: ${ARM_FLAGS}, files: ${src_arm}")
    # set_source_files_properties(${src_arm} PROPERTIES COMPILE_FLAGS "${ARM_FLAGS}")
# else()
    foreach(file ${src_arm})
        list(REMOVE_ITEM src_cpu_common "${file}")
    endforeach()
endif(NOT ENABLE_ARM_V84_V9)

list(APPEND KERNEL_SRC ${src_cpu_common})
list(APPEND KERNEL_3RD_LIBS ${THREAD_LIB} ${CBLAS_LIBRARY})

if (ENABLE_CUDA)
    list(APPEND KERNEL_3RD_LIBS ${CUDA_3RD_PARTY_LIBS})
    set(CUDA_COMMON_KERNEL_SRC)

    file(
      GLOB
      CUDA_COMMON_KERNEL_SRC_TMP
      cuda/*.cu
    )
    list(APPEND CUDA_COMMON_KERNEL_SRC ${CUDA_COMMON_KERNEL_SRC_TMP})

    file(
      GLOB_RECURSE
      CUDA_COMMON_KERNEL_SRC_TMP
      cuda/attention/*.cu
      cuda/cache/*.cu
      cuda/cache_quant/*.cu
      cuda/cuda_fused_matmul/*.cu
      cuda/flashv2/*.cu
      cuda/gemm_lowp/*.cu
      cuda/gemm_lowp/*.cpp
      cuda/hie/*.cu
      cuda/mha_quant_cache/*.cu
      cuda/moe_lowp/*.cu
      cuda/span_attention/*.cu
      cuda/span_attention_quant/*.cu
      cuda/span_attention_v2/*.cu
      cuda/strided_softmax/*.cu
      cuda/topk_radix/*.cu
      cuda/topp/*.cu
      cuda/trivial_mha/*.cu
      cuda/utils/*.cu
    )
    list(APPEND CUDA_COMMON_KERNEL_SRC ${CUDA_COMMON_KERNEL_SRC_TMP})

      file(GLOB_RECURSE CUDA_XFORMER_SRC cuda/xformer_mha/*)
      file(GLOB_RECURSE CUDA_MOE_SRC cuda/moe/*.cu)

    list(APPEND KERNEL_SRC ${CUDA_COMMON_KERNEL_SRC} ${CUDA_XFORMER_SRC} ${CUDA_MOE_SRC})
endif()

if (ENABLE_JSON_MODE)
  list(APPEND KERNEL_INC_DIR ${LMFE_INCLUDE})
  list(APPEND KERNEL_3RD_LIBS ${LMFE_LIBRARY})
endif()

if (ENABLE_MULTINUMA)
  list(APPEND KERNEL_3RD_LIBS CONAN_PKG::openmpi)
endif()
list(APPEND ALLSPARK_3RD_LIBS CONAN_PKG::zlib)

set(ALLSPARK_PUBLIC_DEFINITIONS ${ALLSPARK_PUBLIC_DEFINITIONS})

message(STATUS "ALLSPARK_PUBLIC_DEFINITIONS:${ALLSPARK_PUBLIC_DEFINITIONS}")

add_library(allspark_kernel STATIC ${KERNEL_SRC})
set_target_properties(allspark_kernel PROPERTIES CXX_STANDARD ${CXX_STD})


target_compile_definitions(
    allspark_kernel
    PRIVATE ${ALLSPARK_PUBLIC_DEFINITIONS}
    PUBLIC ${ALLSPARK_PUBLIC_DEFINITIONS})
target_include_directories(allspark_kernel PUBLIC ${KERNEL_INC_DIR})
target_link_libraries(allspark_kernel PRIVATE ${KERNEL_3RD_LIBS} CONAN_PKG::glog CONAN_PKG::protobuf CONAN_PKG::zlib)
target_compile_definitions(allspark_kernel PUBLIC ${ALLSPARK_DEFINITION})
add_dependencies(allspark_kernel protobuf_target)
target_compile_options(allspark_kernel PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${ALLSPARK_CXX_FLAGS}>)
if(ALLSPARK_CBLAS MATCHES "BLIS")
    add_dependencies(allspark_kernel project_blis)
endif()

if (ENABLE_CUDA)
  if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
    add_dependencies(allspark_kernel project_flashattn)
  endif()
endif()
