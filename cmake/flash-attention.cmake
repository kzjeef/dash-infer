message("========== flash-attention ==========")
set(FLASHATTN_CUDA_VERSION
    ${CUDA_VERSION}
    CACHE STRING "flash-attn cuda version")
# Flash-attention uses its own gencode generation, so we pass raw architecture
# numbers without CMake's -real/-virtual suffixes.
set(_FA_ARCHS "")
foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "-(real|virtual)$" "" _arch_clean "${_arch}")
  list(APPEND _FA_ARCHS "${_arch_clean}")
endforeach()
list(REMOVE_DUPLICATES _FA_ARCHS)
set(FLASHATTN_GPU_ARCHS
    ${_FA_ARCHS}
    CACHE STRING "flash-attn gpu archs" FORCE)
list(REMOVE_ITEM FLASHATTN_GPU_ARCHS "70")
list(REMOVE_ITEM FLASHATTN_GPU_ARCHS "75")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "FLASHATTN_GPU_ARCHS: ${FLASHATTN_GPU_ARCHS}")

# Use the project-wide CUTLASS (built by cmake/cutlass.cmake) instead of
# flash-attention's bundled submodule.  This ensures a single CUTLASS version
# across the entire DashInfer build (FlashAttention, SpanAttention, FlashMLA).
set(FLASHATTN_USE_EXTERNAL_CUTLASS
    ON
    CACHE BOOL "flash-attn use external cutlass target")

set(FLASHATTN_USE_CUDA_STATIC
    ON
    CACHE BOOL "flash-attn use static CUDA")

set(FLASHATTN_USE_STATIC_LIB
    ON
    CACHE BOOL "use flash-attn static lib")

set(TARGET_HEADDIM_LIST "128;192" CACHE STRING "List of target HEADDIM values (overrides ALLOWED_HEADDIMS_LIST)")

# only static link when needed, to reduce size.
if(ENABLE_NV_STATIC_LIB)
  set(FLASHATTN_USE_CUDA_STATIC ON)
else()
  set(FLASHATTN_USE_CUDA_STATIC OFF)
endif()

if (FLASHATTN_USE_STATIC_LIB)
  set(FLASHATTN_LIBRARY_NAME libflash-attn.a)
else()
  set(FLASHATTN_LIBRARY_NAME libflash-attn.so)
endif()


include(ExternalProject)

  message(STATUS "build flash-attention from source")

    message(STATUS "Use flash-attention from external project")
    set(FLASH_ATTENTION_GIT_REPO https://github.com/Dao-AILab/flash-attention.git)
# mirror for china.
#    set(FLASH_ATTENTION_GIT_REPO https://gitee.com/lanyuflying/flash-attention.git)
    # v2.8.3 (2025-08-14), commit 060c9188beec3a8b62b33a3bfa6d5d2d44975fab
    set(FLASH_ATTENTION_GIT_TAG v2.8.3)
    set(FLASH_ATTENTION_GIT_PATCH ${PROJECT_SOURCE_DIR}/third_party/patch/flash-attn.patch)

  set(FLASHATTN_INSTALL ${INSTALL_LOCATION}/flash-attention/install)
  set(FLASHATTN_LIBRARY_PATH ${FLASHATTN_INSTALL}/lib/)

  ExternalProject_Add(
    project_flashattn
    GIT_REPOSITORY ${FLASH_ATTENTION_GIT_REPO}
    GIT_TAG ${FLASH_ATTENTION_GIT_TAG}
    GIT_SUBMODULES ""
    PATCH_COMMAND ${CMAKE_COMMAND} -E env bash -c "git apply --reverse --check ${FLASH_ATTENTION_GIT_PATCH} 2>/dev/null || git apply ${FLASH_ATTENTION_GIT_PATCH} || true"
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/flash-attention
    SOURCE_SUBDIR csrc
    DEPENDS project_cutlass
    CMAKE_GENERATOR "Ninja"
    BUILD_COMMAND ${CMAKE_COMMAND} --build . -j8
    INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory ${FLASHATTN_LIBRARY_PATH}
        COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/libflash-attn.a ${FLASHATTN_LIBRARY_PATH}/
        COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/libflash-attn.so ${FLASHATTN_LIBRARY_PATH}/ || true
    BUILD_BYPRODUCTS ${FLASHATTN_LIBRARY_PATH}/${FLASHATTN_LIBRARY_NAME}
    CMAKE_CACHE_ARGS
        -DFLASHATTN_GPU_ARCHS:STRING=${FLASHATTN_GPU_ARCHS}
        -DTARGET_HEADDIM_LIST:STRING=${TARGET_HEADDIM_LIST}
    CMAKE_ARGS
        -DFLASHATTN_CUDA_VERSION=${FLASHATTN_CUDA_VERSION}
        -DFLASHATTN_USE_EXTERNAL_CUTLASS=${FLASHATTN_USE_EXTERNAL_CUTLASS}
        -DFLASHATTN_USE_CUDA_STATIC=${FLASHATTN_USE_CUDA_STATIC}
        -DCMAKE_INSTALL_PREFIX=${FLASHATTN_INSTALL}
        -DCUTLASS_INSTALL_PATH=${CUTLASS_INSTALL}
  )

  ExternalProject_Get_Property(project_flashattn SOURCE_DIR)
  ExternalProject_Get_Property(project_flashattn SOURCE_SUBDIR)
  set(FLASHATTN_INCLUDE_DIR ${SOURCE_DIR}/${SOURCE_SUBDIR})


message(STATUS "FLASHATTN_LIBRARY_PATH: ${FLASHATTN_LIBRARY_PATH}")
message(STATUS "FLASHATTN_INCLUDE_DIR: ${FLASHATTN_INCLUDE_DIR}")

if (FLASHATTN_USE_STATIC_LIB)
  add_library(flash-attention::flash-attn STATIC IMPORTED)
else()
  add_library(flash-attention::flash-attn SHARED IMPORTED)
  install(FILES ${FLASHATTN_LIBRARY_PATH}/libflash-attn.so
          DESTINATION ${CMAKE_INSTALL_LIBDIR})
  message(STATUS "libflash-attn.so installing path: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

set_target_properties(flash-attention::flash-attn PROPERTIES
                      IMPORTED_LOCATION ${FLASHATTN_LIBRARY_PATH}/${FLASHATTN_LIBRARY_NAME})
include_directories(${FLASHATTN_INCLUDE_DIR})
set(FLASHATTN_LIBRARY flash-attention::flash-attn)

unset(TARGET_HEADDIM_LIST)
unset(FLASHATTN_CUDA_VERSION)
unset(FLASHATTN_GPU_ARCHS)
unset(FLASHATTN_USE_EXTERNAL_CUTLASS)
unset(FLASHATTN_USE_CUDA_STATIC)
message("=====================================")
