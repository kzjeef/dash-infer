message("========== flash-mla ==========")

set(FLASHMLA_CUDA_VERSION
    ${CUDA_VERSION}
    CACHE STRING "flash-mla cuda version")

# FlashMLA uses its own gencode generation, so we pass raw architecture
# numbers without CMake's -real/-virtual suffixes.
set(_FMLA_ARCHS "")
foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "-(real|virtual)$" "" _arch_clean "${_arch}")
  list(APPEND _FMLA_ARCHS "${_arch_clean}")
endforeach()
list(REMOVE_DUPLICATES _FMLA_ARCHS)

# FlashMLA requires SM90+ (Hopper / Blackwell)
set(_FMLA_FILTERED_ARCHS "")
foreach(_arch IN LISTS _FMLA_ARCHS)
  string(REGEX MATCH "^([0-9]+)" _cc_base "${_arch}")
  if(_cc_base AND _cc_base GREATER_EQUAL 90)
    list(APPEND _FMLA_FILTERED_ARCHS "${_arch}")
  endif()
endforeach()

if(NOT _FMLA_FILTERED_ARCHS)
  message(STATUS "FlashMLA: No SM90+ architectures in CMAKE_CUDA_ARCHITECTURES, skipping")
  set(FLASHMLA_ENABLED OFF)
  return()
endif()

set(FLASHMLA_GPU_ARCHS
    ${_FMLA_FILTERED_ARCHS}
    CACHE STRING "flash-mla gpu archs" FORCE)
message(STATUS "FLASHMLA_GPU_ARCHS: ${FLASHMLA_GPU_ARCHS}")

set(FLASHMLA_USE_CUDA_STATIC
    ON
    CACHE BOOL "flash-mla use static CUDA")

set(FLASHMLA_USE_STATIC_LIB
    ON
    CACHE BOOL "use flash-mla static lib")

# only static link when needed, to reduce size.
if(ENABLE_NV_STATIC_LIB)
  set(FLASHMLA_USE_CUDA_STATIC ON)
else()
  set(FLASHMLA_USE_CUDA_STATIC OFF)
endif()

if(FLASHMLA_USE_STATIC_LIB)
  set(FLASHMLA_LIBRARY_NAME libflash-mla.a)
else()
  set(FLASHMLA_LIBRARY_NAME libflash-mla.so)
endif()


include(ExternalProject)

message(STATUS "build flash-mla from source")

set(FLASH_MLA_GIT_REPO https://github.com/deepseek-ai/FlashMLA.git)
set(FLASH_MLA_GIT_TAG main)
set(FLASH_MLA_GIT_PATCH ${PROJECT_SOURCE_DIR}/third_party/patch/flashmla.patch)

set(FLASHMLA_INSTALL ${INSTALL_LOCATION}/flash-mla/install)
set(FLASHMLA_LIBRARY_PATH ${FLASHMLA_INSTALL}/lib/)

ExternalProject_Add(
  project_flashmla
  GIT_REPOSITORY ${FLASH_MLA_GIT_REPO}
  GIT_TAG ${FLASH_MLA_GIT_TAG}
  GIT_SUBMODULES ""
  PATCH_COMMAND git apply --reverse --check ${FLASH_MLA_GIT_PATCH} 2>/dev/null || git apply --check ${FLASH_MLA_GIT_PATCH} 2>/dev/null && git apply ${FLASH_MLA_GIT_PATCH} || true
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/flash-mla
  SOURCE_SUBDIR csrc
  DEPENDS project_cutlass
  CMAKE_GENERATOR "Ninja"
  BUILD_COMMAND ${CMAKE_COMMAND} --build . -j2 -v
  BUILD_BYPRODUCTS ${FLASHMLA_LIBRARY_PATH}/${FLASHMLA_LIBRARY_NAME}
  USES_TERMINAL true
  CMAKE_CACHE_ARGS
      -DFLASHMLA_GPU_ARCHS:STRING=${FLASHMLA_GPU_ARCHS}
  CMAKE_ARGS
      -DFLASHMLA_CUDA_VERSION=${FLASHMLA_CUDA_VERSION}
      -DFLASHMLA_USE_CUDA_STATIC=${FLASHMLA_USE_CUDA_STATIC}
      -DCMAKE_INSTALL_PREFIX=${FLASHMLA_INSTALL}
      -DCUTLASS_INSTALL_PATH=${CUTLASS_INSTALL}
)

ExternalProject_Get_Property(project_flashmla SOURCE_DIR)
ExternalProject_Get_Property(project_flashmla SOURCE_SUBDIR)
set(FLASHMLA_INCLUDE_DIR ${SOURCE_DIR}/${SOURCE_SUBDIR})

message(STATUS "FLASHMLA_LIBRARY_PATH: ${FLASHMLA_LIBRARY_PATH}")
message(STATUS "FLASHMLA_INCLUDE_DIR: ${FLASHMLA_INCLUDE_DIR}")

if(FLASHMLA_USE_STATIC_LIB)
  add_library(flashmla::flashmla STATIC IMPORTED)
else()
  add_library(flashmla::flashmla SHARED IMPORTED)
  install(FILES ${FLASHMLA_LIBRARY_PATH}/libflash-mla.so
          DESTINATION ${CMAKE_INSTALL_LIBDIR})
  message(STATUS "libflash-mla.so installing path: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

set_target_properties(flashmla::flashmla PROPERTIES
                      IMPORTED_LOCATION ${FLASHMLA_LIBRARY_PATH}/${FLASHMLA_LIBRARY_NAME})
include_directories(${FLASHMLA_INCLUDE_DIR})
set(FLASHMLA_LIBRARY flashmla::flashmla)
set(FLASHMLA_ENABLED ON)

unset(FLASHMLA_CUDA_VERSION)
unset(FLASHMLA_GPU_ARCHS)
unset(FLASHMLA_USE_CUDA_STATIC)
message("=====================================")
