message("======= FindNCCL")

if (USE_SYSTEM_NV_LIB)
  message("Bypass download nccl, use system provided nccl.")
  return()
endif()

include(FindPackageHandleStandardArgs)

# ---------------------------------------------------------------------------
# Option: force building NCCL from source (GitHub)
# ---------------------------------------------------------------------------
option(NCCL_BUILD_FROM_SOURCE
       "Download and build NCCL from GitHub source instead of using system installation" OFF)

# Default NCCL version (usually passed from build.sh / setup.py)
if(NOT DEFINED NCCL_VERSION)
  set(NCCL_VERSION "2.23.4")
endif()

# Library name depends on static/shared preference
if(ENABLE_NV_STATIC_LIB)
  set(NCCL_LIBNAME "nccl_static")
else()
  set(NCCL_LIBNAME "nccl")
endif()

message("find nccl with ${NCCL_LIBNAME}")

# ===========================================================================
# 1. Try to find a pre-installed (system) NCCL  -- skipped when forced source
# ===========================================================================
set(NCCL_FOUND_SYSTEM FALSE)

if(NOT NCCL_BUILD_FROM_SOURCE)
  find_path(
    NCCL_INCLUDE_DIR nccl.h
    PATH_SUFFIXES cuda/include include
                  nccl-${NCCL_VERSION}-cuda-${CUDA_VERSION}/include)

  # Prefer the versioned library name first
  find_library(
    AS_NCCL_LIBRARY_VERSIONED
    NAMES nccl-${NCCL_VERSION}
    PATH_SUFFIXES lib lib64 nccl-${NCCL_VERSION}-cuda-${CUDA_VERSION}/lib64)

  if(NOT AS_NCCL_LIBRARY_VERSIONED)
    message("find nccl without version number, searching ${CUDAToolkit_LIBRARY_DIR}")
    find_library(
      AS_NCCL_LIBRARY
      NAMES nccl
      PATHS ${CUDAToolkit_LIBRARY_DIR})
  else()
    message("found nccl with version number")
    set(AS_NCCL_LIBRARY ${AS_NCCL_LIBRARY_VERSIONED})
  endif()

  if(AS_NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
    set(NCCL_FOUND_SYSTEM TRUE)
  endif()
endif()

# ===========================================================================
# 2. If system NCCL was NOT found (or source build forced), build from GitHub
# ===========================================================================
set(NCCL_BUILT_FROM_SOURCE FALSE)

if(NOT NCCL_FOUND_SYSTEM)
  message(STATUS "---------------------------------------------------------------")
  message(STATUS "System NCCL not found or NCCL_BUILD_FROM_SOURCE=ON.")
  message(STATUS "Will download and build NCCL ${NCCL_VERSION} from GitHub source.")
  message(STATUS "---------------------------------------------------------------")

  include(ExternalProject)

  # --- Determine CUDA_HOME ------------------------------------------------
  if(DEFINED ENV{CUDA_HOME})
    set(_NCCL_CUDA_HOME "$ENV{CUDA_HOME}")
  elseif(CUDAToolkit_ROOT)
    set(_NCCL_CUDA_HOME "${CUDAToolkit_ROOT}")
  elseif(CMAKE_CUDA_COMPILER)
    get_filename_component(_NCCL_CUDA_HOME "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(_NCCL_CUDA_HOME "${_NCCL_CUDA_HOME}" DIRECTORY)
  else()
    message(FATAL_ERROR
      "Cannot determine CUDA_HOME for building NCCL from source. "
      "Please set the CUDA_HOME environment variable or ensure CUDAToolkit is found.")
  endif()
  message(STATUS "NCCL build: using CUDA_HOME=${_NCCL_CUDA_HOME}")

  # --- Directories --------------------------------------------------------
  set(NCCL_PREFIX_DIR  "${CMAKE_BINARY_DIR}/_deps/nccl")
  set(NCCL_INSTALL_DIR "${NCCL_PREFIX_DIR}/install")

  # --- Git tag  (default: v2.23.4-1) -------------------------------------
  # Normalize NCCL_VERSION: strip leading "v" and trailing "-N" release number
  # so that inputs like "2.23.4", "v2.23.4", "v2.23.4-1" all produce "v2.23.4-1"
  set(_NCCL_VER_RAW "${NCCL_VERSION}")
  string(REGEX REPLACE "^v" "" _NCCL_VER_RAW "${_NCCL_VER_RAW}")   # strip leading v
  string(REGEX REPLACE "-[0-9]+$" "" _NCCL_VER_RAW "${_NCCL_VER_RAW}")  # strip trailing -N
  if(NOT NCCL_GIT_TAG_OVERRIDE)
    set(NCCL_GIT_TAG "v${_NCCL_VER_RAW}-1")
  else()
    set(NCCL_GIT_TAG "${NCCL_GIT_TAG_OVERRIDE}")
  endif()
  message(STATUS "NCCL build: git tag = ${NCCL_GIT_TAG}")

  # --- Generate NVCC_GENCODE from CMAKE_CUDA_ARCHITECTURES ----------------
  set(_NCCL_NVCC_GENCODE "")
  if(CMAKE_CUDA_ARCHITECTURES)
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
      # Handle formats: "80", "90a", "80-real", "80-virtual"
      string(REGEX MATCH "^([0-9]+)(.*)" _match "${_arch}")
      set(_arch_num    "${CMAKE_MATCH_1}")
      set(_arch_suffix "${CMAKE_MATCH_2}")

      if(_arch_suffix STREQUAL "a")
        # sm_90a  (architecture with feature suffix)
        string(APPEND _NCCL_NVCC_GENCODE
          " -gencode=arch=compute_${_arch_num}a,code=sm_${_arch_num}a")
      elseif(_arch_suffix STREQUAL "-virtual")
        string(APPEND _NCCL_NVCC_GENCODE
          " -gencode=arch=compute_${_arch_num},code=compute_${_arch_num}")
      else()
        # plain number or "-real"
        string(APPEND _NCCL_NVCC_GENCODE
          " -gencode=arch=compute_${_arch_num},code=sm_${_arch_num}")
      endif()
    endforeach()
    string(STRIP "${_NCCL_NVCC_GENCODE}" _NCCL_NVCC_GENCODE)
    message(STATUS "NCCL build: NVCC_GENCODE = ${_NCCL_NVCC_GENCODE}")
  endif()

  # --- Parallel job count -------------------------------------------------
  include(ProcessorCount)
  ProcessorCount(_NCCL_JOBS)
  if(_NCCL_JOBS EQUAL 0)
    set(_NCCL_JOBS 8)
  endif()

  # --- Assemble build & install commands ----------------------------------
  if(_NCCL_NVCC_GENCODE)
    set(_NCCL_BUILD_CMD
      make -j${_NCCL_JOBS} src.build
        "CUDA_HOME=${_NCCL_CUDA_HOME}"
        "NVCC_GENCODE=${_NCCL_NVCC_GENCODE}")
  else()
    set(_NCCL_BUILD_CMD
      make -j${_NCCL_JOBS} src.build
        "CUDA_HOME=${_NCCL_CUDA_HOME}")
  endif()

  set(_NCCL_INSTALL_CMD
    make install
      "PREFIX=${NCCL_INSTALL_DIR}")

  # --- ExternalProject_Add ------------------------------------------------
  ExternalProject_Add(nccl_external
    GIT_REPOSITORY  https://github.com/NVIDIA/nccl.git
    GIT_TAG         ${NCCL_GIT_TAG}
    GIT_SHALLOW     TRUE
    PREFIX          ${NCCL_PREFIX_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ${_NCCL_BUILD_CMD}
    INSTALL_COMMAND   ${_NCCL_INSTALL_CMD}
    BUILD_IN_SOURCE   TRUE
    LOG_DOWNLOAD      TRUE
    LOG_BUILD         TRUE
    LOG_INSTALL       TRUE
  )

  # --- Pre-create install directories so CMake doesn't error during configure ---
  file(MAKE_DIRECTORY "${NCCL_INSTALL_DIR}/include")
  file(MAKE_DIRECTORY "${NCCL_INSTALL_DIR}/lib")

  # --- Expected output paths (files will exist after build) ---------------
  set(NCCL_INCLUDE_DIR "${NCCL_INSTALL_DIR}/include"
      CACHE PATH "NCCL include directory" FORCE)

  if(ENABLE_NV_STATIC_LIB)
    set(AS_NCCL_LIBRARY "${NCCL_INSTALL_DIR}/lib/libnccl_static.a"
        CACHE FILEPATH "NCCL library path" FORCE)
  else()
    set(AS_NCCL_LIBRARY "${NCCL_INSTALL_DIR}/lib/libnccl.so"
        CACHE FILEPATH "NCCL library path" FORCE)
  endif()

  set(NCCL_BUILT_FROM_SOURCE TRUE)
endif()

# ===========================================================================
# 3. Create the imported target  CUDA::nccl  /  CUDA::nccl_static
# ===========================================================================
if(ENABLE_NV_STATIC_LIB)
  message("add nccl static lib")
  add_library(CUDA::${NCCL_LIBNAME} STATIC IMPORTED GLOBAL)
else()
  message("add nccl shared lib")
  add_library(CUDA::${NCCL_LIBNAME} SHARED IMPORTED GLOBAL)
endif()

set_property(TARGET CUDA::${NCCL_LIBNAME}
             PROPERTY IMPORTED_LOCATION ${AS_NCCL_LIBRARY})
set_property(TARGET CUDA::${NCCL_LIBNAME}
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${NCCL_INCLUDE_DIR})

# When built from source, downstream targets must wait for the build
if(NCCL_BUILT_FROM_SOURCE)
  add_dependencies(CUDA::${NCCL_LIBNAME} nccl_external)
endif()

# ===========================================================================
# 4. Install NCCL shared libraries (skip for static builds)
# ===========================================================================
if(NOT ENABLE_NV_STATIC_LIB)
  if(NCCL_BUILT_FROM_SOURCE)
    # Install all NCCL shared objects produced by the source build
    install(DIRECTORY "${NCCL_INSTALL_DIR}/lib/"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            FILES_MATCHING PATTERN "libnccl*")
  else()
    get_filename_component(NCCL_LIB_DIR ${AS_NCCL_LIBRARY} DIRECTORY)
    file(GLOB NCCL_LIBS "${NCCL_LIB_DIR}/*nccl.so*")
    install(FILES ${NCCL_LIBS}
            DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
endif()

# ===========================================================================
# 5. Summary
# ===========================================================================
message("find nccl at ${NCCL_INCLUDE_DIR} lib: ${AS_NCCL_LIBRARY}")

if(NOT NCCL_BUILT_FROM_SOURCE)
  find_package_handle_standard_args(NCCL DEFAULT_MSG
    NCCL_INCLUDE_DIR AS_NCCL_LIBRARY)
endif()

if(NCCL_BUILT_FROM_SOURCE)
  message(STATUS "NCCL: will be built from source (${NCCL_GIT_TAG})")
  message(STATUS "  include => ${NCCL_INCLUDE_DIR}")
  message(STATUS "  library => ${AS_NCCL_LIBRARY}")
else()
  message(STATUS "Found NCCL: success, library path : ${AS_NCCL_LIBRARY}")
endif()
