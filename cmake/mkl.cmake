include(GNUInstallDirs)

set(MKL_PROJECT "extern_mkl")
include(FetchContent)
set(MKL_URL ${CMAKE_SOURCE_DIR}/third_party/mkl_2022.0.2.tar.gz)

message("MKL root: ${MKL_ROOT_DIR}, module path:${CMAKE_MODULE_PATH}")

if (${RUNTIME_THREAD} STREQUAL "TBB")
    FetchContent_Declare(${MKL_PROJECT}
        URL ${MKL_URL}
        )
    set(MKL_THREADING tbb_thread)
elseif(${RUNTIME_THREAD} STREQUAL "OMP")
    FetchContent_Declare(${MKL_PROJECT}
        URL ${MKL_URL}
        )
    set(MKL_THREADING gnu_thread)
endif()

message(STATUS "Fetch MKL from ${MKL_URL}")
FetchContent_MakeAvailable(${MKL_PROJECT})

set(MKL_ROOT_DIR
    ${${MKL_PROJECT}_SOURCE_DIR}
    CACHE PATH "MKL library")

set(MKL_ROOT
    ${${MKL_PROJECT}_SOURCE_DIR}
    CACHE PATH "MKL library")

set(MKL_ROOT ${MKL_ROOT_DIR})

set(MKL_INTERFACE lp64)
set(MKL_LINK static)
set(MKL_H ${MKL_ROOT}/include)
set(MKL_INCLUDE ${MKL_ROOT}/include)

# Try find_package first (works with MKL 2022 which bundles cmake configs)
find_package(MKL QUIET)
if (NOT MKL_FOUND)
    # Manual setup for newer MKL versions (2024+) without bundled cmake configs
    message(STATUS "MKL cmake config not found, using manual library setup")
    # MKL static libs have circular dependencies, must use --start-group/--end-group
    set(MKL_LIBRARIES
        -Wl,--start-group
        ${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a
        ${MKL_ROOT}/lib/intel64/libmkl_core.a
        ${MKL_ROOT}/lib/intel64/libmkl_gnu_thread.a
        -Wl,--end-group
        -lgomp -ldl -lpthread
    )
    add_library(MKL::MKL INTERFACE IMPORTED)
    set_target_properties(MKL::MKL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE}"
        INTERFACE_LINK_LIBRARIES "${MKL_LIBRARIES}"
    )
    message(STATUS "MKL include: ${MKL_INCLUDE}")
    message(STATUS "MKL libraries: ${MKL_LIBRARIES}")
endif()
