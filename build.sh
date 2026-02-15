set -ex

clean="OFF"

# Ensure ninja is available (required by hiednn, flash-attention, span-attention)
if ! command -v ninja &> /dev/null; then
    echo "ERROR: ninja-build is required but not found."
    echo "       Install with: apt install ninja-build  (or yum install ninja-build)"
    exit 1
fi

# Ensure nvcc is on PATH (common CUDA toolkit locations)
if ! command -v nvcc &> /dev/null; then
    for cuda_dir in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11; do
        if [ -x "${cuda_dir}/bin/nvcc" ]; then
            export PATH="${cuda_dir}/bin:$PATH"
            echo "Added ${cuda_dir}/bin to PATH"
            break
        fi
    done
    if ! command -v nvcc &> /dev/null; then
        echo "WARNING: nvcc not found on PATH. CUDA build may fail."
    fi
fi

# with_platform, to support cuda/x86/arm build
with_platform="${AS_PLATFORM:-cuda}"
# cuda related version, provide a defualt value for cuda 11.4
cuda_version="${AS_CUDA_VERSION:-12.9}"
cuda_sm="${AS_CUDA_SM:-80;90a;100}"
NCCL_VERSION="${AS_NCCL_VERSION:-2.23.4}"
nccl_from_source="${AS_NCCL_FROM_SOURCE:-OFF}"
build_folder="${AS_BUILD_FOLDER:-build}"
force_conan="${AS_FORCE_CONAN:-OFF}"

## NCCL Version Map:
## the corresponding pre-build nccl will download on oss.
# | CUDA Version | NCCL Version |
# | 10.2, 11.8        | 2.15.5       |
# | 11.[3,4,6],12.1   | 2.11.4       |
# | 12.2              | 2.21.5       |
# | 12.4              | 2.23.4       |
# Set AS_NCCL_FROM_SOURCE=ON to build NCCL from GitHub source automatically.

system_nv_lib="${AS_SYSTEM_NV_LIB:-OFF}"
build_type="${AS_BUILD_TYPE:-Release}"
cuda_static="${AS_CUDA_STATIC:-OFF}"
build_package="${AS_BUILD_PACKAGE:-ON}"
enable_glibcxx11_abi="${AS_CXX11_ABI:-OFF}"
build_hiednn="${AS_BUILD_HIEDNN:-ON}"
enable_span_attn="${ENABLE_SPAN_ATTENTION:-ON}"
enable_multinuma="${ENABLE_MULTINUMA:-OFF}"
# DNNL (oneDNN): disabled by default (MKL provides better performance)
enable_dnnl="${AS_ENABLE_DNNL:-OFF}"
function clone_pull {
  GIT_URL=$1
  DIRECTORY=$2
  GIT_COMMIT=$3
  if [ -d "$DIRECTORY" ]; then
    pushd "$DIRECTORY"
    git remote update
    popd
  else
    git clone "$GIT_URL" "$DIRECTORY"
  fi
  pushd "$DIRECTORY"
  git reset --hard "$GIT_COMMIT"
  popd
}

if [ "$clean" == "ON" ]; then
    rm -rf ${build_folder}
fi

if [ ! -d "./${build_folder}" ] || [ "$force_conan" != "OFF" ] ; then
    mkdir -p ${build_folder} && cd ${build_folder}

    # Conan 2.x: select profile and options
    conan_profile="../conan/conanprofile.x86_64"
    conan_options=""

    if [ "${enable_multinuma}" == "ON" ]; then
      conan_options="${conan_options} -o enable_multinuma=True"
    fi

    if [ "${with_platform,,}" == "armclang" ]; then
      conan_profile="../conan/conanprofile_armclang.aarch64"
      conan_options="${conan_options} -o arm=True"
    fi

    if [ "$enable_glibcxx11_abi" == "ON" ]; then
      libcxx_setting="libstdc++11"
    else
      libcxx_setting="libstdc++"
    fi

    conan install ../conan \
      -pr:h ${conan_profile} \
      -s compiler.libcxx=${libcxx_setting} \
      -of . \
      --build=missing \
      --build=protobuf \
      --build=gtest \
      --build=glog \
      ${conan_options}
    cd ../
fi

cd ${build_folder}
source ./conanbuild.sh
export PATH=`pwd`/bin:$PATH

if [ "${with_platform,,}" == "cuda" ]; then
  cmake .. \
      -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake \
      -DCMAKE_BUILD_TYPE=${build_type} \
      -DBUILD_PACKAGE=${build_package} \
      -DCONFIG_ACCELERATOR_TYPE=CUDA \
      -DCONFIG_HOST_CPU_TYPE=X86 \
      -DNCCL_VERSION=${NCCL_VERSION} \
      -DNCCL_BUILD_FROM_SOURCE=${nccl_from_source} \
      -DCUDA_VERSION=${cuda_version} \
      -DCMAKE_CUDA_ARCHITECTURES="${cuda_sm}" \
      -DUSE_SYSTEM_NV_LIB=${system_nv_lib} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DENABLE_NV_STATIC_LIB=${cuda_static} \
      -DENABLE_GLIBCXX11_ABI=${enable_glibcxx11_abi} \
      -DBUILD_PYTHON=OFF \
      -DALWAYS_READ_LOAD_MODEL=OFF \
      -DENABLE_SPAN_ATTENTION=${enable_span_attn} \
      -DBUILD_HIEDNN=${build_hiednn} \
      -DENABLE_DNNL=${enable_dnnl} \
      -DENABLE_MULTINUMA=OFF
elif [ "${with_platform,,}" == "x86" ]; then
  cmake .. \
      -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake \
      -DCMAKE_BUILD_TYPE=${build_type} \
      -DBUILD_PACKAGE=${build_package} \
      -DCONFIG_ACCELERATOR_TYPE=NONE \
      -DCONFIG_HOST_CPU_TYPE=X86 \
      -DENABLE_GLIBCXX11_ABI=${enable_glibcxx11_abi} \
      -DBUILD_PYTHON=OFF \
      -DALLSPARK_CBLAS=MKL \
      -DENABLE_CUDA=OFF \
      -DENABLE_SPAN_ATTENTION=OFF \
      -DENABLE_DNNL=${enable_dnnl} \
      -DALWAYS_READ_LOAD_MODEL=ON \
      -DENABLE_MULTINUMA=${enable_multinuma}
elif [ "${with_platform,,}" == "armclang" ]; then
  cmake .. \
      -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake \
      -DCMAKE_BUILD_TYPE=${build_type} \
      -DBUILD_PACKAGE=${build_package} \
      -DCONFIG_ACCELERATOR_TYPE=NONE \
      -DCONFIG_HOST_CPU_TYPE=ARM \
      -DENABLE_GLIBCXX11_ABI=${enable_glibcxx11_abi} \
      -DBUILD_PYTHON=OFF \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DENABLE_ARMCL=ON \
      -DALLSPARK_CBLAS=BLIS \
      -DENABLE_CUDA=OFF \
      -DENABLE_AVX2=OFF \
      -DENABLE_AVX512=OFF \
      -DENABLE_ARM_V84_V9=ON \
      -DENABLE_BF16=ON \
      -DENABLE_FP16=ON \
      -DCMAKE_C_COMPILER=armclang \
      -DCMAKE_CXX_COMPILER=armclang++ \
      -DENABLE_SPAN_ATTENTION=OFF \
      -DENABLE_DNNL=${enable_dnnl} \
      -DALWAYS_READ_LOAD_MODEL=ON \
      -DENABLE_MULTINUMA=${enable_multinuma}
fi

# do the make and package.
make -j16
make install

if [ "${build_package}" == "ON" ]; then
  make package
fi

