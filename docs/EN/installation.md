# Installation

## Requirements

OS: Linux

Python: 3.10, 3.11, 3.12

Tested compiler version:

- gcc: 11.4.0
- arm compiler: 24.04

For multi-NUMA inference, `numactl`, `openmpi` are required:

- for Ubuntu: `apt-get install numactl libopenmpi-dev`

> For CPUs with multiple NUMA nodes, it is recommended to install the above dependencies even if you only want to run the program on one NUMA node, otherwise remote memory accesses may occur and performance is not guaranteed to be optimal.

> If there are unmentioned dependency issues during the use of the package, it is recommended to use a pre-built docker image or refer to the dockerfile under `<path_to_dashinfer>/scripts/docker` to build the development environment.

## Install Python Package

install requirements:

```shell
pip install -r examples/python/requirements.txt
```

install DashInfer python package:

- install from pip: `pip install dashinfer`
- install local package: `pip install dashinfer-<dashinfer-version>-xxx.whl`
- uninstall: `pip uninstall dashinfer -y`

## Install C++ Package

Download corresponding C++ package, and execute following command to install:

for Ubuntu:

- install: `dpkg -i DashInfer-<dashinfer-version>-ubuntu.deb`
- uninstall: `dpkg -r dashinfer`

for CentOS:

- install: `rpm -i DashInfer-<dashinfer-version>-centos.rpm`
- uninstall (x86): `rpm -e dashinfer-<dashinfer-version>-1.x86_64`
- uninstall (arm): `rpm -e dashinfer-<dashinfer-version>-1.aarch64`

# Build from Source

## Docker Environment

It is recommended to use a pre-built docker image or refer to the dockerfile under `<path_to_dashinfer>/scripts/docker` to build your own docker environment.

Pull official docker image or build from Dockerfile:

- CUDA development (Ubuntu 24.04 + CUDA 12.6 + Python 3.12):

```shell
docker pull docker.cnb.cool/thinksrc/dashinfer/dev-ubuntu-cuda:latest

# Or build from Dockerfile:
docker build -f scripts/docker/dev_ubuntu_cuda.Dockerfile -t dashinfer/dev-ubuntu-cuda:latest .
```

- CPU-only development (Ubuntu 24.04 + Python 3.12):

```shell
docker pull docker.cnb.cool/thinksrc/dashinfer/dev-x86-ubuntu:latest

# Or build from Dockerfile:
docker build -f scripts/docker/dev_x86_ubuntu.Dockerfile -t dashinfer/dev-x86-ubuntu:latest .
```

Create a container:

```shell
docker run -d --name="dashinfer-dev-${USER}" \
  --network=host --ipc=host \
  --cap-add SYS_NICE --cap-add SYS_PTRACE \
  -v $(pwd):/root/workspace/DashInfer \
  -w /root/workspace \
  -it <docker_image_tag>
```

> When creating a container, `--cap-add SYS_NICE --cap-add SYS_PTRACE --ipc=host` arguments are required, because components such as numactl and openmpi need the appropriate permissions to run. If you only need to use the single NUMA API, you may not grant this permission.

Run the container:

```shell
docker exec -it "dashinfer-dev-${USER}" /bin/bash
```

## Clone the Repository

```shell
git clone git@github.com:modelscope/dash-infer.git
git lfs pull
```

## Third-party Dependencies

DashInfer uses conan to manage third-party dependencies.

During the initial compilation, downloading third-party dependency packages may take a considerable amount of time.

## Build C++ Package

Execute the following command under DashInfer root path:

- x86 CPU

```shell
AS_PLATFORM="x86" AS_RELEASE_VERSION="1.0.0" AS_BUILD_PACKAGE=ON AS_CXX11_ABI=ON ./build.sh
```

- ARM CPU

```shell
AS_PLATFORM="armclang" AS_RELEASE_VERSION="1.0.0" AS_BUILD_PACKAGE=ON AS_CXX11_ABI=ON ./build.sh
```

> Note:
> - AS_RELEASE_VERSION: Specifies the version number of the installation package.
> - AS_BUILD_PACKAGE option: Compile Linux software installation packages. For Ubuntu, it compiles .deb packages; for CentOS, it compiles .rpm packages. The compiled .deb/.rpm packages is located in the `<path_to_dashinfer>/build`.
> - AS_CXX11_ABI: Enable or disable CXX11 ABI.

## Build Python Package

Execute the following command under `<path_to_dashinfer>/python`:

- x86 CPU

```shell
AS_PLATFORM="x86" AS_RELEASE_VERSION="1.0.0" AS_CXX11_ABI="ON" python3 setup.py bdist_wheel
```

- ARM CPU

```shell
AS_PLATFORM="armclang" AS_RELEASE_VERSION="1.0.0" AS_CXX11_ABI="ON" python3 setup.py bdist_wheel
```

The compiled .whl installer is located in the `<path_to_dashinfer>/python/dist`.
