from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain
from conan.tools.env import VirtualBuildEnv, VirtualRunEnv


class DashInferConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    options = {"enable_multinuma": [True, False], "arm": [True, False]}
    default_options = {
        "enable_multinuma": False,
        "arm": False,
        "protobuf/*:with_zlib": False,
        "glog/*:with_gflags": False,
        "gtest/*:build_gmock": False,
        "libunwind/*:minidebuginfo": False,
        "libunwind/*:zlibdebuginfo": False,
    }

    def requirements(self):
        self.requires("protobuf/3.18.3")
        self.requires("gtest/1.11.0")
        self.requires("glog/0.5.0")
        self.requires("pybind11/2.13.6")
        self.requires("zlib/1.2.13")
        if self.options.arm:
            self.requires("libunwind/1.7.2")
        if self.options.enable_multinuma:
            self.requires("openmpi/4.1.0")
            self.requires("hwloc/2.9.3")
            self.requires("grpc/1.50.1")
            self.requires("openssl/1.0.2t")
            self.requires("abseil/20230125.3")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()
        buildenv = VirtualBuildEnv(self)
        buildenv.generate()
        runenv = VirtualRunEnv(self)
        runenv.generate()
