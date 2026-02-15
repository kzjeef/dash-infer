# Roadmap: Infrastructure & Build System Upgrade for DashInfer v3.0

> **Status**: In Progress  
> **Created**: 2026-02-14  
> **Last Updated**: 2026-02-14

## 1. Motivation

DashInfer v3.0 is a major release that introduces significant new features (CUDA Graph, MLA,
DeepSeek V3, FP8, LoRA). The build system, Docker images, and toolchain dependencies have
accumulated technical debt since v1.0 (2024), with many components now outdated or EOL.

Key issues:

- **Python 3.8/3.9 EOL**: Python 3.8 reached end-of-life in October 2024, Python 3.9 in
  October 2025. Mainstream inference engines (vLLM, SGLang) already require Python >= 3.10.
- **Conan 1.x EOL**: The project migrated to Conan 2.x internally, but Docker images and
  some documentation still reference `conan==1.60.0` (Conan 1.x).
- **CentOS 7 EOL**: CentOS 7 reached end-of-life in June 2024. Several Docker images still
  use `centos:7` as a base, which no longer receives security updates.
- **Docker image fragmentation**: 12+ Dockerfiles with inconsistent Python versions, toolchain
  versions, and dependency management.

This roadmap tracks the remaining work to fully modernize the infrastructure.

## 2. Current State Audit

### 2.1 What's Already Done (v3.0 docs update)

| Item | Before | After | Status |
|---|---|---|---|
| `setup.py` version | 2.0.0 | 3.0.0 | Done |
| `setup.py` python_requires | >= 3.8 | >= 3.10 | Done |
| `setup.py` classifiers | 3.8, 3.10 | 3.10, 3.11, 3.12 | Done |
| `requirements_dev_cpu.txt` conan | 1.60.0 | >= 2.0 | Done |
| `requirements_dev_cuda.txt` conan | 1.60.0 | >= 2.0 | Done |
| `docs/sphinx/devel/source_code_build_en.rst` | conan==1.60.0 | conan>=2.0 | Done |
| All installation docs (EN/CN/Sphinx) | Python 3.8+ | Python 3.10+ | Done |
| README.md / README_CN.md | v2.0 features | v3.0 features, dependencies | Done |

### 2.2 What Still Needs Work

#### Docker Images: Python Version (`PY_VER`)

| Dockerfile | Current Default | Target | Priority |
|---|---|---|---|
| `dev_x86_ubuntu.Dockerfile` | 3.8 | **3.10** | P0 |
| `dev_x86_centos7.Dockerfile` | 3.8 | **3.10** | P0 |
| `dev_arm_alinux.Dockerfile` | 3.8 | **3.10** | P0 |
| `dev_arm_centos8.Dockerfile` | 3.8 | **3.10** | P0 |
| `test_x86_ubuntu.Dockerfile` | 3.8 | **3.10** | P0 |
| `test_aarch64_centos.Dockerfile` | 3.8 | **3.10** | P0 |
| `release_x86_manylinux2.Dockerfile` | 3.8 | **3.10** | P0 |
| `release_aarch64_manylinux2.Dockerfile` | 3.8 | **3.10** | P0 |
| `dev_cuda_124.Dockerfile` | 3.10 | 3.10 | Already OK |
| `dev_ubuntu_cuda_124.Dockerfile` | 3.10 | 3.10 | Already OK |
| `dev_ubi8_cuda_124.Dockerfile` | 3.10 | 3.10 | Already OK |
| `test_cuda_ubuntu.Dockerfile` | 3.10 | 3.10 | Already OK |

#### Docker Images: Conan Version

| Dockerfile | Current | Target | Priority |
|---|---|---|---|
| `dev_x86_ubuntu.Dockerfile` | `conan==1.60.0` | `"conan>=2.0"` | P0 |
| `dev_x86_centos7.Dockerfile` | `conan==1.60.0` | `"conan>=2.0"` | P0 |
| `dev_arm_alinux.Dockerfile` | `conan==1.60.0` | `"conan>=2.0"` | P0 |
| `dev_arm_centos8.Dockerfile` | `conan==1.60.0` | `"conan>=2.0"` | P0 |
| `dev_cuda_124.Dockerfile` | `conan==1.60.0` | `"conan>=2.0"` | P0 |
| `dev_ubi8_cuda_124.Dockerfile` | `conan==1.60.0` | `"conan>=2.0"` | P0 |
| `dev_ubuntu_cuda_124.Dockerfile` | `conan==1.60.0` | `"conan>=2.0"` | P0 |

#### Docker Base Image Modernization

| Dockerfile | Current Base | Issue | Recommendation |
|---|---|---|---|
| `dev_x86_centos7.Dockerfile` | `centos:7` | EOL June 2024 | Migrate to `almalinux:8` or `ubuntu:22.04` |
| `dev_cuda_124.Dockerfile` | `nvidia/cuda:12.4.0-devel-centos7` | EOL base | Migrate to `nvidia/cuda:12.4.0-devel-rockylinux8` or `ubuntu22.04` |
| `test_aarch64_centos.Dockerfile` | `docker.io/centos:8` | EOL Dec 2021 | Migrate to `almalinux:8` or `rockylinux:8` |
| `dev_arm_centos8.Dockerfile` | `docker.io/centos:8` | EOL Dec 2021 | Migrate to `almalinux:8` or `rockylinux:8` |

#### Docker Registry Tags

The documentation still references `v1` tags (e.g., `dashinfer/dev-ubuntu-22.04-x86:v1`).
Once images are rebuilt, tags should be bumped to `v3` or date-tagged.

## 3. Upgrade Plan

### 3.1 Phase 1: Quick Wins (Low Risk)

These changes are mechanical and can be done without rebuilding base images.

#### 3.1.1 Update Default `PY_VER` in All Dockerfiles

| | |
|---|---|
| **Priority** | P0 |
| **Effort** | Low |
| **Risk** | Low — `PY_VER` is an ARG, users can still override |

Change `ARG PY_VER=3.8` to `ARG PY_VER=3.10` in all 8 Dockerfiles listed above.

#### 3.1.2 Update Conan Version in All Dockerfiles

| | |
|---|---|
| **Priority** | P0 |
| **Effort** | Low |
| **Risk** | Medium — Conan 2.x has different CLI and profile format |

Replace all `conan==1.60.0` with `"conan>=2.0"` in Docker pip install lines.

**Validation**: After updating, verify that `conan install` with the project's `conanfile.py`
and existing Conan profiles (`conan/conanprofile.x86_64`, `conan/conanprofile_armclang.aarch64`)
works correctly in the new Docker images.

#### 3.1.3 Update `transformers` Pin

Several Dockerfiles pin `transformers==4.41.0`. The `setup.py` requires `transformers>=4.40.0`.
Consider bumping the pinned version to a more recent release for compatibility with newer models
(especially DeepSeek V3).

---

### 3.2 Phase 2: Base Image Migration (Medium Risk)

These changes require rebuilding and re-testing all dependent images.

#### 3.2.1 Retire CentOS 7 Base Images

| | |
|---|---|
| **Priority** | P1 |
| **Effort** | Medium |
| **Risk** | Medium — may affect users who depend on glibc 2.17 compatibility |

CentOS 7 is EOL. The two affected Dockerfiles are:

- `dev_x86_centos7.Dockerfile` (`centos:7`)
- `dev_cuda_124.Dockerfile` (`nvidia/cuda:12.4.0-devel-centos7`)

**Options**:

| Option | Base Image | glibc | Pros | Cons |
|---|---|---|---|---|
| A. AlmaLinux 8 | `almalinux:8` | 2.28 | CentOS-compatible, LTS until 2029 | Higher glibc minimum |
| B. Rocky Linux 8 | `rockylinux:8` | 2.28 | CentOS-compatible, community-driven | Same as A |
| C. Ubuntu 22.04 | `ubuntu:22.04` | 2.35 | Already used for other images | Different package manager |
| D. manylinux2014 | `quay.io/pypa/manylinux2014` | 2.17 | Maximum binary compatibility | Limited dev tooling |

**Recommendation**: Option A (AlmaLinux 8) for dev/test images. Keep `manylinux2014` for
release wheel builds to maintain broad binary compatibility.

#### 3.2.2 Retire CentOS 8 Base Images

| | |
|---|---|
| **Priority** | P1 |
| **Effort** | Medium |
| **Risk** | Low |

CentOS 8 reached EOL in December 2021. Affected:

- `test_aarch64_centos.Dockerfile` (`centos:8`)
- `dev_arm_centos8.Dockerfile` (`centos:8`)

Migrate to `almalinux:8` or `rockylinux:8` (both are drop-in replacements).

---

### 3.3 Phase 3: Image Consolidation (Nice to Have)

#### 3.3.1 Reduce Dockerfile Count

| | |
|---|---|
| **Priority** | P2 |
| **Effort** | High |
| **Risk** | Medium |

Currently there are 12+ Dockerfiles with significant duplication. Consider consolidating
using multi-stage builds and build arguments:

```
Dockerfile.dev       --build-arg PLATFORM={cuda,x86,arm}  --build-arg CUDA_VER=12.4
Dockerfile.test      --build-arg PLATFORM={cuda,x86,arm}
Dockerfile.release   --build-arg PLATFORM={x86,arm}
```

This reduces maintenance burden and ensures consistency across platforms.

#### 3.3.2 Docker Image CI/CD

Add automated Docker image builds to CI, triggered by changes to Dockerfiles or
dependency files. Tag images with `v{version}` and `latest`.

#### 3.3.3 Multi-Python Wheel Builds

Set up matrix builds for Python 3.10, 3.11, 3.12 to publish all wheels to PyPI.

---

### 3.4 Phase 4: Flash Attention & CUTLASS Upgrade

#### 3.4.1 Current State

| Component | Current Version | Source | Notes |
|---|---|---|---|
| Flash Attention | v2.6.1 (`7551202`) | `Dao-AILab/flash-attention` git | SM80+, head dims 128/192 |
| CUTLASS | 3.5.0 | `third_party/cutlass_3.5.0.tgz` | Used by FA and FlashMLA |
| FlashMLA | main branch | `deepseek-ai/FlashMLA` git | SM90+ only, disabled by default |

Key files:
- `cmake/flash-attention.cmake` — FA build configuration
- `cmake/cutlass.cmake` — CUTLASS build configuration
- `cmake/flashmla.cmake` — FlashMLA build configuration
- `csrc/core/kernel/cuda/flashv2/` — FA v2 kernel wrappers

#### 3.4.2 Upgrade to Flash Attention 3 (Hopper)

| | |
|---|---|
| **Priority** | P1 |
| **Effort** | High |
| **Risk** | Medium — new API surface, Hopper-only |
| **Depends on** | CUTLASS upgrade (3.4.4) |

**What**: Integrate [Flash Attention 3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)
for Hopper (SM90a) GPUs, which leverages hardware features like WGMMA, TMA, and FP8 support.

**Why**: FA3 achieves up to 1.5-2x speedup over FA2 on H100/H200 by utilizing Hopper-specific
hardware (asynchronous warpgroup-level GEMM, tensor memory accelerator, hardware scheduler overlap).

**Key changes**:
- FA3 is Hopper-only (SM90a). FA2 must remain as fallback for SM70-SM89.
- FA3 requires CUTLASS 3.x with Hopper support.
- New features: FP8 attention (`e4m3` for Q/K/V), variable-length sequences in single kernel call,
  persistent warpgroup kernels, pipelining via `cudaTMA`.
- Kernel wrapper in `csrc/core/kernel/cuda/` needs a new `flashv3/` directory alongside `flashv2/`.
- Runtime dispatch: select FA2 vs FA3 based on SM architecture at runtime.

**Implementation sketch**:
1. Add `cmake/flash-attention-v3.cmake` or extend `flash-attention.cmake` with version selection.
2. Create `csrc/core/kernel/cuda/flashv3/` with wrappers for FA3 API.
3. Update attention operators (`batch_mha`, `batch_mqa`, `span_attn`) to dispatch FA2/FA3
   based on `DeviceProperties::sm_version`.
4. Update `TARGET_HEADDIM_LIST` if FA3 supports different head dimensions.

#### 3.4.3 Upgrade to Flash Attention 4 (Blackwell)

| | |
|---|---|
| **Priority** | P2 |
| **Effort** | High |
| **Risk** | High — SM100 experimental, API likely evolving |
| **Depends on** | 3.4.2 (FA3), SM100 support stabilization |

**What**: Integrate Flash Attention 4 for Blackwell (SM100) GPUs when available and stable.

**Why**: FA4 targets Blackwell's 5th-gen Tensor Cores with new GEMM instructions,
WGMMA improvements, and enhanced TMA capabilities. Expected significant speedup over FA3
on B100/B200 hardware.

**Key considerations**:
- SM100 support is currently experimental in DashInfer (`build.sh` includes `100` in `cuda_sm`).
- FA4 API may differ significantly from FA3. Plan for a separate `flashv4/` wrapper.
- CUTLASS must also support SM100 (likely requires CUTLASS 3.7+).
- Wait for stable release from Dao-AILab before integrating.

#### 3.4.4 CUTLASS Upgrade

| | |
|---|---|
| **Priority** | P1 |
| **Effort** | Medium |
| **Risk** | Medium — ABI and API changes between CUTLASS versions |

**What**: Upgrade CUTLASS from 3.5.0 to latest stable (3.7+).

**Why**: Required by FA3 (Hopper support), and provides improved SM90/SM100 kernels,
bug fixes, and new GEMM configurations used by FlashMLA and FP8 operators.

**Steps**:
1. Download new CUTLASS release and replace `third_party/cutlass_3.5.0.tgz`.
2. Update hash in `cmake/cutlass.cmake`.
3. Verify SpanAttention (which bundles its own CUTLASS in `span-attention/thirdparty/cutlass/`)
   is compatible or needs a separate upgrade.
4. Run full kernel test suite (`cpp_kernel_test`) across SM80/SM90.

#### 3.4.5 Migration Strategy

```
Step 1: CUTLASS 3.5.0 → 3.7+                    [prerequisite]
Step 2: Add FA3 alongside FA2 (runtime dispatch)  [SM90a, Hopper]
Step 3: Validate FA3 + MLA + FP8 integration       [DeepSeek V3 on H100]
Step 4: Add FA4 when stable                        [SM100, Blackwell]
```

Runtime dispatch logic:

```
if (sm_version >= 100 && fa4_available):
    use FlashAttention 4            # Blackwell
elif (sm_version >= 90 && fa3_available):
    use FlashAttention 3            # Hopper
else:
    use FlashAttention 2            # Ampere, Volta, Turing
```

---

## 4. Compiler Toolchain Status

| Component | Current | Status | Action Needed |
|---|---|---|---|
| gcc (x86) | devtoolset-10 (gcc 10) | OK | Consider gcc 11/12 for C++20 features |
| ARM compiler | 24.04 | OK | No change needed |
| CUDA toolkit | 12.4 (default) | OK | 12.9 supported in build.sh |
| CMake | 3.27.9 | OK | No change needed |
| clang-format | 17.0.6 | OK | No change needed |
| Flash Attention | 2.6.1 | **Outdated** | Upgrade to v3 (Hopper), v4 (Blackwell) |
| CUTLASS | 3.5.0 | **Outdated** | Upgrade to 3.7+ for FA3/SM100 support |

## 5. Summary and Priority Matrix

```
Phase 1 — Quick Wins (target: v3.0 release)
├── 3.1.1  Update PY_VER default to 3.10 in 8 Dockerfiles         [P0, Low effort]
├── 3.1.2  Update Conan to >=2.0 in 7 Dockerfiles                 [P0, Low effort]
└── 3.1.3  Bump transformers pin in Dockerfiles                    [P0, Low effort]

Phase 2 — Base Image Migration (target: v3.1)
├── 3.2.1  Retire CentOS 7 → AlmaLinux 8 (2 Dockerfiles)          [P1, Medium effort]
└── 3.2.2  Retire CentOS 8 → AlmaLinux 8 (2 Dockerfiles)          [P1, Medium effort]

Phase 3 — Consolidation (target: v3.2+)
├── 3.3.1  Consolidate 12+ Dockerfiles into 3 parameterized files  [P2, High effort]
├── 3.3.2  Add Docker image CI/CD                                  [P2, Medium effort]
└── 3.3.3  Multi-Python wheel build matrix                         [P2, Medium effort]

Phase 4 — Flash Attention & CUTLASS Upgrade (target: v3.1 ~ v3.2)
├── 3.4.4  CUTLASS 3.5.0 → 3.7+                                   [P1, Medium effort]
├── 3.4.2  Flash Attention 3 (Hopper, SM90a)                       [P1, High effort]
├── 3.4.3  Flash Attention 4 (Blackwell, SM100)                    [P2, High effort]
└── 3.4.5  Runtime FA2/FA3/FA4 dispatch                            [P1, Medium effort]
```

## 6. Validation Checklist

Before each phase is considered complete:

- [ ] All Dockerfiles build successfully
- [ ] `conan install` completes with Conan 2.x in all images
- [ ] Python wheel builds for 3.10, 3.11, 3.12
- [ ] C++ package builds on all platforms (CUDA, x86, ARM)
- [ ] Unit tests pass (`cpp_interface_test`, `cpp_operator_test`, `cpp_kernel_test`)
- [ ] Python integration tests pass (`run_full_test.sh`)
- [ ] Docker registry images tagged and pushed with new version tags
