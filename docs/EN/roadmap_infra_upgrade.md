# Roadmap: Infrastructure & Build System Upgrade for DashInfer v3.0

> **Status**: In Progress  
> **Created**: 2026-02-14  
> **Last Updated**: 2026-02-15

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

### 2.2 Docker Cleanup (Completed 2026-02-15)

#### Deleted Obsolete Dockerfiles (6 files)

All CentOS 7/8, UBI 8, and ALinux Dockerfiles have been **deleted**:

| Deleted Dockerfile | Former Base | Reason |
|---|---|---|
| `dev_x86_centos7.Dockerfile` | `centos:7` | CentOS 7 EOL June 2024 |
| `dev_cuda_124.Dockerfile` | `nvidia/cuda:12.4.0-devel-centos7` | CentOS 7 EOL |
| `dev_arm_centos8.Dockerfile` | `docker.io/centos:8` | CentOS 8 EOL Dec 2021 |
| `test_aarch64_centos.Dockerfile` | `docker.io/centos:8` | CentOS 8 EOL |
| `dev_ubi8_cuda_124.Dockerfile` | `nvidia/cuda:12.4.1-devel-ubi8` | Consolidated to Ubuntu |
| `dev_arm_alinux.Dockerfile` | ALinux 3 | Consolidated to Ubuntu |

#### Upgraded Ubuntu Dockerfiles (4 files)

| Dockerfile | Before | After |
|---|---|---|
| `dev_ubuntu_cuda.Dockerfile` (renamed) | Ubuntu 22.04 + CUDA 12.4 + Conda + Python 3.10 + Conan 1.60 | **Ubuntu 24.04 + CUDA 12.6 + venv + Python 3.12 + Conan >=2.0** |
| `dev_x86_ubuntu.Dockerfile` | Ubuntu 22.04 + Conda + Python 3.8 + Conan 1.60 | **Ubuntu 24.04 + venv + Python 3.12 + Conan >=2.0** |
| `test_cuda_ubuntu.Dockerfile` | Ubuntu 22.04 + CUDA 12.4 + Conda + Python 3.10 | **Ubuntu 24.04 + CUDA 12.6 runtime + venv + Python 3.12** |
| `test_x86_ubuntu.Dockerfile` | Ubuntu 22.04 + Conda + Python 3.8 | **Ubuntu 24.04 + venv + Python 3.12** |

Key improvements:
- **Dropped Miniconda** in favor of system Python 3.12 + `python3 -m venv`
- **Conan 2.x** with `conan profile detect --force` for auto-configuration
- **No hardcoded Chinese mirrors** — removed aliyun/tsinghua mirror configs
- **Smaller images** — `--no-install-recommends`, `--no-cache-dir`, minimal test images
- **CUDA 12.6.3** base for GPU images (runtime variant for test)

#### Updated multimodal/Dockerfile

| Item | Before | After |
|---|---|---|
| DashInfer version | `v2.0.0-rc2` (`dashinfer-2.0.0rc2`) | **`v3.0.0` (`dashinfer-3.0.0`)** |
| pip install flags | none | `--no-cache-dir` |
| aliyun mirror | hardcoded `--index-url` | **removed** |

#### Updated CI Workflows (4 files)

| Workflow | Before | After |
|---|---|---|
| `build-check.yml` | UBI8 image, conda activation | **Ubuntu CUDA image, venv activation** |
| `build-check-share-runner.yml` | CentOS 7 image, conda, `actions/checkout@v3` | **Ubuntu CUDA image, venv, `actions/checkout@v4`** |
| `release_packages_cuda_only.yml` | UBI8 image, conda activation | **Ubuntu CUDA image, venv activation** |
| `release_packages_all.yml` | UBI8 + CentOS ARM images, ARM64 matrix | **Ubuntu CUDA image only, X64 only** |

> **Note**: ARM64 CI builds removed since ARM dev Dockerfiles were deleted.
> ARM wheel builds still use `release_aarch64_manylinux2.Dockerfile`.

### 2.3 Remaining Docker Files

After cleanup, the remaining Dockerfiles are:

| Dockerfile | Purpose | Base |
|---|---|---|
| `dev_ubuntu_cuda.Dockerfile` | CUDA dev environment | `nvidia/cuda:12.6.3-devel-ubuntu24.04` |
| `dev_x86_ubuntu.Dockerfile` | CPU dev environment | `ubuntu:24.04` |
| `test_cuda_ubuntu.Dockerfile` | CUDA test runner | `nvidia/cuda:12.6.3-runtime-ubuntu24.04` |
| `test_x86_ubuntu.Dockerfile` | CPU test runner | `ubuntu:24.04` |
| `release_x86_manylinux2.Dockerfile` | x86 wheel builds | `quay.io/pypa/manylinux2014_x86_64` |
| `release_aarch64_manylinux2.Dockerfile` | ARM64 wheel builds | `quay.io/pypa/manylinux_2_28_aarch64` |
| `multimodal/Dockerfile` | VLM container | `nvcr.io/nvidia/pytorch:24.10-py3` |

**Total: 7 Dockerfiles** (down from 13)

---

## 3. Remaining Upgrade Plan

### 3.1 Phase 3: Image Consolidation (Nice to Have)

#### 3.1.1 Reduce Dockerfile Count Further

| | |
|---|---|
| **Priority** | P2 |
| **Effort** | High |
| **Risk** | Medium |

Consider consolidating dev + test Dockerfiles using multi-stage builds and build arguments:

```
Dockerfile           --build-arg VARIANT={dev,test}  --build-arg PLATFORM={cuda,x86}
Dockerfile.release   --build-arg ARCH={x86_64,aarch64}
```

This would reduce from 6 to 2 Dockerfiles (+ multimodal).

Reference: SGLang uses a single multi-stage Dockerfile with build arguments.

#### 3.1.2 Docker Image CI/CD

Add automated Docker image builds to CI, triggered by changes to Dockerfiles or
dependency files. Tag images with `v{version}` and `latest`.

#### 3.1.3 Multi-Python Wheel Builds

Set up matrix builds for Python 3.10, 3.11, 3.12 to publish all wheels to PyPI.

#### 3.1.4 ManyLinux Dockerfile Modernization

The manylinux Dockerfiles still use:
- `ARG PY_VER=3.8` (should be 3.10+)
- Old Miniconda installers
- No Conan (OK for release-only builds)

Consider updating these in a future pass.

---

### 3.2 Phase 4: Flash Attention & CUTLASS Upgrade

#### 3.2.1 Current State

| Component | Current Version | Source | Notes |
|---|---|---|---|
| Flash Attention | v2.6.1 (`7551202`) | `Dao-AILab/flash-attention` git | SM80+, head dims 128/192 |
| CUTLASS | 3.5.0 | `third_party/cutlass_3.5.0.tgz` | Used by FA and FlashMLA |
| FlashMLA | main branch | `deepseek-ai/FlashMLA` git | SM90+ only, disabled by default |

#### 3.2.2 Upgrade to Flash Attention 3 (Hopper)

| | |
|---|---|
| **Priority** | P1 |
| **Effort** | High |
| **Risk** | Medium — new API surface, Hopper-only |
| **Depends on** | CUTLASS upgrade (3.2.4) |

**What**: Integrate [Flash Attention 3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)
for Hopper (SM90a) GPUs.

**Why**: FA3 achieves up to 1.5-2x speedup over FA2 on H100/H200.

#### 3.2.3 Upgrade to Flash Attention 4 (Blackwell)

| | |
|---|---|
| **Priority** | P2 |
| **Effort** | High |
| **Risk** | High — SM100 experimental |

#### 3.2.4 CUTLASS Upgrade

| | |
|---|---|
| **Priority** | P1 |
| **Effort** | Medium |
| **Risk** | Medium |

Upgrade CUTLASS from 3.5.0 to latest stable (3.7+).

#### 3.2.5 Migration Strategy

```
Step 1: CUTLASS 3.5.0 → 3.7+                    [prerequisite]
Step 2: Add FA3 alongside FA2 (runtime dispatch)  [SM90a, Hopper]
Step 3: Validate FA3 + MLA + FP8 integration       [DeepSeek V3 on H100]
Step 4: Add FA4 when stable                        [SM100, Blackwell]
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
Completed ✓
├── Docker: Delete CentOS 7/8, UBI 8, ALinux Dockerfiles (6 files)
├── Docker: Upgrade Ubuntu Dockerfiles to 24.04 + CUDA 12.6 + Python 3.12 + Conan 2.x
├── Docker: Update multimodal/Dockerfile dashinfer v2.0.0-rc2 → v3.0.0
├── Docker: Remove hardcoded Chinese mirrors
└── CI: Update all 4 workflows to use new Ubuntu image + venv activation

Phase 3 — Consolidation (target: v3.2+)
├── 3.1.1  Consolidate 6 Dockerfiles into 2 parameterized files       [P2, High effort]
├── 3.1.2  Add Docker image CI/CD                                      [P2, Medium effort]
├── 3.1.3  Multi-Python wheel build matrix                             [P2, Medium effort]
└── 3.1.4  Update manylinux Dockerfiles (PY_VER, conda)               [P2, Low effort]

Phase 4 — Flash Attention & CUTLASS Upgrade (target: v3.1 ~ v3.2)
├── 3.2.4  CUTLASS 3.5.0 → 3.7+                                       [P1, Medium effort]
├── 3.2.2  Flash Attention 3 (Hopper, SM90a)                           [P1, High effort]
├── 3.2.3  Flash Attention 4 (Blackwell, SM100)                        [P2, High effort]
└── 3.2.5  Runtime FA2/FA3/FA4 dispatch                                [P1, Medium effort]
```

## 6. Validation Checklist

Before each phase is considered complete:

- [x] Obsolete Dockerfiles deleted (CentOS 7/8, UBI 8, ALinux)
- [x] Ubuntu Dockerfiles upgraded to 24.04 + CUDA 12.6 + Python 3.12 + Conan 2.x
- [x] CI workflows updated to use new images
- [x] multimodal/Dockerfile updated to dashinfer v3.0.0
- [ ] New Docker images built and pushed to registry
- [ ] `conan install` completes with Conan 2.x in all images
- [ ] Python wheel builds for 3.10, 3.11, 3.12
- [ ] C++ package builds on all platforms (CUDA, x86)
- [ ] Unit tests pass (`cpp_interface_test`, `cpp_operator_test`, `cpp_kernel_test`)
- [ ] Python integration tests pass (`run_full_test.sh`)
- [ ] Docker registry images tagged and pushed with new version tags
