# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HPC-Ops is a production-grade, high-performance operator library for LLM inference developed by the Tencent Hunyuan AI Infra team. It provides optimized CUDA kernels for NVIDIA H20 GPUs (SM90 architecture) with up to 2.22x performance improvement over baselines like FlashInfer, FA2, FA3, and TensorRT-LLM.

## Architecture

**Main Components**:
- `hpc/` - Python package with clean API for operators (attention, group_gemm, fuse_moe, act)
- `src/` - C++/CUDA kernel implementations
  - `src/attention/` - Decode and prefill attention kernels
  - `src/fuse_moe/` - Fused MoE kernels
  - `src/group_gemm/` - Grouped GEMM kernels
  - `src/activation/` - Activation function kernels
  - `src/utils/` - CUDA utilities (TMA, vector operations)
- `tests/` - pytest test files with reference implementations

**Key Technologies**:
- CuTe and CUTLASS for modern CUDA kernel development
- PyTorch C++ extension for Python bindings
- Targets NVIDIA SM90 architecture (H20 GPUs)

## Common Commands

### Build
```bash
make all              # Build the project
make wheel           # Build wheel package
make clean           # Clean build artifacts
```

### Formatting & Linting
```bash
make format          # Format all files (black + clang-format)
make format-check    # Check if formatting passes
```

### Testing
```bash
make test            # Run all tests
make sanitizer       # Run tests with compute-sanitizer

# Run single test file
pytest tests/test_group_gemm_pertensor.py -v

# Run specific test function
pytest tests/test_group_gemm_pertensor.py::test_group_gemm_pertensor_fp8 -v
```

## Requirements

- NVIDIA SM90 architecture GPU (H20)
- CUDA Toolkit 12.8+
- Python 3.8+
- PyTorch 2.7.0+
- C++17 compatible compiler

## Configuration Files

- `CMakeLists.txt` - CMake configuration for CUDA compilation
- `setup.py` - Python package configuration
- `requirements-dev.txt` - Development dependencies
- `.clang-format` - C++/CUDA formatting (LLVM style)
- `CPPLINT.cfg` - C++ linting (Google style)
