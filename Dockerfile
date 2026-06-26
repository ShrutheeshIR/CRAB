FROM ubuntu:noble

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# Core system packages
# ------------------------------------------------------------
# • Build tools (gcc, g++, clang, llvm, cmake, ninja, make)
# • Development headers for Clang (libclang-dev) – fixes the
#   missing `clang/Frontend/CompilerInstance.h` error.
# • Testing & static‑analysis tools (gtest, clang‑tidy, cppcheck)
# • OpenGL/ffmpeg runtime (needed for MeshCat / OpenCV)
# ------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        clang clangd llvm llvm-dev libllvm-15-ocaml-dev libllvm15 clang-format libclang-dev \
        gcc g++ \
        cmake make ninja-build git wget curl ca-certificates \
        python3 python3-pip python3-venv \
        libgtest-dev clang-tidy cppcheck \
        libgl1 libglx-mesa0 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python packages – CPU‑only PyTorch wheels
# ------------------------------------------------------------
# The `-f https://download.pytorch.org/whl/cpu/torch_stable.html`
# flag points pip to the CPU‑only wheel index, eliminating any
# CUDA/NVIDIA dependencies.
# ------------------------------------------------------------

RUN pip3 install --no-cache-dir --break-system-packages \
        symforce \
        urdfpy \
        matplotlib seaborn plotly pandas \
        meshcat \
        pytest pytest-cov

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages

WORKDIR /workspace

CMD ["/bin/bash"]
