FROM ubuntu:22.04 AS base

ARG PREFIX=/usr/local

# Support multiarch
RUN dpkg --add-architecture i386

# Install rocm key
RUN apt-get update && apt-get install -y software-properties-common gnupg2 --no-install-recommends curl && \
    curl -sL http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -

# Add rocm repository
RUN sh -c 'echo deb [arch=amd64 trusted=yes] http://repo.radeon.com/rocm/apt/7.2/ jammy main > /etc/apt/sources.list.d/rocm.list'

# From docs.amd.com for installing rocm. Needed to install properly
RUN sh -c "echo 'Package: *\nPin: release o=repo.radeon.com\nPin-priority: 600' > /etc/apt/preferences.d/rocm-pin-600"

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    bison \
    build-essential \
    clang-14 \
    cmake \
    curl \
    flex \
    g++ \
    gdb \
    git \
    lcov \
    locales \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-full \
    libpython3.8 \
    wget \
    rocm-device-libs \
    hip-dev \
    libnuma-dev \
    miopen-hip \
    libomp-dev \
    rocblas \
    hipfft \
    hipsolver \
    rocthrust \
    rocrand \
    hipsparse \
    rccl \
    rocm-smi-lib \
    rocm-dev \
    roctracer-dev \
    hipcub  \
    hipblas  \
    hipify-clang \
    hiprand-dev \
    half \
    libssl-dev \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pytorch
RUN pip3 install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.7.1%2Brocm7.2.0.lw.git262e50d5-cp310-cp310-linux_x86_64.whl\
                 https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.22.1%2Brocm7.2.0.git59a3e1f9-cp310-cp310-linux_x86_64.whl\
                 https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.3.1%2Brocm7.2.0.git28a7371e-cp310-cp310-linux_x86_64.whl

# add this for roctracer dependancies
RUN pip3 install CppHeaderParser

# Workaround broken rocm packages
RUN ln -s /opt/rocm-* /opt/rocm
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
RUN echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/rocm-llvm.conf
RUN ldconfig

# Workaround broken miopen cmake files
RUN sed -i 's,;/usr/lib/x86_64-linux-gnu/librt.so,,g' /opt/rocm/lib/cmake/miopen/miopen-targets.cmake

# Workaround for distributions running cmake < 3.25
RUN sed -i -e 's/^block/if(COMMAND block)\nblock/g' -e 's/^endblock/endblock\(\)\nendif/g' /opt/rocm/lib/cmake/hipblaslt/hipblaslt-config.cmake

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install dependencies
ADD dev-requirements.txt /dev-requirements.txt
ADD requirements.txt /requirements.txt
ADD rbuild.ini /rbuild.ini

# Location where onnx unit tests models are cached
ENV ONNX_HOME=/.onnx
RUN mkdir -p $ONNX_HOME/models && chmod 777 $ONNX_HOME/models

COPY ./tools/install_prereqs.sh /
COPY ./tools/requirements-py.txt /requirements-py.txt
RUN /install_prereqs.sh /usr/local / && rm /install_prereqs.sh && rm /requirements-py.txt
RUN test -f /usr/local/hash || exit 1

# Install cmake
ARG CMAKE=3.31.10
RUN wget https://github.com/Kitware/CMake/releases/download/v$CMAKE/cmake-$CMAKE-Linux-x86_64.tar.gz && \
    tar -xzf cmake-$CMAKE-Linux-x86_64.tar.gz -C /opt && \
    rm cmake-$CMAKE-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-$CMAKE-linux-x86_64/bin:$PATH"

COPY ./test/onnx/.onnxrt-commit /

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=v1.26.0
ARG ONNXRUNTIME_COMMIT

RUN git clone --depth 1 --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    cd onnxruntime && \
    if [ -z "$ONNXRUNTIME_COMMIT" ] ; then git checkout $(cat /.onnxrt-commit) ; else git checkout ${ONNXRUNTIME_COMMIT} ; fi && \
    /bin/sh /onnxruntime/dockerfiles/scripts/install_common_deps.sh

ENV MIOPEN_FIND_DB_PATH=/tmp/miopen/find-db
ENV MIOPEN_USER_DB_PATH=/tmp/miopen/user-db
ENV LD_LIBRARY_PATH=$PREFIX/lib

# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1
# Disable odr detection since its broken with shared libraries
# See: https://github.com/google/sanitizers/issues/1017
ENV ASAN_OPTIONS=detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1
RUN ln -s /opt/rocm/llvm/bin/llvm-symbolizer /usr/bin/llvm-symbolizer

FROM base AS build

WORKDIR /code
RUN apt install -qq -y libomp-dev
RUN git clone --depth 1 --branch v1.8.1 https://github.com/uxlfoundation/oneDNN.git dnnl && \
    cd dnnl && mkdir build && mkdir out && cd build && \
    CXX=/opt/rocm/llvm/bin/clang++ CC=/opt/rocm/llvm/bin/clang cmake .. \
        -DCMAKE_INSTALL_PREFIX=/code/dnnl/out \
        -DDNNL_BUILD_EXAMPLES=OFF \
        -DNNL_BUILD_TESTS=OFF \
        -DDNNL_CPU_RUNTIME=OMP \
        -DDNNL_ARCH_OPT_FLAGS="-march=znver2" \
        -DONEDNN_BUILD_GRAPH=ON && \
    make -j && \
    make install

WORKDIR /code
RUN git clone --recursive --branch cpp-3.3.0 https://github.com/msgpack/msgpack-c.git msgpack && \
    cd msgpack && mkdir build && mkdir out && cd build && \
    CXX=/opt/rocm/llvm/bin/clang++ CC=/opt/rocm/llvm/bin/clang cmake .. \
        -DCMAKE_INSTALL_PREFIX=/code/msgpack/out \
        -DMSGPACK_BUILD_EXAMPLES=OFF \
        -DMSGPACK_BUILD_TESTS=OFF && \
    make -j && \
    make install

WORKDIR /code/AMDMIGraphX
COPY . .
RUN mkdir build && mkdir out && cd build && \
    CXX=/opt/rocm/llvm/bin/clang++ CC=/opt/rocm/llvm/bin/clang cmake .. \
        -DCMAKE_INSTALL_PREFIX=/code/AMDMIGraphX/out \
        -DCMAKE_BUILD_TYPE=release \
        -DGPU_TARGETS="gfx900;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1200;gfx1201" \
        -DBUILD_TESTING=OFF \
        -DMIGRAPHX_ENABLE_CPU=ON \
        -DMIGRAPHX_USE_COMPOSABLEKERNEL=ON && \
    make -j && \
    make install

# Include DNNL libraries
RUN apt install -qq -y rsync && \
    rsync -avPH /code/dnnl/out/ /code/AMDMIGraphX/out/

WORKDIR /
RUN tar -czvf out.tar.gz -C /code/AMDMIGraphX/out/ .

FROM scratch
COPY --from=build /out.tar.gz /
