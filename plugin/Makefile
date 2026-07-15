# OpenNR — OpenFX plugin build
#
#   macOS:  make            (universal arm64 + x86_64, Metal + OpenCL)
#   Linux:  make            (CUDA + OpenCL; needs nvcc)
#   Install locally: sudo make install

UNAME_SYSTEM := $(shell uname -s)

CXXFLAGS = -O3 -std=c++14 -fvisibility=hidden -I../ofx/include -I../ofx/Support/include

ifeq ($(UNAME_SYSTEM), Linux)
	AMDAPP_PATH ?= /opt/AMDAPP
	CXXFLAGS += -I${AMDAPP_PATH}/include -fPIC
	CUDAPATH ?= /usr/local/cuda
	NVCC = ${CUDAPATH}/bin/nvcc
	NVCCFLAGS = --compiler-options="-fPIC" -I.
	LDFLAGS = -shared -fvisibility=hidden -L${CUDAPATH}/lib64 -lcuda -lcudart_static
	BUNDLE_DIR = OpenNR.ofx.bundle/Contents/Linux-x86-64/
	GPU_OBJ = CudaKernel.o OpenCLKernel.o
else
	# Deployment target matters: without it the binary inherits the build
	# SDK's minimum OS and silently refuses to load on older Macs (Resolve
	# then logs "plugin not supported on this platform, is being skipped").
	MACOS_MIN = -mmacosx-version-min=11.0
	ARCH_FLAGS = -arch arm64 -arch x86_64 ${MACOS_MIN}
	CXXFLAGS += ${ARCH_FLAGS}
	LDFLAGS = -bundle -fvisibility=hidden -F/Library/Frameworks -framework OpenCL -framework Metal -framework AppKit
	LDFLAGS += ${ARCH_FLAGS}
	BUNDLE_DIR = OpenNR.ofx.bundle/Contents/MacOS/
	GPU_OBJ = MetalKernel.o OpenCLKernel.o
endif

OpenNR.ofx: OpenNRPlugin.o ${GPU_OBJ} ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp OpenNR.ofx $(BUNDLE_DIR)
	cp Info.plist OpenNR.ofx.bundle/Contents/ 2>/dev/null || true

CudaKernel.o: CudaKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

MetalKernel.o: MetalKernel.mm
	$(CXX) -c $< $(CXXFLAGS)

%.o: ../ofx/Support/Library/%.cpp
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f *.o *.ofx
	rm -fr OpenNR.ofx.bundle

install: OpenNR.ofx
	mkdir -p /Library/OFX/Plugins
	cp -fr OpenNR.ofx.bundle /Library/OFX/Plugins
