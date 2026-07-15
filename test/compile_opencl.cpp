// OpenNR — offline compile check for the OpenCL kernel source.
//
// Resolve builds the OpenCL program at plugin load; a syntax error would
// otherwise only appear on users' machines. This harness extracts nothing —
// pass the kernel source file (extracted from OpenCLKernel.cpp with awk, see
// CLAUDE.md) as argv[1] and it builds it against the first OpenCL device.
//
// Build: c++ -O2 -std=c++14 compile_opencl.cpp -framework OpenCL -o compile_opencl
// Run:   awk '/R"CLC\(/{flag=1;next} /^\)CLC";/{flag=0} flag' \
//            ../plugin/OpenCLKernel.cpp > /tmp/opennr.cl && ./compile_opencl /tmp/opennr.cl

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <kernel.cl>\n", argv[0]);
        return 2;
    }
    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::string src;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        src.append(buf, n);
    fclose(f);

    cl_platform_id platform = nullptr;
    if (clGetPlatformIDs(1, &platform, nullptr) != CL_SUCCESS || !platform) {
        printf("no OpenCL platform — skipping compile check\n");
        return 0;
    }
    cl_device_id device = nullptr;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS) {
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr) != CL_SUCCESS || !device) {
            printf("no OpenCL device — skipping compile check\n");
            return 0;
        }
    }
    char devName[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(devName) - 1, devName, nullptr);

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext failed [%d]\n", err);
        return 1;
    }
    const char* srcPtr = src.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &srcPtr, nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource failed [%d]\n", err);
        return 1;
    }
    err = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::vector<char> log(1 << 20, 0);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log.size() - 1, log.data(), nullptr);
        fprintf(stderr, "OPENCL BUILD FAILED on %s:\n%s\n", devName, log.data());
        return 1;
    }
    printf("OPENCL COMPILE OK on %s\n", devName);
    return 0;
}
