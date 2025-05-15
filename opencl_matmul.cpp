
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>
#include <string>
#include <CL/cl.h>

// OpenCL kernel source for matrix multiplication
const char* matrixMultiplyKernel = R"(
__kernel void matrixMultiply(__global const float* A, 
                             __global const float* B, 
                             __global float* C, 
                             int M, int N, int K) {
    // Get global position in grid
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}
)";

// Function to initialize matrix with random values
void initializeMatrix(std::vector<float>& matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dist(gen);
    }
}

// CPU matrix multiplication for reference
double matrixMultiplyCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, 
                       int m, int n, int k) {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    return duration.count();
}

// Check OpenCL error
void checkOpenCLError(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << msg << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Get OpenCL platform and device info
void getOpenCLDeviceInfo() {
    cl_int err;
    cl_uint num_platforms;
    
    // Get platform count
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    checkOpenCLError(err, "Failed to get platform count");
    
    // Get all platforms
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    checkOpenCLError(err, "Failed to get platform IDs");
    
    for (cl_uint i = 0; i < num_platforms; i++) {
        // Get platform name
        char platform_name[128];
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        checkOpenCLError(err, "Failed to get platform name");
        
        std::cout << "Platform " << i << ": " << platform_name << std::endl;
        
        // Get device count for this platform
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        
        if (err != CL_SUCCESS) {
            std::cout << "  No devices found for this platform\n";
            continue;
        }
        
        // Get all devices for this platform
        std::vector<cl_device_id> devices(num_devices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices.data(), NULL);
        checkOpenCLError(err, "Failed to get device IDs");
        
        for (cl_uint j = 0; j < num_devices; j++) {
            char device_name[128];
            char device_vendor[128];
            cl_device_type device_type;
            cl_uint compute_units;
            size_t max_work_group_size;
            size_t max_work_item_dims;
            std::vector<size_t> max_work_item_sizes(3);
            
            // Get device info
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dims), &max_work_item_dims, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, max_work_item_sizes.data(), NULL);
            
            // Print device info
            std::cout << "  Device " << j << ": " << device_name << " (" << device_vendor << ")" << std::endl;
            std::cout << "    Type: ";
            if (device_type & CL_DEVICE_TYPE_CPU) std::cout << "CPU ";
            if (device_type & CL_DEVICE_TYPE_GPU) std::cout << "GPU ";
            if (device_type & CL_DEVICE_TYPE_ACCELERATOR) std::cout << "Accelerator ";
            std::cout << std::endl;
            
            std::cout << "    Compute Units: " << compute_units << std::endl;
            std::cout << "    Max Work Group Size: " << max_work_group_size << std::endl;
            std::cout << "    Max Work Item Dimensions: " << max_work_item_dims << std::endl;
            std::cout << "    Max Work Item Sizes: [" 
                      << max_work_item_sizes[0] << ", " 
                      << max_work_item_sizes[1] << ", " 
                      << max_work_item_sizes[2] << "]" << std::endl;
        }
    }
}

// OpenCL matrix multiplication with specific workgroup size
double matrixMultiplyOpenCL(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, 
                          int m, int n, int k, int workgroup_size_x, int workgroup_size_y) {
    cl_int err;
    
    // Get a platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    checkOpenCLError(err, "Failed to get platform ID");
    
    // Get a device (prefer GPU, fall back to CPU)
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "GPU device not found, trying CPU..." << std::endl;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        checkOpenCLError(err, "Failed to get device ID");
    }
    
    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkOpenCLError(err, "Failed to create context");
    
    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkOpenCLError(err, "Failed to create command queue");
    
    // Create memory buffers
    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), NULL, &err);
    checkOpenCLError(err, "Failed to create buffer A");
    
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * k * sizeof(float), NULL, &err);
    checkOpenCLError(err, "Failed to create buffer B");
    
    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, m * k * sizeof(float), NULL, &err);
    checkOpenCLError(err, "Failed to create buffer C");
    
    // Copy matrices A and B to their respective memory buffers
    err = clEnqueueWriteBuffer(queue, a_mem, CL_TRUE, 0, m * n * sizeof(float), A.data(), 0, NULL, NULL);
    checkOpenCLError(err, "Failed to write to buffer A");
    
    err = clEnqueueWriteBuffer(queue, b_mem, CL_TRUE, 0, n * k * sizeof(float), B.data(), 0, NULL, NULL);
    checkOpenCLError(err, "Failed to write to buffer B");
    
    // Create and build program
    cl_program program = clCreateProgramWithSource(context, 1, &matrixMultiplyKernel, NULL, &err);
    checkOpenCLError(err, "Failed to create program");
    
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // If the build fails, get the build log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        
        std::cerr << "ERROR: Failed to build program: " << std::endl << build_log.data() << std::endl;
        
        // Clean up
        clReleaseProgram(program);
        clReleaseMemObject(a_mem);
        clReleaseMemObject(b_mem);
        clReleaseMemObject(c_mem);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        
        return -1.0;
    }
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "matrixMultiply", &err);
    checkOpenCLError(err, "Failed to create kernel");
    
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    checkOpenCLError(err, "Failed to set kernel arg 0");
    
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    checkOpenCLError(err, "Failed to set kernel arg 1");
    
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    checkOpenCLError(err, "Failed to set kernel arg 2");
    
    err = clSetKernelArg(kernel, 3, sizeof(int), &m);
    checkOpenCLError(err, "Failed to set kernel arg 3");
    
    err = clSetKernelArg(kernel, 4, sizeof(int), &n);
    checkOpenCLError(err, "Failed to set kernel arg 4");
    
    err = clSetKernelArg(kernel, 5, sizeof(int), &k);
    checkOpenCLError(err, "Failed to set kernel arg 5");
    
    // Set work sizes
    size_t global_work_size[2] = { (size_t)k, (size_t)m };
    size_t local_work_size[2] = { (size_t)workgroup_size_x, (size_t)workgroup_size_y };
    
    // Make sure global work size is a multiple of local work size
    global_work_size[0] = ((global_work_size[0] + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];
    global_work_size[1] = ((global_work_size[1] + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1];
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Execute kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: Failed to execute kernel: " << err << std::endl;
        
        // Clean up
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(a_mem);
        clReleaseMemObject(b_mem);
        clReleaseMemObject(c_mem);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        
        return -1.0;
    }
    
    // Wait for kernel to finish
    clFinish(queue);
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    // Read results back
    err = clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0, m * k * sizeof(float), C.data(), 0, NULL, NULL);
    checkOpenCLError(err, "Failed to read output buffer");
    
    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(a_mem);
    clReleaseMemObject(b_mem);
    clReleaseMemObject(c_mem);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return duration.count();
}

// Verify results between CPU and OpenCL
bool verifyResult(const std::vector<float>& cpuResult, const std::vector<float>& openclResult, float tolerance = 1e-5) {
    if (cpuResult.size() != openclResult.size()) {
        return false;
    }
    
    float diffSum = 0.0f;
    float maxDiff = 0.0f;
    int maxDiffIdx = -1;
    
    for (size_t i = 0; i < cpuResult.size(); i++) {
        float diff = std::abs(cpuResult[i] - openclResult[i]);
        diffSum += diff;
        
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = i;
        }
    }
    
    std::cout << "Verification - Diff sum: " << diffSum 
              << ", Max diff: " << maxDiff << " at index " << maxDiffIdx << std::endl;
    
    // Consider it verified if the average difference is small
    return (diffSum / cpuResult.size()) < tolerance;
}

int main() {
    // Print OpenCL device info
    getOpenCLDeviceInfo();
    
    // Define potential workgroup sizes to test
    std::vector<std::pair<int, int>> workgroupSizes = {
        {8, 8},    // 64 threads
        {8, 16},   // 128 threads
        {16, 8},   // 128 threads
        {16, 16},  // 256 threads
        {32, 4},   // 128 threads
        {32, 8}    // 256 threads
    };
    
    std::cout << "\nTesting different workgroup sizes on a 512x512 matrix:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::left << std::setw(15) << "Block Size" 
              << std::setw(20) << "Time (ms)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    // Test matrix size for workgroup optimization
    int testSize = 512;
    std::vector<float> A_test(testSize * testSize);
    std::vector<float> B_test(testSize * testSize);
    std::vector<float> C_test(testSize * testSize);
    
    // Initialize test matrices
    initializeMatrix(A_test, testSize * testSize);
    initializeMatrix(B_test, testSize * testSize);
    
    // Find the best workgroup size
    double bestTime = std::numeric_limits<double>::max();
    std::pair<int, int> bestWorkgroupSize = {16, 16}; // Default in case all tests fail
    
    for (const auto& size : workgroupSizes) {
        int workgroup_size_x = size.first;
        int workgroup_size_y = size.second;
        
        // Run the test with this workgroup size
        double time = matrixMultiplyOpenCL(
            A_test, B_test, C_test, testSize, testSize, testSize, 
            workgroup_size_x, workgroup_size_y);
        
        std::cout << std::left << std::setw(5) << workgroup_size_x << " x " 
                  << std::setw(5) << workgroup_size_y
                  << std::fixed << std::setprecision(2) 
                  << std::setw(20) << time << std::endl;
        
        // Update best workgroup size if this one is faster
        if (time > 0 && time < bestTime) {
            bestTime = time;
            bestWorkgroupSize = size;
        }
    }
    
    std::cout << "\nOptimal workgroup size: " << bestWorkgroupSize.first << "x" 
              << bestWorkgroupSize.second << " with " << bestTime << " ms" << std::endl;
    
    // Now run the full benchmark with the optimal workgroup size
    std::vector<int> sizes = {256, 512, 1024, 2048};
    
    std::cout << "\nMatrix Multiplication Benchmark (Workgroup: " 
              << bestWorkgroupSize.first << "x" << bestWorkgroupSize.second << ")" << std::endl;
    std::cout << std::string(75, '=') << std::endl;
    std::cout << std::left << std::setw(10) << "Size" 
              << std::setw(20) << "CPU Time (ms)" 
              << std::setw(20) << "OpenCL Time (ms)" 
              << std::setw(15) << "Speedup" 
              << std::setw(10) << "Verified" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    // Prepare results container
    std::vector<double> cpu_times;
    std::vector<double> opencl_times;
    std::vector<double> speedups;
    
    // Test each matrix size
    for (int size : sizes) {
        std::vector<float> A(size * size);
        std::vector<float> B(size * size);
        std::vector<float> C_cpu(size * size);
        std::vector<float> C_opencl(size * size);
        
        // Initialize matrices
        initializeMatrix(A, size * size);
        initializeMatrix(B, size * size);
        
        // Run CPU version
        double cpu_time = matrixMultiplyCPU(A, B, C_cpu, size, size, size);
        cpu_times.push_back(cpu_time);
        
        // Run OpenCL version with optimal workgroup size
        double opencl_time = matrixMultiplyOpenCL(
            A, B, C_opencl, size, size, size, 
            bestWorkgroupSize.first, bestWorkgroupSize.second);
        opencl_times.push_back(opencl_time);
        
        // Calculate speedup
        double speedup = cpu_time / opencl_time;
        speedups.push_back(speedup);
        
        // Verify results
        bool verified = (opencl_time > 0) ? verifyResult(C_cpu, C_opencl) : false;
        
        // Print results
        std::cout << std::left << std::setw(10) << size 
                  << std::fixed << std::setprecision(2) 
                  << std::setw(20) << cpu_time 
                  << std::setw(20) << opencl_time 
                  << std::setw(15) << speedup 
                  << std::setw(10) << (verified ? "Yes" : "No") << std::endl;
    }
    
    // Save results to CSV for plotting
    std::ofstream outfile("opencl_results.csv");
    outfile << "Size,CPU,OpenCL,Speedup,WorkgroupSizeX,WorkgroupSizeY" << std::endl;
    
    for (size_t i = 0; i < sizes.size(); i++) {
        outfile << sizes[i] << "," 
                << cpu_times[i] << "," 
                << opencl_times[i] << "," 
                << speedups[i] << ","
                << bestWorkgroupSize.first << ","
                << bestWorkgroupSize.second << std::endl;
    }
    
    outfile.close();
    std::cout << "\nResults saved to opencl_results.csv" << std::endl;
    
    // Get device properties for justification
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    
    size_t max_work_group_size;
    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    
    // Provide workgroup size justification
    std::cout << "\nWorkgroup size justification:" << std::endl;
    std::cout << "1. Optimal workgroup size determined experimentally: " 
              << bestWorkgroupSize.first << "x" << bestWorkgroupSize.second << std::endl;
    std::cout << "2. This configuration balances thread utilization and memory access patterns" << std::endl;
    std::cout << "3. Total thread count (" << bestWorkgroupSize.first * bestWorkgroupSize.second 
              << ") is appropriate for the device's compute units (" << compute_units << ")" << std::endl;
    std::cout << "4. Experimentally proven to be the fastest configuration for our problem" << std::endl;
    std::cout << "5. Well within the hardware limits of " << max_work_group_size << " threads per workgroup" << std::endl;
    
    return 0;
}

