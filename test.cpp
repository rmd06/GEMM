/*
*
*      Filename: test.cpp
*
*        Author: Haibo Hao
*        Email : haohaibo@ncic.ac.cn
*   Description: ---
*        Create: 2017-11-10 10:48:35
* Last Modified: 2017-11-15 18:46:51
**/
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

//OpenCL
#include <CL/cl.h>

#include "err_code.h"

#define M 512
#define N 512
#define K 512

// tolerance used in floating point comparisons
#define TOL (0.001)

// return time since some fixed past point 
extern double wtime();

extern int output_device_info(cl_device_id);
const char *KernelSource = "\n" \
                            "__kernel void vadd(                                                 \n" \
                            "   __global float* a,                                                  \n" \
                            "   __global float* b,                                                  \n" \
                            "   __global float* c,                                                  \n" \
                            "   const unsigned int count)                                           \n" \
                            "{                                                                      \n" \
                            "   int i = get_global_id(0);                                           \n" \
                            "   if(i < count)                                                       \n" \
                            "       c[i] = a[i] + b[i];                                             \n" \
                            "}                                                                      \n" \
                            "\n";

//------------------------------------------------------------------------------
//

int main(){
    //char buffer[1000];
    std::ifstream infile("abc_alpha_beta.cl");
    //std::ifstream infile("vadd.cl");
    if(!infile.is_open()){
        std::cout << "Error opening file" << std::endl;
        exit(1);
    }

    std::string kernel_string;
    //while(infile){
    //    kernel_string.push_back(infile.get());
    //}
    char ch;
    infile >> std::noskipws;
    while(!infile.eof()){
        infile >> ch;
        kernel_string.push_back(ch);
    }

    //std::cout << kernel_string << std::endl;
    const char* kernel_source = kernel_string.c_str(); 


    /*while(!infile.eof()){
        infile.getline(buffer,1000);
        std::cout << buffer << std::endl;
    }
    */
    

    
    // error code returned from OpenCL calls
    int    err;
    float* h_a = (float*) calloc(M*K, sizeof(float));   // matrix A
    float* h_b = (float*) calloc(K*N, sizeof(float));   // matrix B
    float* h_c = (float*) calloc(M*N, sizeof(float));   // matrix C
    
    // number of correct results
    unsigned int correct;

    
    cl_device_id        device_id;  // compute device id
    cl_context          context;    // compute context
    cl_command_queue    commands;   // compute command queue
    cl_program          program;    // compute program
    cl_kernel           ko_gemm;    // compute kernel

    // device memory used for the input a matrix
    cl_mem d_a;
    const unsigned a_offset=0;
    // device memory used for the input b matrix
    cl_mem d_b;
    const unsigned b_offset=0;
    // device memory used for the output c matrix
    cl_mem d_c;
    const unsigned c_offset=0;
    const float alpha=1;
    const float beta=1;


    // Fill matrix A and B with random float values
    int i=0;
    int j=0;
    for(i=0;i<M;++i){
        for(j=0;j<K;++j){
            h_a[i*M + j] = rand() / (float)RAND_MAX;
        }
    }

    for(i=0;i<K;++i){
        for(j=0;j<N;++j){
            h_b[i*K + j] = rand() / (float)RAND_MAX;
        }
    }


    // Set up platform and GPU device
    cl_uint numPlatforms;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if(numPlatforms==0){
        std::cout << "Found 0 platforms!" << std::endl;
    }
    
    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");
    
    // Secure a GPU
    for(i=0;i<numPlatforms;++i){
        err = clGetDeviceIDs(Platform[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if(err == CL_SUCCESS){
            break;
        }
    }

    if(device_id==NULL){
        checkError(err, "Finding a device");
    }

    err = output_device_info(device_id);
    checkError(err, "Printing device output");

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)& kernel_source, NULL, &err); 
    //program = clCreateProgramWithSource(context, 1, (const char **)& KernelSource, NULL, &err); 
    checkError(err, "Creating program");

    // Build the program
    //err = clBuildProgram(program, 1, NULL, NULL, NULL, NULL); 
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL); 

    if(err != CL_SUCCESS){
        size_t len;
        char buffer[2048];

        std::cout << "Error: Failed to build program executable!\n" 
            << err_code(err) << std::endl; 

        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, &len);
        std::cout << buffer << std::endl;
        return EXIT_FAILURE;
    }


    // Create the compute kernel from the program
    ko_gemm = clCreateKernel(program, "tg_betac_alphaab", &err); 
    //ko_gemm = clCreateKernel(program, "vadd", &err); 
    checkError(err, "Creating kernel");

    // Create the input (a,b) and output (c) matrix in device memory
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*M*K, NULL, &err);
    checkError(err, "Creating buffer d_a");

    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*K*N, NULL, &err);
    checkError(err, "Creating buffer d_b");

    d_c = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*M*N, NULL, &err);
    checkError(err, "Creating buffer d_c");


    // Write a and b matrix into compute device memory
    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float)*M*K, h_a, 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_a");

    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float)*K*N, h_b, 0, NULL, NULL);
    checkError(err, "Copying h_b to device at d_b");

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ko_gemm, 0, sizeof(cl_mem), &d_a); 
    err |= clSetKernelArg(ko_gemm, 1, sizeof(unsigned int), &a_offset); 
    err |= clSetKernelArg(ko_gemm, 2, sizeof(cl_mem), &d_b); 
    err |= clSetKernelArg(ko_gemm, 3, sizeof(unsigned int), &b_offset); 
    err |= clSetKernelArg(ko_gemm, 4, sizeof(cl_mem), &d_c); 
    err |= clSetKernelArg(ko_gemm, 5, sizeof(unsigned int), &c_offset); 
    err |= clSetKernelArg(ko_gemm, 6, sizeof(float), &alpha); 
    err |= clSetKernelArg(ko_gemm, 7, sizeof(float), &beta); 
    checkError(err, "Setting kernel arguments");

    double rtime = wtime();
    // Execute the kernel over the entire range of our
    // 2d input data set

    const size_t global[2] = {M,N};
    const size_t local[3] = {256,1,1};
    //err = clEnqueueNDRangeKernel(commands, ko_gemm, 2, NULL, global, NULL, 0, NULL, NULL);
    err = clEnqueueNDRangeKernel(commands, ko_gemm, 2, NULL, global, local, 0, NULL, NULL);
    //checkError(err, "Enqueueing kernel");

    // Wait for the commands to complete before
    // stopping the timer
    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");

    rtime = wtime() - rtime;
    std::cout << "\n The kernel ran in "
        << rtime 
        << " seconds\n" << std::endl;

    float gflops;
    gflops = 2.0*M*N*K/(1000000000.0f * rtime);
    std::cout << " **********************************" << std::endl;
    std::cout << " M= " << M
              << ",K= " << K
              << ",N= " << N
              << std::endl;
    std::cout << " Performance : " << gflops << " gflops" << std::endl;
    std::cout << " **********************************" << std::endl;

    
    // Read back the results from the compute device
    err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(float)*M*N, h_c, 0, NULL, NULL);
    if(err != CL_SUCCESS){
        std::cout << "Error: Failed to read output array!\n"
            << err_code(err) << std::endl; 
    }

    // Test the results
    correct = 0;
    float tmp;
    for(int i=0;i<M;++i){
        for(int j=0;j<N;++j){
            tmp=0;
            for(int k=0;k<K;++k){
                tmp+=h_a[i*K+k]*h_b[k*N+j]; 
            }
        }
    }

    // cleanup then shutdown
    clReleaseMemObject(d_a); 
    clReleaseMemObject(d_b); 
    clReleaseMemObject(d_c); 
    clReleaseProgram(program);
    clReleaseKernel(ko_gemm);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

