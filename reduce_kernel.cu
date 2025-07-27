#include <cuda_runtime.h>
#include <iostream>

// Kernel: 块内使用共享内存进行归约求和
__global__ void reduce_sum_kernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[]; // 拷贝到共享内存的目的是为了加快访问速度

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 拷贝数据到共享内存
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    // 块内规约：对半加法
    for (int s = blockDim.x / 2; s > 0; s = s/2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // 必须要加同步
        __syncthreads();
    }

    // 每个 block 的第一个线程写回结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Host接口：多步规约完成总和
void reduce_sum(const float* h_input, float* h_output, int size) {
    float *d_input, *d_intermediate;
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    cudaMalloc(&d_input, sizeof(float) * size);
    cudaMemcpy(d_input, h_input, sizeof(float) * size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_intermediate, sizeof(float) * blocks);

    // 第一步规约
    reduce_sum_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
        d_input, d_intermediate, size);

    // 复制中间结果回 Host 并求最终和
    float* h_intermediate = new float[blocks];
    cudaMemcpy(h_intermediate, d_intermediate, sizeof(float) * blocks, cudaMemcpyDeviceToHost);

    float final_sum = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        final_sum += h_intermediate[i];
    }

    *h_output = final_sum;

    delete[] h_intermediate;
    cudaFree(d_input);
    cudaFree(d_intermediate);
}

// 测试用例
int main() {
    const int size = 1024;
    float h_input[size];
    for (int i = 0; i < size; ++i) h_input[i] = 1.0f; // 所有元素设为1，预期总和为1024

    float h_output = 0.0f;
    reduce_sum(h_input, &h_output, size);

    std::cout << "Reduce sum result: " << h_output << std::endl;
    return 0;
}
