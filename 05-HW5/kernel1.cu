#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define BLOCK_SIZE 16

// Device function for mandelbrot calculation
__device__ int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

__global__ void mandel_kernel(float lower_x, float lower_y, float step_x, float step_y, 
                               int res_x, int res_y, int max_iterations, int *output)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX >= res_x || thisY >= res_y)
        return;

    float x = lower_x + thisX * step_x;
    float y = lower_y + thisY * step_y;

    int index = thisY * res_x + thisX;
    output[index] = mandel(x, y, max_iterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int *img,
             int res_x,
             int res_y,
             int max_iterations)
{
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    int size = res_x * res_y * sizeof(int);

    // Allocate host memory using new
    int *host_out = new int[res_x * res_y];

    // Allocate device memory using cudaMalloc
    int *device_out;
    cudaMalloc((void **)&device_out, size);

    // Setup execution configuration
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((res_x + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (res_y + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    mandel_kernel<<<numBlocks, threadsPerBlock>>>(lower_x, lower_y, step_x, step_y, 
                                                   res_x, res_y, max_iterations, device_out);

    // Copy result back to host
    cudaMemcpy(host_out, device_out, size, cudaMemcpyDeviceToHost);
    
    // Copy to output image
    memcpy(img, host_out, size);

    // Free memory
    cudaFree(device_out);
    delete[] host_out;
}
