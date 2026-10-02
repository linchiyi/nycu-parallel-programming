#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define BLOCK_SIZE 16
#define GROUP_SIZE 4  // Each thread processes GROUP_SIZE x GROUP_SIZE pixels

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
                               int res_x, int res_y, size_t pitch, int max_iterations, int *output)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    // Each thread handles GROUP_SIZE x GROUP_SIZE pixels
    int baseX = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE;
    int baseY = (blockIdx.y * blockDim.y + threadIdx.y) * GROUP_SIZE;

    // Process GROUP_SIZE x GROUP_SIZE pixels
    for (int dy = 0; dy < GROUP_SIZE; dy++)
    {
        int thisY = baseY + dy;
        if (thisY >= res_y)
            continue;

        int *row = (int *)((char *)output + thisY * pitch);

        for (int dx = 0; dx < GROUP_SIZE; dx++)
        {
            int thisX = baseX + dx;
            if (thisX >= res_x)
                continue;

            float x = lower_x + thisX * step_x;
            float y = lower_y + thisY * step_y;

            row[thisX] = mandel(x, y, max_iterations);
        }
    }
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

    // Allocate pinned host memory using cudaHostAlloc
    int *host_out;
    cudaHostAlloc((void **)&host_out, size, cudaHostAllocDefault);

    // Allocate pitched device memory using cudaMallocPitch
    int *device_out;
    size_t pitch;
    cudaMallocPitch((void **)&device_out, &pitch, res_x * sizeof(int), res_y);

    // Setup execution configuration
    // Each thread processes GROUP_SIZE x GROUP_SIZE pixels
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((res_x + BLOCK_SIZE * GROUP_SIZE - 1) / (BLOCK_SIZE * GROUP_SIZE), 
                   (res_y + BLOCK_SIZE * GROUP_SIZE - 1) / (BLOCK_SIZE * GROUP_SIZE));

    // Launch kernel
    mandel_kernel<<<numBlocks, threadsPerBlock>>>(lower_x, lower_y, step_x, step_y, 
                                                   res_x, res_y, pitch, max_iterations, device_out);

    // Copy result back to host using cudaMemcpy2D
    cudaMemcpy2D(host_out, res_x * sizeof(int), device_out, pitch, 
                 res_x * sizeof(int), res_y, cudaMemcpyDeviceToHost);
    
    // Copy to output image
    memcpy(img, host_out, size);

    // Free memory
    cudaFree(device_out);
    cudaFreeHost(host_out);
}
