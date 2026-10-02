#include <stdio.h>
#include <stdlib.h>
#include "host_fe.h"
#include "helper.h"

void host_fe(int filter_width, float *filter, int image_height, int image_width,
             float *input_image, float *output_image, cl_device_id *device,
             cl_context *context, cl_program *program)
{
    cl_int status;
    static cl_command_queue queue = NULL;
    static cl_kernel kernel = NULL;
    static cl_mem input_mem = NULL;
    static cl_mem filter_mem = NULL;
    static cl_mem output_mem = NULL;
    static int prev_img_w = 0;
    static int prev_img_h = 0;
    static int prev_filter_w = 0;

    // Check if we need to re-initialize (first run or parameters changed)
    if (queue == NULL || prev_img_w != image_width || prev_img_h != image_height || prev_filter_w != filter_width) {
        
        if (queue) {
            clReleaseMemObject(input_mem);
            clReleaseMemObject(filter_mem);
            clReleaseMemObject(output_mem);
            clReleaseKernel(kernel);
            clReleaseCommandQueue(queue);
        }

        // Create Command Queue
        queue = clCreateCommandQueue(*context, *device, 0, &status);
        CHECK(status, "clCreateCommandQueue");

        size_t image_size = image_height * image_width * sizeof(float);
        size_t filter_size = filter_width * filter_width * sizeof(float);

        // Input: Use Host Ptr (Zero Copy)
        input_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                   image_size, input_image, &status);
        CHECK(status, "clCreateBuffer input");

        // Filter: Copy Host Ptr (Small, Constant)
        filter_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    filter_size, filter, &status);
        CHECK(status, "clCreateBuffer filter");

        // Output: Use Host Ptr (Zero Copy)
        output_mem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                    image_size, output_image, &status);
        CHECK(status, "clCreateBuffer output");

        // Create Kernel
        kernel = clCreateKernel(*program, "convolution", &status);
        CHECK(status, "clCreateKernel");

        // Set Arguments
        status = clSetKernelArg(kernel, 0, sizeof(int), &filter_width);
        CHECK(status, "clSetKernelArg 0");
        status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &filter_mem);
        CHECK(status, "clSetKernelArg 1");
        status = clSetKernelArg(kernel, 2, sizeof(int), &image_height);
        CHECK(status, "clSetKernelArg 2");
        status = clSetKernelArg(kernel, 3, sizeof(int), &image_width);
        CHECK(status, "clSetKernelArg 3");
        status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &input_mem);
        CHECK(status, "clSetKernelArg 4");
        status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &output_mem);
        CHECK(status, "clSetKernelArg 5");
        
        prev_img_w = image_width;
        prev_img_h = image_height;
        prev_filter_w = filter_width;
    }

    size_t local_work_size[2] = {16, 16};
    size_t global_work_size[2];
    global_work_size[0] = (image_width + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0];
    global_work_size[1] = (image_height + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1];

    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");

    // Read back result
    status = clEnqueueReadBuffer(queue, output_mem, CL_TRUE, 0, image_height * image_width * sizeof(float), output_image, 0, NULL, NULL);
    CHECK(status, "clEnqueueReadBuffer");
}