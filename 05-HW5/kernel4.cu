#include <cuda.h>
#include <cstdio>
#include <cstdlib>

// --- Template Kernel ---
// UseGeometricOpt = true: View 1 幾何優化 (秒殺黑洞區)
// UseGeometricOpt = false: View 2 純運算 (極致速度)
template <bool UseGeometricOpt>
__global__ void mandel_kernel_template(float lower_x, float lower_y, float step_x, float step_y, 
                                       int * __restrict__ img, int res_x, int res_y, int max_iterations) {
    
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX >= res_x || thisY >= res_y) return;

    int idx = thisY * res_x + thisX;

    float c_re = lower_x + thisX * step_x;
    float c_im = lower_y + thisY * step_y;

    // --- View 1 幾何優化 ---
    if constexpr (UseGeometricOpt) {
        float c_re2 = c_re * c_re;
        float c_im2 = c_im * c_im;
        // Period-2 Bulb
        if ((c_re + 1.f) * (c_re + 1.f) + c_im2 < 0.0625f) {
            img[idx] = max_iterations;
            return;
        }
        // Cardioid
        float q = (c_re - 0.25f) * (c_re - 0.25f) + c_im2;
        if (q * (q + (c_re - 0.25f)) < 0.25f * c_im2) {
            img[idx] = max_iterations;
            return;
        }
    }

    float z_re = c_re;
    float z_im = c_im;
    float z_re2 = z_re * z_re;
    float z_im2 = z_im * z_im;

    int i = 0;

    // --- View 2 加速: Batch-5 Lazy Check ---
    // 連續跑 5 次只檢查 1 次，大幅減少分支指令
    while (i + 5 < max_iterations) {
        float old_re = z_re;
        float old_im = z_im;

        // 5 次盲跑 (Compiler 會自動優化這段純數學運算)
        #pragma unroll
        for (int k = 0; k < 5; ++k) {
            float new_im = 2.f * z_re * z_im + c_im;
            z_re = (z_re * z_re) - (z_im * z_im) + c_re;
            z_im = new_im;
        }

        // 延遲檢查: 檢查第 5 次的結果
        // 注意: 這裡直接算平方和，不依賴 z_re2 (因為還沒更新)
        if (z_re * z_re + z_im * z_im > 4.f) {
            // 發生發散！倒帶回去找出確切位置
            z_re = old_re;
            z_im = old_im;
            z_re2 = z_re * z_re;
            z_im2 = z_im * z_im;
            
            for (int k = 0; k < 5; ++k) {
                if (z_re2 + z_im2 > 4.f) {
                    img[idx] = i + k;
                    return;
                }
                float new_im = 2.f * z_re * z_im + c_im;
                z_re = z_re2 - z_im2 + c_re;
                z_im = new_im;
                z_re2 = z_re * z_re;
                z_im2 = z_im * z_im;
            }
        }

        // [關鍵修復] 同步 z_re2/z_im2
        // 之前的版本漏了這步，導致 Tail Loop 用到舊值，造成 correctness 錯誤
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;

        i += 5;
    }

    // --- Tail Loop: 處理剩下的迭代 ---
    // 這裡使用正確更新後的 z_re2/z_im2
    for (; i < max_iterations; ++i) {
        if (z_re2 + z_im2 > 4.f) break;
        float new_im = 2.f * z_re * z_im + c_im;
        z_re = z_re2 - z_im2 + c_re;
        z_im = new_im;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;
    }

    img[idx] = i;
}

void host_fe(float upper_x, float upper_y, float lower_x, float lower_y, 
             int *img, int res_x, int res_y, int max_iterations) {
    float step_x = (upper_x - lower_x) / res_x;
    float step_y = (upper_y - lower_y) / res_y;

    // --- Persistent Memory ---
    static int* d_img = nullptr;
    static size_t d_img_size = 0;
    size_t needed_size = res_x * res_y * sizeof(int);

    if (d_img == nullptr || needed_size > d_img_size) {
        if (d_img != nullptr) cudaFree(d_img);
        cudaMalloc((void **)&d_img, needed_size);
        d_img_size = needed_size;
    }

    // --- View 判斷 ---
    bool is_view1 = false;
    if ((lower_x < 0.25f && upper_x > -0.5f && lower_y < 0.5f && upper_y > -0.5f) || 
        (lower_x < -0.75f && upper_x > -1.25f && lower_y < 0.25f && upper_y > -0.25f)) 
    {
        is_view1 = true;
        if (lower_y > 0.25f) is_view1 = false;
    }

    // Block 設定: 32x8 經測試為最佳
    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks((res_x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (res_y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    if (is_view1) {
        mandel_kernel_template<true><<<numBlocks, threadsPerBlock>>>(
            lower_x, lower_y, step_x, step_y, d_img, res_x, res_y, max_iterations);
    } else {
        mandel_kernel_template<false><<<numBlocks, threadsPerBlock>>>(
            lower_x, lower_y, step_x, step_y, d_img, res_x, res_y, max_iterations);
    }
    
    cudaMemcpy(img, d_img, needed_size, cudaMemcpyDeviceToHost);
}