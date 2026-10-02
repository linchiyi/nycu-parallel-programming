#define TILE_SIZE 16
#define HALO_SIZE 3
#define LOCAL_SIZE (TILE_SIZE + 2 * HALO_SIZE)

__kernel void convolution(int filter_width, 
                          __constant float *filter, 
                          int imageHeight, 
                          int imageWidth, 
                          __global float *inputImage, 
                          __global float *outputImage) 
{
    // 取得各種 ID
    int tx = get_local_id(0); // 0 ~ 15
    int ty = get_local_id(1); // 0 ~ 15
    int gx = get_group_id(0) * TILE_SIZE; // WorkGroup 左上角的 Global X
    int gy = get_group_id(1) * TILE_SIZE; // WorkGroup 左上角的 Global Y
    
    // 定義 Local Memory 緩衝區
    // 大小為 [22][22] (16+6)x(16+6)
    __local float localImage[LOCAL_SIZE][LOCAL_SIZE];

    // --- 階段 1: 載入資料到 Local Memory ---
    
    // 每個 WorkGroup 有 16x16 = 256 個執行緒
    // 我們需要載入 22x22 = 484 個像素
    // 每個執行緒平均負責載入約 1.89 個像素
    // 我們將 2D 的 Local Index 攤平成 1D 線性 Index 來分配工作
    
    int thread_id = ty * TILE_SIZE + tx; // 0 ~ 255
    int pixels_to_load = LOCAL_SIZE * LOCAL_SIZE; // 484

    // 使用迴圈讓每個執行緒載入它負責的那些像素
    for (int i = thread_id; i < pixels_to_load; i += 256) {
        int r = i / LOCAL_SIZE; // Local Row (0~21)
        int c = i % LOCAL_SIZE; // Local Col (0~21)
        
        // 對應到的 Global 座標 (需減去 Halo 偏移量)
        int global_r = gy + r - HALO_SIZE;
        int global_c = gx + c - HALO_SIZE;
        
        float val = 0.0f;
        // 邊界檢查：確保讀取不越界 (Zero Padding)
        if (global_r >= 0 && global_r < imageHeight && global_c >= 0 && global_c < imageWidth) {
            val = inputImage[global_r * imageWidth + global_c];
        }
        
        localImage[r][c] = val;
    }

    // 等待該 Group 所有執行緒完成載入
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- 階段 2: 進行捲積運算 ---

    // 只需要計算 TILE 範圍內的 Output (不包含 Halo)
    // 這裡需要再次檢查是否超出影像邊界 (針對影像邊緣的 Tile)
    int global_x = gx + tx;
    int global_y = gy + ty;

    if (global_x < imageWidth && global_y < imageHeight) {
        int halffilter_size = filter_width / 2;
        float sum = 0.0f;
        
        // 在 Local Memory 中進行捲積
        // 我們的 Local Memory 中心點對應 (tx + HALO_SIZE, ty + HALO_SIZE)
        
        for (int k = -halffilter_size; k <= halffilter_size; k++) {
            for (int l = -halffilter_size; l <= halffilter_size; l++) {
                // filter 是 Row-Major: [row * width + col]
                // localImage 索引: center_y + k, center_x + l
                float pixel = localImage[ty + HALO_SIZE + k][tx + HALO_SIZE + l];
                float f_val = filter[(k + halffilter_size) * filter_width + (l + halffilter_size)];
                sum += pixel * f_val;
            }
        }
        
        outputImage[global_y * imageWidth + global_x] = sum;
    }
}