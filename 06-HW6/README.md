# HW6｜OpenCL 影像卷積

本作業使用 OpenCL 實作影像卷積，將影像切成 16×16 tile，並使用含 halo 的 local memory 降低 global memory 存取。

- kernel.cl：OpenCL convolution kernel。
- host_fe.c：建立 command queue、配置 buffer、設定 kernel 參數並執行工作。

完整執行需要課程提供的 helper、header、測試資料與 OpenCL runtime。

