# HW5｜CUDA Mandelbrot

本作業以 CUDA kernel 平行化 Mandelbrot 集合計算，並逐步比較不同的 GPU 優化策略：

- 基本 thread mapping。
- 使用 pitched memory。
- 讓單一 thread 處理一組像素。
- 以幾何性質快速判斷 Mandelbrot 集合內的區域。

主要檔案為 kernel1.cu 至 kernel4.cu。完整編譯與 benchmark 需要搭配課程提供的 host code、測試框架與 CUDA 環境。

