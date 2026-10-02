# 平行程式設計

本 repository 整理課程中完成的 SIMD、多執行緒、OpenMP、MPI、CUDA 與 OpenCL 作業，以及專案提案、期末報告與 HackMD 紀錄。部分作業依賴課程提供的 starter code、標頭檔或測試框架，因此不保證可以脫離課程環境直接編譯。

## 作業總覽

| 作業 | 內容 | 技術 |
| --- | --- | --- |
| [HW1](./01-HW1/) | SIMD 向量化絕對值運算 | C++、PPintrin |
| [HW2](./02-HW2/) | Monte Carlo 計算 π、Mandelbrot | Pthreads、std::thread |
| [HW3](./03-HW3/) | BFS、PageRank、Conjugate Gradient | OpenMP |
| [HW4](./04-HW4/) | MPI 通訊與矩陣乘法 | MPI |
| [HW5](./05-HW5/) | Mandelbrot GPU kernel 最佳化 | CUDA |
| [HW6](./06-HW6/) | Tile-based 影像卷積 | OpenCL |

## 技術重點

- 使用 thread、process 與 GPU work-item 分配平行工作。
- 比較不同同步與通訊模式對效能的影響。
- 使用 block、tile、local memory 與 memory layout 降低記憶體存取成本。
- 以執行時間、正確性與可擴充性作為實驗觀察重點。

## 報告與連結

- [專案提案報告](./報告/Proposal_Report.pdf)
- [期末報告](./報告/Final_Report.pdf)
- [HW1 HackMD 作業紀錄](./報告/HW1_HackMD.md)
- [HW2 HackMD 作業紀錄](./報告/HW2_HackMD.md)
- [期末專案程式碼連結](./報告/Final_Project_Link.md)
