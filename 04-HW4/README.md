# HW4｜MPI 分散式程式設計

本作業使用 MPI 實作並比較不同程序間通訊方式：

- part1/hello.c：MPI 基本程序與 rank 資訊。
- part1/pi_*.c：以 Monte Carlo 方法計算 π，涵蓋 reduce、gather、blocking、nonblocking 與 one-sided communication。
- part2/matmul.cc：分散式矩陣乘法與矩陣資料廣播。

此作業需要 MPI compiler、課程提供的測試程式與執行環境；程式中的執行時間以 MPI_Wtime 量測。

