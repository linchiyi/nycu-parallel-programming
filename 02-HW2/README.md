# HW2｜多執行緒程式設計

本作業比較兩種多執行緒應用：

- part1：使用 Pthreads 以 Monte Carlo 方法估算 π。
- part2：使用 std::thread 平行計算 Mandelbrot 圖形。

part1 內含 Makefile，可在具備課程 starter code 的環境中執行：

~~~
make -C part1
./part1/pi.out <num_threads> <num_tosses>
~~~

part2 依賴課程提供的 serial Mandelbrot 實作與 cycle timer。

