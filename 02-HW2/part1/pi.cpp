#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <ctime>

struct ThreadArg {
    long long tosses;
    long long local_hits;
    unsigned int seed;
};

void* monte_carlo(void* arg) {
    ThreadArg* data = static_cast<ThreadArg*>(arg);
    long long hits = 0;
    unsigned int seed = data->seed;

    for (long long i = 0; i < data->tosses; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0) {
            hits++;
        }
    }

    data->local_hits = hits;
    pthread_exit(nullptr);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./pi.out <num_threads> <num_tosses>\n";
        return 1;
    }

    int num_threads = std::atoi(argv[1]);
    long long total_tosses = std::atoll(argv[2]);
    if (num_threads <= 0 || total_tosses <= 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }

    pthread_t* threads = new pthread_t[num_threads];
    ThreadArg* args = new ThreadArg[num_threads];

    // allocate works to every single thread
    long long base = total_tosses / num_threads;
    long long rem = total_tosses % num_threads;

    for (int i = 0; i < num_threads; i++) {
        args[i].tosses = base + (i < rem ? 1 : 0);
        args[i].local_hits = 0;
        args[i].seed = static_cast<unsigned int>(time(nullptr)) ^ (i * 7919); // 不同 seed
        pthread_create(&threads[i], nullptr, monte_carlo, &args[i]);
    }

    long long total_hits = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
        total_hits += args[i].local_hits;
    }

    double pi = 4.0 * (double)total_hits / (double)total_tosses;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << pi << std::endl;

    delete[] threads;
    delete[] args;
    return 0;
}
