#include "CalculatingPi.h"
#include "utils.h"

#include <cmath>
#include <thread>
#include <future>
#include <vector>


long double pi_num_integration_cpp(const std::string& name, unsigned long long numRect)
{
    auto chronometer = StopWatch("Inside start: " + name);

    long double width;       /* width of a rectangle subinterval */
    long double x;           /* an x_i value for determining height of rectangle */
    long double sum;         /* accumulates areas all rectangles so far */

    sum = 0;
    width = 2.0 / numRect;
    for (unsigned long long i = 0; i < numRect; i++) {
        x = -1 + (i + 0.5) * width;
        sum += std::sqrt(1 - x * x) * width;
    }
    sum *= 2;

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return sum;
}

void pi_num_integration_cpp_worker(std::promise<long double> && p,
                           unsigned long long numRect,
                           unsigned long long start,
                           unsigned long long end)
{
    long double width;       /* width of a rectangle subinterval */
    long double x;           /* an x_i value for determining height of rectangle */
    long double sum;         /* accumulates areas all rectangles so far */

    sum = 0;
    width = 2.0 / numRect;
    for (unsigned long long i = start; i < end; i++) {
        x = -1 + (i + 0.5) * width;
        sum += std::sqrt(1 - x * x) * width;
    }

    p.set_value(sum);
}


long double pi_num_integration_cpp_threads(const std::string& name, unsigned long long n)
{
    auto chronometer = StopWatch("Inside start: " + name);

    long double sum = 0;
    std::vector<std::thread> workers;
    std::vector<std::future<decltype(sum)>> futures;

    // n_cpu is the number of logical CPUs. It should be the number of physical CPUs.
    // Hyperthreading does not help here, it just wastes time with context switches.
    // Maybe the FPU?
    auto n_cpu = std::thread::hardware_concurrency();

    // Launch n_cpu threads
    unsigned long long start, end;
    unsigned long long chunk_size = (n + n_cpu - 1) / n_cpu;
    for (uint proc = 0; proc < n_cpu; proc++) {
        std::promise<decltype(sum)> p;
        futures.push_back(p.get_future());
        start = chunk_size * proc;
        end = start + chunk_size > n ? n : start + chunk_size;
        workers.emplace_back(pi_num_integration_cpp_worker, std::move(p), n, start, end);
    }

    for (auto & w : workers)
        w.join();

    for(auto & f : futures)
        sum += f.get();
    sum *= 2;

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return sum;
}

long double pi_leibniz(const std::string& name, unsigned long long n)
{
    auto chronometer = StopWatch("Inside start: " + name);

    long double s = 1;
    long double k = 3;
    for (unsigned long long i = 1; i <= n; i++){
        // s += pow(-1, i) / (2 * i + 1);
        if (i % 2 == 0)  // much faster
            s += 1.0 / k;
        else
            s -= 1.0 / k;
        k += 2;
    }
    s *= 4;

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return s;
}

void pi_leibniz_worker(std::promise<long double> && p,
                       unsigned long long start,
                       unsigned long long end)
{
    long double s = start == 1 ? 1: 0;
    long double k = 2 * start + 1;

    for (unsigned long long i = start; i <= end; i++) {
        // s += pow(-1, i) / (2 * i + 1);
        if (i % 2 == 0) // much faster
            s += 1.0 / k;
        else
            s -= 1.0 / k;
        k += 2;
    }

    p.set_value(s);
}

long double pi_leibniz_threads(const std::string& name, unsigned long long n)
{
    auto chronometer = StopWatch("Inside start: " + name);

    long double sum = 0;
    std::vector<std::thread> workers;
    std::vector<std::future<decltype(sum)>> futures;
    auto n_cpu = std::thread::hardware_concurrency();

    // Launch n_cpu threads
    unsigned long long start, end;
    unsigned long long chunk_size = (n + n_cpu - 1) / n_cpu;
    for (uint proc = 0; proc < n_cpu; proc++) {
        std::promise<decltype(sum)> p;
        futures.push_back(p.get_future());
        start = (chunk_size * proc) + 1;
        end = start + chunk_size - 1 > n ? n : start + chunk_size - 1;
        workers.emplace_back(pi_leibniz_worker, std::move(p), start, end);
    }

    for (auto & w : workers)
        w.join();

    for(auto & f : futures)
        sum += f.get();
    sum *= 4;

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return sum;
}
