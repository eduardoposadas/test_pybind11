#include "SieveOfEratosthenes.h"
#include "utils.h"

#include <vector>
#include <thread>
#include <future>


std::list<unsigned long long> SieveOfEratosthenes_std_list(const std::string& name, unsigned long long n)
{
    auto chronometer = StopWatch("Inside start: " + name);

    // Create a boolean array "sieve[0..n]" and initialize
    // all entries it as true. A value in sieve[i] will
    // finally be false if i is Not a prime, else true.
//    std::vector<bool> sieve(n + 1, true);
    std::shared_ptr<bool[]> sieve(new bool[n + 1]);
    sieve[0] = sieve[1] = false;  // 0 and 1 are not primes
    std::memset(sieve.get() + 2, true, n - 1); // -1 = +1 - 2


    for (unsigned long long p = 2; p * p <= n; p++) {
        // If sieve[p] is unchanged, then it is a prime number.
        if (sieve[p]) {
            // Update all multiples of p greater than or
            // equal to the square of it numbers which are
            // multiple of p and are less than p^2 are
            // already been marked.
            for (unsigned long long i = p * p; i <= n; i += p)
                sieve[i] = false;
        }
    }

    std::list<unsigned long long> primes;
    for (unsigned long long p = 2; p <= n; p++)
        if (sieve[p])
            primes.push_back(p);

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return primes;
}


/*****************************************************************************/

py::list SieveOfEratosthenes_python_list(const std::string& name, unsigned long long n)
{
    auto chronometer = StopWatch("Inside start: " + name);

    // Create a boolean array "sieve[0..n]" and initialize
    // all entries it as true. A value in sieve[i] will
    // finally be false if i is Not a prime, else true.
    std::vector<bool> sieve(n + 1, true);

    for (unsigned long long p = 2; p * p <= n; p++) {
        // If sieve[p] is unchanged, then it is a prime number.
        if (sieve[p]) {
            // Update all multiples of p greater than or
            // equal to the square of it numbers which are
            // multiple of p and are less than p^2 are
            // already been marked.
            for (unsigned long long i = p * p; i <= n; i += p)
                sieve[i] = false;
        }
    }

    py::list primes;
    for (unsigned long long p = 2; p <= n; p++)
        if (sieve[p])
            primes.append(p);

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return primes;
}

/*****************************************************************************/

// The returned value becomes a VectorULongLongInt in python
// std::vector<unsigned long long> is an opaque type in test_pybind11.cpp
std::vector<unsigned long long> SieveOfEratosthenes_std_vector(const std::string& name, unsigned long long n)
{
    auto chronometer = StopWatch("Inside start: " + name);

    // Create a boolean array "sieve[0..n]" and initialize
    // all entries it as true. A value in sieve[i] will
    // finally be false if i is Not a prime, else true.
    std::vector<bool> sieve(n + 1, true);

    for (unsigned long long p = 2; p * p <= n; p++) {
        // If sieve[p] is unchanged, then it is a prime number.
        if (sieve[p]) {
            // Update all multiples of p greater than or
            // equal to the square of it numbers which are
            // multiple of p and are less than p^2 are
            // already been marked.
            for (unsigned long long i = p * p; i <= n; i += p)
                sieve[i] = false;
        }
    }

    std::vector<unsigned long long> primes;
    for (unsigned long long p = 2; p <= n; p++)
        if (sieve[p])
            primes.push_back(p);

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return primes;
}


/*****************************************************************************/

// Taken from https://github.com/ssciwr/pybind11-numpy-example/blob/main/python/pybind11-numpy-example_python.cpp
// helper function to avoid making a copy when returning a py::array_t
// author: https://github.com/YannickJadoul
// source: https://github.com/pybind/pybind11/issues/1042#issuecomment-642215028
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq) {
  auto size = seq.size();
  auto data = seq.data();
  std::unique_ptr<Sequence> seq_ptr =
      std::make_unique<Sequence>(std::move(seq));
  auto capsule = py::capsule(seq_ptr.get(), [](void *p) {
    std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p));
  });
  seq_ptr.release();
  return py::array(size, data, capsule);
}

py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy(const std::string& name, unsigned long long n)
{
    auto chronometer = StopWatch("Inside start: " + name);

    // Create a boolean array "sieve[0..n]" and initialize
    // all entries it as true. A value in sieve[i] will
    // finally be false if i is Not a prime, else true.
    std::vector<bool> sieve(n + 1, true);

    for (unsigned long long p = 2; p * p <= n; p++) {
        // If sieve[p] is unchanged, then it is a prime number.
        if (sieve[p]) {
            // Update all multiples of p greater than or
            // equal to the square of it numbers which are
            // multiple of p and are less than p^2 are
            // already been marked.
            for (unsigned long long i = p * p; i <= n; i += p)
                sieve[i] = false;
        }
    }

    std::vector<unsigned long long> primes;
    for (unsigned long long p = 2; p <= n; p++)
        if (sieve[p])
            primes.push_back(p);

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return as_pyarray(std::move(primes));
}

/*****************************************************************************/
#include <omp.h>

py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_omp(const std::string& name, unsigned long long n)
{
    auto chronometer = StopWatch("Inside start: " + name);

    // Create a boolean array "sieve[0..n]" and initialize
    // all entries it as true. A value in sieve[i] will
    // finally be false if i is Not a prime, else true.
    // std::vector<bool> sieve(n + 1, true);  // std::vector<bool> is slow and it is not thread safe
    std::shared_ptr<bool[]> sieve(new bool[n + 1]);
    sieve[0] = sieve[1] = false;  // 0 and 1 are not primes
    std::memset(sieve.get() + 2, true, n - 1); // -1 = +1 - 2

    unsigned long long n_sqrt = std::sqrt(n);  // OpenMP doesn't like p * p <= n as cond-expression in for loops
#pragma omp parallel for schedule(dynamic)
    for (unsigned long long p = 2; p <= n_sqrt; p++) {
        // If sieve[p] is unchanged, then it is a prime number.
        if (sieve[p]) {
            // Update all multiples of p greater than or
            // equal to the square of it numbers which are
            // multiple of p and are less than p^2 are
            // already been marked.
            for (unsigned long long i = p * p; i <= n; i += p)
                sieve[i] = false;
        }
    }

    chronometer.elapsed("Before filling in the list of prime numbers. Duration:");
    std::vector<unsigned long long> primes;
//    #pragma omp parallel for shared(sieve, primes)  // push_back it is no thread safe
    for (unsigned long long p = 2; p <= n; p++)
        if (sieve[p])
            primes.push_back(p);

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return as_pyarray(std::move(primes));
}

/*****************************************************************************/
void SieveOfEratosthenes_worker(std::shared_ptr<bool[]> sieve,
                                unsigned long long start,
                                unsigned long long end)
{
    unsigned long long min_j;

    for (unsigned long long i = 2; i*i <= end; i++){
        min_j = ((start+i-1)/i)*i;
        if (min_j < i*i)
            min_j = i*i;
        for (unsigned long long j = min_j; j <= end; j += i){
            sieve[j] = false;
        }
    }
}

py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_thread(const std::string& name, unsigned long long n)
{
    auto chronometer = StopWatch("Inside start: " + name);

    // std::vector<bool> sieve(n, true);  // std::vector<bool> is slow and it is not thread safe
    // auto array = std::make_shared<bool[]>(n + 1);  // C++20
    std::shared_ptr<bool[]> sieve(new bool[n + 1]);
    sieve[0] = sieve[1] = false;  // 0 and 1 are not primes
    std::memset(sieve.get() + 2, true, n - 1); // -1 = +1 - 2

    auto n_workers = std::thread::hardware_concurrency();
    if (n_workers == 0)
        n_workers = 1;

    // chunk_size = The amount of data that fit in the L1 data cache of the CPU.
    //              The workers use a list of booleans, so to find out the number of elements in the list you
    //              have to divide by the size of a boolean.
    const unsigned long long chunk_size = get_CPU_L1_data_cache_size() / sizeof(bool);

    // Launches at most n_workers concurrent workers working on a portion of chunk_size elements of sieve array.
    using task_type = std::packaged_task< void(std::shared_ptr<bool[]>,
                                              unsigned long long,
                                              unsigned long long) >;
    std::vector<std::future<void>> future_list;
    unsigned long long start = 0;
    unsigned long long end = 0;
    while (start < n){
        if (future_list.size() < n_workers){
            // Launch a thread
            end = start + chunk_size > n ? n : start + chunk_size;
            task_type task(SieveOfEratosthenes_worker);
            future_list.push_back(task.get_future());
            std::thread t(std::move(task), sieve, start, end);
            t.detach();
            start += chunk_size;
        }
        else {
            // Wait for a thread to end
            bool all_threads_running = true;
            do {
                for (auto f = future_list.begin(); f != future_list.end(); f++){
                    if (f->wait_for(std::chrono::milliseconds(2)) == std::future_status::ready){
                        future_list.erase(f);
                        all_threads_running = false;
                        break;
                    }
                }
            } while (all_threads_running);
        }
    }

    // Wait for the threads to finish
    for (auto & f : future_list)
        f.wait();

    chronometer.elapsed("Before filling in the list of prime numbers. Duration:");
    std::vector<unsigned long long> primes;
    for (long long unsigned i = 0; i <= n; i++)
        if (sieve[i])
            primes.push_back(i);

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return as_pyarray(std::move(primes));
}


/*****************************************************************************/
#include "jobs_fifo_queue.h"

// Job type for the queue of jobs
struct job_type{
    unsigned long long input_start;
    unsigned long long input_end;
    std::vector<unsigned long long> output_primes;
};

void SieveOfEratosthenes_pool_worker(jobs_fifo_queue<std::shared_ptr<job_type>> &queue)
{
    unsigned long long min_j;

    // Wait for a new job or end this thread
    while (auto job = queue.dequeue()){

        // Job from the queue
        auto start = (*job)->input_start;
        auto end = (*job)->input_end;
        auto n = end - start;

//        std::vector<bool> sieve(n, true);          // Slower than the two below
//        auto sieve = std::make_unique<bool[]>(n + 1);  // I don't need zero initialization
        std::unique_ptr<bool[]> sieve(new bool[n + 1]);
        if (start == 0){
            sieve[0] = sieve[1] = false;  // 0 and 1 are not primes
            std::memset(sieve.get() + 2, true, n - 1); // -1 = +1 - 2
        } else {
            std::memset(sieve.get(), true, n + 1);
        }

        for (unsigned long long i = 2; i*i <= end; i++){
            min_j = ((start+i-1)/i)*i;
            if (min_j < i*i)
                min_j = i*i;
            for (unsigned long long j = min_j; j <= end; j += i)
                sieve[j - start] = false;
        }

        // Save prime numbers
        for (long long unsigned i = 0; i <= n; i++)
            if (sieve[i])
                (*job)->output_primes.push_back(i + start);
    }
}

py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_thread_pool(const std::string& name, unsigned long long n)
{
    auto chronometer = StopWatch("Inside start: " + name);

    // Number of workers
    auto n_workers = std::thread::hardware_concurrency();
    if (n_workers == 0)
        n_workers = 1;

    // Queue of jobs
    jobs_fifo_queue<std::shared_ptr<job_type>> jobs_queue(n_workers * 100);

    // Workers pool
    std::vector<std::thread> workers_pool;
    for(decltype(n_workers) i = 0; i < n_workers; i++)
        workers_pool.emplace_back(SieveOfEratosthenes_pool_worker, std::ref(jobs_queue));

    // chunk_size = The amount of data that fit in the L1 data cache of the CPU.
    //              The workers use a list of booleans, so to find out the number of elements in the list you
    //              have to divide by the size of a boolean.
    const unsigned long long chunk_size = get_CPU_L1_data_cache_size() / sizeof(bool);
    // chunk_size *= 1.2;  // I don't know why this speeds up the process

    // Produce jobs for workers
    unsigned long long start = 0;
    while (start <= n){
        auto job = std::make_unique<job_type>();
        job->input_start = start;
        job->input_end = start + chunk_size > n ? n : start + chunk_size;
        job->output_primes = {};
        jobs_queue.enqueue(std::move(job));

        start += chunk_size;
    }

    // No more jobs, close the queue (close the channel) and ends the threads
    jobs_queue.finish();

    // Wait for the threads to finish
    for (auto & worker : workers_pool)
        if (worker.joinable())
            worker.join();

    chronometer.elapsed("Before filling in the list of prime numbers. Duration:");

    // Collects the lists of prime numbers from the job queue
    std::vector<unsigned long long> primes;
    for(auto & j : jobs_queue)
        primes.insert(primes.end(), j->output_primes.begin(), j->output_primes.end());

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return as_pyarray(std::move(primes));
}
/*****************************************************************************/
#include "thread_pool.h"

std::vector<unsigned long long> SieveOfEratosthenes_worker_2(std::shared_ptr<bool[]> sieve,
                                unsigned long long start,
                                unsigned long long end)
{
    // It's faster to share the same sieve
    // std::shared_ptr<bool[]> sieve(new bool[end]);
    // sieve[0] = sieve[1] = false;  // 0 and 1 are not primes
    // std::memset(sieve.get() + 2, true, end - 2);

    unsigned long long min_j;

    for (unsigned long long i = 2; i*i <= end; i++){
        min_j = ((start+i-1)/i)*i;
        if (min_j < i*i)
            min_j = i*i;
        for (unsigned long long j = min_j; j <= end; j += i){
            sieve[j] = false;
        }
    }

    std::vector<unsigned long long> primes;
    for (long long unsigned i = start; i <= end; i++)
        if (sieve[i])
            primes.push_back(i);

    return primes;
}

py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_generic_thread_pool(const std::string& name, unsigned long long n)
{
    auto chronometer = StopWatch("Inside start: " + name);

    // std::vector<bool> sieve(n, true);  // std::vector<bool> is slow and it is not thread safe
    // auto array = std::make_shared<bool[]>(n + 1);  // C++20
    std::shared_ptr<bool[]> sieve(new bool[n + 1]);
    sieve[0] = sieve[1] = false;  // 0 and 1 are not primes
    std::memset(sieve.get() + 2, true, n - 1); // -1 = +1 - 2

    ThreadPool<std::vector<unsigned long long>> workers_pool;

    // chunk_size = The amount of data that fit in the L1 data cache of the CPU.
    //              The workers use a list of booleans, so to find out the number of elements in the list you
    //              have to divide by the size of a boolean.
    const unsigned long long chunk_size = get_CPU_L1_data_cache_size() / sizeof(bool);
    // chunk_size *= 1.2;  // I don't know why this speeds up the process

    // Produce jobs for workers
    unsigned long long start = 0;
    while (start <= n){
        auto end = start + chunk_size > n ? n : start + chunk_size;
        workers_pool.enqueueAndCollect(SieveOfEratosthenes_worker_2, sieve, start, end);

        start += chunk_size;
    }

    // No more jobs. Waits until all tasks have finished and ends the threads
    workers_pool.finish();

    chronometer.elapsed("Before filling in the list of prime numbers. Duration:");

    // sort the results of the job queue
    auto results = workers_pool.getResults();
    std::sort(//std::execution::par,   // use TBB lib
              results.begin(), results.end(),
              [](std::vector<unsigned long long> a, std::vector<unsigned long long> b)
              {
                    return a[0] < b[0];
              });

    // Collects the lists of prime numbers
    std::vector<unsigned long long> primes;
    for(auto & l : results)
        primes.insert(primes.end(), l.begin(), l.end());

    chronometer.elapsed("Inside end: " + name + " Duration:");
    return as_pyarray(std::move(primes));
}

/*****************************************************************************/
