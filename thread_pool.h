#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <optional>
#include <functional>
#include <condition_variable>
#include <future>
#include <type_traits>
#include <cassert>

template <typename T>
class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads = 0)
        : stop(false)
    {
        if (numThreads == 0){
            numThreads = std::thread::hardware_concurrency();
            if (numThreads == 0)
                numThreads = 1;
        }

        for (size_t i = 0; i < numThreads; i++) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !taskQueue.empty(); });

                        if (stop && taskQueue.empty()) {
                            return;
                        }

                        task = std::move(taskQueue.front());
                        taskQueue.pop();
                    }

                    task();
                }
            });
        }
    }

    ~ThreadPool()
    {
        {
            std::scoped_lock<std::mutex> lock(queueMutex);
            stop = true;
        }

        condition.notify_all();

        for (std::thread &worker : workers)
            if (worker.joinable())
                worker.join();
    }

    void finish()
    {
        {
            std::scoped_lock<std::mutex> lock(queueMutex);
            stop = true;
        }

        condition.notify_all();

        for (std::thread &worker : workers)
            worker.join();

    }

    template <typename F, typename... Args>
    auto enqueue(F &&f, Args &&... args) -> std::future<std::invoke_result_t<F, Args...>>
    {
        using return_type = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> result = task->get_future();

        {
            std::scoped_lock<std::mutex> lock(queueMutex);

            // don't allow enqueueing after stopping the pool
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            taskQueue.emplace([task]() { (*task)(); });
        }

        condition.notify_one();
        return result;
    }

    template <typename F, typename... Args>
    void enqueueAndCollect(F &&f, Args &&... args)
    {
        using return_type = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        {
            std::scoped_lock<std::mutex> lock(queueMutex);

            // don't allow enqueueing after stopping the pool
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            taskQueue.emplace([task, this]() {
                std::future<return_type> result = task->get_future();
                (*task)();
                std::scoped_lock<std::mutex> resultLock(resultMutex);
                results.push_back(result.get());
            });
        }

        condition.notify_one();
    }

    const std::vector<T>& getResults()
    {
        assert(stop == true);
        std::scoped_lock<std::mutex> resultLock(resultMutex);
        return results;
    }

private:
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers;

    // the task queue
    std::queue<std::function<void()>> taskQueue;

    // synchronization
    std::mutex queueMutex;
    std::condition_variable condition;

    // results and synchronization
    std::vector<T> results;
    std::mutex resultMutex;

    // flag to stop the threads
    bool stop;
};

/*************************************************
// Example of use

#include <iostream>
#include <random>

int main() {
    {
        // A pool with 2 threads. Tasks will return an integer
        ThreadPool<int> pool(2);

        // Launch a lambda and get the result without saving it in the result queue
        // The result type is std::future<int>
        auto r = pool.enqueue([](int x, int y) { std::this_thread::sleep_for(std::chrono::seconds(1)); return x + y; }, 20, 3);
        std::cout << "Result: " << r.get() << std::endl;
    }

    {
        // A pool with "std::thread::hardware_concurrency()" threads.
        // Tasks will return a vector with std::string items
        ThreadPool<std::vector<std::string>> pool;

        // Long task
        auto long_task = [] (int prefix)
        {
            // wait between 10 ms and 1 second
            std::mt19937_64 eng{std::random_device{}()};
            std::uniform_int_distribution<> dist{10, 1000};
            std::this_thread::sleep_for(std::chrono::milliseconds{dist(eng)});

            std::vector<std::string> r;
            for (auto i = 0; i < 100; i++)
                r.push_back(std::to_string(prefix) + "-" + std::to_string(i));

            return r;
        };

        // Enqueue some tasks and collect results
        for (auto prefix = 0; prefix < 10; prefix++)
            pool.enqueueAndCollect(long_task, prefix);

        // Wait until the end of all tasks
        pool.finish();

        // Get and print the results.
        // getResults() returns a vector with the values returned by the tasks
        // auto results = pool.getResults();  // is the common way to get the results
        const std::vector< std::vector<std::string> > results = pool.getResults();
        for (auto & result : results)
            std::cout << "Result with: " << result.size() << " items" << std::endl;
    }

    return 0;
}
*************************************************/
