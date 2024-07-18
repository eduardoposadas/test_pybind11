// A FIFO queue that keeps all the items that were dequeued.
// The first item of the FIFO queue is the item at not_processed_init.
// The last item of the FIFO queue is the item at not_processed_end.
// When an item is enqueued, it is added to the queue and
// not_processed_end is incremented.
// When an item is dequeued, the item at the position not_processed_init
// is returned and not_processed_init is incremented.
// When all items are dequeued not_processed_init == not_processed_end
// and the class member "queue" keeps all the items.
// This queue can be traversed with the iterators.

// Modified from:
// https://stackoverflow.com/questions/15278343/c11-thread-safe-queue
// https://stackoverflow.com/a/68055756

#pragma once

#include <queue>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <cassert>

template <typename T>
class jobs_fifo_queue{
public:
    explicit jobs_fifo_queue(int max_not_processed_ = 10);
    void enqueue(T input_job);
    std::optional<T> dequeue();
    void finish();
    typename std::deque<T>::iterator begin();
    typename std::deque<T>::iterator end();
private:
    std::deque<T> queue;
    bool finished = false;
    int max_not_processed;
    int not_processed_init = 0;
    int not_processed_end = 0;

    std::mutex mux;
    std::condition_variable cond;
};

template<typename T>
jobs_fifo_queue<T>::jobs_fifo_queue(int max_not_processed_)
    : max_not_processed(max_not_processed_)
{
}

template<typename T>
void jobs_fifo_queue<T>::enqueue(T input)
{
    std::unique_lock lock(mux);
    cond.wait(lock, [this]() {
        return (not_processed_end - not_processed_init) < max_not_processed;});

    queue.push_back(input);
    not_processed_end++;

    cond.notify_one();
}

template<typename T>
std::optional<T> jobs_fifo_queue<T>::dequeue()
{
    T returned_value;
    std::unique_lock lock(mux);
    cond.wait(lock, [this]() {
        return (not_processed_end - not_processed_init != 0) || finished;
    });

    if (not_processed_end - not_processed_init == 0) {
        assert(finished);
        return std::nullopt;
    }

    returned_value = queue.at(not_processed_init);
    not_processed_init++;
    cond.notify_one();
    return returned_value;
}

template<typename T>
void jobs_fifo_queue<T>::finish()
{
    std::scoped_lock lock(mux);
    finished = true;
    cond.notify_all();
}


// Iterators

template<typename T>
typename std::deque<T>::iterator jobs_fifo_queue<T>::begin()
{
    return queue.begin();
}

template<typename T>
typename std::deque<T>::iterator jobs_fifo_queue<T>::end()
{
    return queue.end();
}
