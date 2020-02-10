#include "ThreadPool.h"
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <iostream>

struct ThreadPoolData {
    struct ThreadData;
    uint32_t thread_count;
    ThreadData * workers;
    ThreadData * next_thread;
    std::vector<std::future<void>> futures;

    ThreadPoolData(uint32_t n) :
        thread_count(n),
        workers(nullptr),
        next_thread(nullptr)
    {
        workers = new ThreadData[n];
        for (uint32_t i = 0; i < n; ++i)
            workers[i].id = i;
        next_thread = workers;
    }

    ~ThreadPoolData() {
        delete[] workers;
    }

    struct ThreadData {
        uint32_t id;
        std::thread thread;
        std::condition_variable condition;
        std::queue<std::packaged_task<void()>> jobs;
        uint32_t n_jobs = 0;
        std::mutex mutex;
        bool stop = false;
        inline ThreadData() : thread(&ThreadPoolData::ThreadFunction, this) {}
        ~ThreadData() {
            stop = true;
            condition.notify_one();
            thread.join();
        }
    };

    static void ThreadFunction(ThreadData * thread_data) {
        std::unique_lock<std::mutex> locker(thread_data->mutex, std::defer_lock);
        while (true) {
            locker.lock();
            thread_data->condition.wait(locker, [thread_data]() { return (thread_data->stop || !thread_data->jobs.empty() ); });
            if (thread_data->stop) { return; }
            std::packaged_task<void()> job = std::move(thread_data->jobs.front());
            thread_data->jobs.pop();
            locker.unlock();
            job();
        }
    }
};

ThreadPool::ThreadPool(uint32_t n) : data(new ThreadPoolData(n)) { }

ThreadPool::~ThreadPool() {
    delete data;
}

ThreadPool& ThreadPool::getInstance(uint32_t n) {
    static ThreadPool pool(n);
    return pool;
}

ThreadPool& ThreadPool::getInstance() {
    static ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
}

std::future<void>& ThreadPool::push_job(const std::function<void()> & f) {
    std::packaged_task<void()> job(f);
    data->futures.emplace_back(job.get_future());
    std::unique_lock<std::mutex> locker(data->next_thread->mutex, std::defer_lock);
    locker.lock();
    data->next_thread->jobs.push(std::move(job));
    data->next_thread->condition.notify_one();
    locker.unlock();
    if (data->next_thread->id == data->thread_count-1)
        data->next_thread = data->workers;
    else
        data->next_thread++;
    return data->futures.back();
}

void ThreadPool::wait() {
    for (std::future<void>& f : data->futures)
        f.get();
    data->futures.clear();
}
