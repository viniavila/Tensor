#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <cinttypes>
#include <functional>
#include <future>

struct ThreadPoolData;

class ThreadPool {
public:
    static ThreadPool& getInstance(uint32_t n);
    static ThreadPool& getInstance();
    std::future<void>& push_job(const std::function<void()>&);
    void wait();

private:
    ThreadPool(uint32_t n);
    ThreadPoolData * const data;

public:
    ~ThreadPool();
    ThreadPool(ThreadPool const&) = delete;
    ThreadPool& operator=(ThreadPool const&) = delete;
};

#endif // THREADPOOL_H
