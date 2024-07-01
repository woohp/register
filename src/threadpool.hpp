#include <future>
#include <mutex>
#include <queue>
#include <thread>

template <typename T>
struct blocked_range : public std::pair<T, T>
{
    blocked_range<T>(const T& first, const T& second)
        : std::pair<T, T>(first, second)
    { }

    T begin() const
    {
        return this->first;
    }

    T end() const
    {
        return this->second;
    }
};

class thread_pool
{
private:
    typedef std::packaged_task<void(size_t)> task_type;

    std::vector<std::thread> workers;
    std::queue<task_type> tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;  // Flag to indicate if the pool has been stopped

public:
    thread_pool(size_t threads = 0)
    {
        if (threads == 0)
            threads = std::thread::hardware_concurrency();

        for (size_t i = 0; i < threads; i++)
        {
            this->workers.emplace_back([this, i] {
                for (;;)
                {
                    task_type task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || this->tasks.size() > 0; });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task(i);
                }
            });
        }
    }

    ~thread_pool()
    {
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->stop = true;
        }
        this->condition.notify_all();
        for (auto& worker : this->workers)
            worker.join();
    }

    size_t num_threads() const
    {
        return this->workers.size();
    }

    template <typename Function>
    std::future<void> enqueue(Function&& f)
    {
        auto task = task_type { f };

        std::future<void> res = task.get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // don't allow enqueueing after stopping the pool
            if (this->stop)
                throw std::runtime_error("enqueue on stopped thread_pool");

            this->tasks.push(std::move(task));
        }
        this->condition.notify_one();

        return res;
    }

    template <typename Index, typename Function>
    void parallel_for(Index start, Index end, Function&& f)
    {
        if (end - start < 2)
        {
            for (Index i = start; i < end; i++)
                f(blocked_range<Index>(i, i + 1), 0);
            return;
        }

        const auto block_size = (end - start) / this->workers.size();
        auto leftover = (end - start) - block_size * this->workers.size();

        std::vector<std::future<void>> futures;

        if (block_size == 0)
        {
            for (Index i = 0; i < leftover; i++)
            {
                blocked_range<Index> range(i, i + 1);
                futures.push_back(this->enqueue([range, &f](size_t thread_idx) { f(range, thread_idx); }));
            }
        }
        else
        {
            Index cumsum = start;
            for (size_t i = 0; i < this->workers.size(); i++)
            {
                // add 1 to this worker's block size if there is any leftover, and then decrease leftover by 1
                auto block_size_ = block_size;
                if (leftover > 0)
                {
                    block_size_++;
                    leftover--;
                }

                blocked_range<Index> range(cumsum, cumsum + block_size_);
                futures.push_back(this->enqueue([range, &f](size_t thread_idx) { f(range, thread_idx); }));
                cumsum += block_size_;
            }
        }

        for (auto& future : futures)
            future.wait();
    }

    template <typename Index, typename Function>
    void parallel_for_each(Index start, Index end, Function&& f)
    {
        auto block_f = [&f](blocked_range<Index> range, size_t thread_idx) {
            for (; range.first < range.second; range.first++)
            {
                f(range.first, thread_idx);
            }
        };
        this->parallel_for<Index>(start, end, block_f);
    }
};
