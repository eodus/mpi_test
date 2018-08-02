#pragma once

#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

namespace partask {

const size_t MPI_MAX_COUNT = std::numeric_limits<int>::max();

void broadcast_string(std::string &str, int root = 0) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    size_t size = -1;
    std::vector<char> s;
    if (world_rank == root) {
        size = str.size();
        s.resize(size + 1);
        ::strcpy(s.data(), str.c_str());
    }

    // Broadcast object size
    MPI_Bcast(&size, sizeof(size), MPI_CHAR, root, MPI_COMM_WORLD);

    if (world_rank != root) {
        s.resize(size + 1);
        s[size] = '\0';
    }

    // Broadcast object itself
    assert(size <= MPI_MAX_COUNT);
    MPI_Bcast(s.data(), size, MPI_CHAR, root, MPI_COMM_WORLD);

    if (world_rank != root) {
        str = std::string(s.data());
    }
}

bool init() {
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    return provided >= MPI_THREAD_MULTIPLE;
}

bool finalize() { return MPI_Finalize() == MPI_SUCCESS; }

std::vector<int> collect_num_threads(int root = 0) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int num_threads = omp_get_max_threads();
    if (world_rank == root) {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        std::vector<int> all_num_threads(world_size);
        MPI_Gather(&num_threads, 1, MPI_INT, all_num_threads.data(), 1, MPI_INT, root, MPI_COMM_WORLD);
        return all_num_threads;
    } else {
        MPI_Gather(&num_threads, 1, MPI_INT, nullptr, 1, MPI_INT, root, MPI_COMM_WORLD);
        return {};
    }
}

void all_set_num_threads(const std::vector<int> &all_num_threads, int root = 0) {
    int num_threads;
    MPI_Scatter(const_cast<int *>(all_num_threads.data()), 1, MPI_INT, &num_threads, 1, MPI_INT, root, MPI_COMM_WORLD);
    omp_set_num_threads(num_threads);
}

void all_set_num_threads(int num_threads, int root = 0) { omp_set_num_threads(num_threads); }

template <typename T, typename Serialize, typename Deserialize>
void broadcast(T &data, Serialize &serialize, Deserialize &deserialize, int root = 0) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    size_t size = -1;
    std::vector<char> s;
    if (world_rank == root) {
        std::stringstream ss;
        serialize(ss, data);
        size = ss.str().size();
        s.resize(size + 1);
        ::strcpy(s.data(), ss.str().c_str());
    }

    // Broadcast object size
    MPI_Bcast(&size, sizeof(size), MPI_CHAR, root, MPI_COMM_WORLD);

    if (world_rank != root) {
        s.resize(size + 1);
        s[size] = '\0';
    }

    // Broadcast object itself
    assert(size <= MPI_MAX_COUNT);
    MPI_Bcast(s.data(), size, MPI_CHAR, root, MPI_COMM_WORLD);

    if (world_rank != root) {
        std::stringstream ss(s.data());
        deserialize(ss, data);
    }
}

class OutputMPIStream : public std::ostringstream {
public:
    OutputMPIStream(int rank, int tag = 0) : rank_{rank}, tag_{tag} {}

    void close() {
        size_t size = this->str().size();
        assert(size <= MPI_MAX_COUNT);
        MPI_Send(&size, sizeof(size), MPI_CHAR, rank_, tag_, MPI_COMM_WORLD);
        MPI_Send(const_cast<char *>(this->str().data()), this->str().size(), MPI_CHAR, rank_, tag_, MPI_COMM_WORLD);
    }

    ~OutputMPIStream() { close(); }

private:
    int rank_;
    int tag_;
};

class InputMPIStream : public std::istringstream {
public:
    InputMPIStream(int rank, int tag = 0) {
        size_t size;
        MPI_Recv(&size, sizeof(size), MPI_CHAR, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        assert(size <= MPI_MAX_COUNT);
        std::vector<char> s;
        s.resize(size + 1);
        s[size] = '\0';
        MPI_Recv(s.data(), size, MPI_CHAR, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        this->str(s.data());
        std::cout << "Data recieved: " << this->str() << std::endl;
    }
};

template <typename MakeSplitter, typename Process, typename Reduce>
auto run(MakeSplitter &make_splitter, Process &process, Reduce &reduce) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    using return_type = decltype(reduce(std::vector<std::istream *>()));
    std::unique_ptr<return_type> result;

    const int MAP_TAG = 13;
    const int REDUCE_TAG = 14;

    // omp_set_num_threads(1);           // Section works even if we specify num_threads = 1
    auto plocal_map = std::make_shared<std::stringstream>();
    auto plocal_reduce = std::make_shared<std::stringstream>();

    auto all_num_threads = collect_num_threads();

    if (world_rank == 0) {
        std::vector<std::shared_ptr<std::ostream>> oss;
        oss.push_back(plocal_map);
        for (size_t rank = 1; rank < world_size; ++rank) {
            oss.emplace_back(new OutputMPIStream(rank, MAP_TAG));
        }

        size_t sum_num_threads = std::accumulate(all_num_threads.cbegin(), all_num_threads.cend(), 0);
        assert(sum_num_threads > 0);
        std::cout << "All threads: " << sum_num_threads << std::endl;
        auto splitter = make_splitter(sum_num_threads * 10);

        auto mult_splitter = [&](auto &os, int rank, size_t count) {
            for (size_t i = 0; i < count; ++i) {
                bool result = splitter(os, rank);
                if (!result) return false;
            }
            return true;
        };

        for (size_t rank = 0; mult_splitter(*oss[rank], rank, all_num_threads[rank]); rank = (rank + 1) % world_size) {
        }
    }

    if (world_rank == 0) {
        process(*plocal_map, *plocal_reduce, world_rank);
    } else {
        InputMPIStream is(0, MAP_TAG);
        OutputMPIStream os(0, REDUCE_TAG);
        process(is, os, world_rank);
    }

    if (world_rank == 0) {
        std::vector<std::shared_ptr<std::istream>> iss;
        iss.push_back(plocal_reduce);
        for (int rank = 1; rank < world_size; ++rank) {
            iss.emplace_back(new InputMPIStream(rank, REDUCE_TAG));
        }

        std::vector<std::istream *> piss;
        for (int rank = 0; rank < world_size; ++rank) {
            piss.push_back(iss[rank].get());
        }

        result.reset(new return_type(reduce(piss)));
    }
    return result;
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function& f, Tuple t, std::index_sequence<I...>) {
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function& f, Tuple t) {
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

class TaskRegistry {
public:
    std::vector<int> all_num_threads_;
    TaskRegistry() {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
        all_num_threads_ = collect_num_threads();
    }

    static const int MAP_TAG = 13;
    static const int REDUCE_TAG = 14;

    template <typename Task, typename World>
    class Job {
    public:
        Job(TaskRegistry &task_registry, size_t job_id, World &world) : task_registry_{task_registry},
            world_{world}, job_id_{job_id} {}

        template <typename... Args>
        auto operator()(Args &&... args) {
            assert(task_registry_.world_rank_ == 0);
            Task task(std::forward<Args>(args)...);

            task_registry_.job_send(job_id_);
            std::stringstream ss;
            task.serialize(ss);
            std::string s = ss.str();
            broadcast_string(s);

            int world_size;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);

            auto plocal_map = std::make_shared<std::stringstream>();
            auto plocal_reduce = std::make_shared<std::stringstream>();
            {
                std::vector<std::shared_ptr<std::ostream>> oss;
                oss.push_back(plocal_map);
                for (size_t rank = 1; rank < world_size; ++rank) {
                    oss.emplace_back(new OutputMPIStream(rank, MAP_TAG));
                }

                const auto &all_num_threads_ = task_registry_.all_num_threads_;
                size_t sum_num_threads = std::accumulate(all_num_threads_.cbegin(), all_num_threads_.cend(), 0);
                assert(sum_num_threads > 0);
                std::cout << "All threads: " << sum_num_threads << std::endl;
                const size_t MULT = 1;
                auto make_splitter_args = std::tuple_cat(std::make_tuple(sum_num_threads * MULT), world_);
                auto make_splitter = [&](auto&&... ts) {return task.make_splitter(std::forward<decltype(ts)>(ts)...);};
                auto splitter = call(make_splitter, make_splitter_args);

                auto mult_splitter = [&](auto &os, int rank, size_t count) {
                    for (size_t i = 0; i < count; ++i) {
                        bool result = splitter(os, rank);
                        if (!result) return false;
                    }
                    return true;
                };

                for (size_t rank = 0; mult_splitter(*oss[rank], rank, all_num_threads_[rank]);
                     rank = (rank + 1) % world_size) {
                }
            } // close streams here

            auto process_args = std::tuple_cat(std::tuple<std::istream&, std::ostream&, int>(*plocal_map, *plocal_reduce, 0), world_);
            auto process = [&](auto&&... ts) {return task.process(std::forward<decltype(ts)>(ts)...);};
            call(process, process_args);

            std::vector<std::shared_ptr<std::istream>> iss;
            iss.push_back(plocal_reduce);
            for (int rank = 1; rank < world_size; ++rank) {
                iss.emplace_back(new InputMPIStream(rank, REDUCE_TAG));
            }

            std::vector<std::istream *> piss;
            for (int rank = 0; rank < world_size; ++rank) {
                piss.push_back(iss[rank].get());
            }

            auto reduce = [&](auto&&... ts) {return task.reduce(std::forward<decltype(ts)>(ts)...);};
            auto reduce_args = std::tuple_cat(std::make_tuple(piss), world_);
            return call(reduce, reduce_args);
        }

    private:
        TaskRegistry &task_registry_;
        World world_;
        size_t job_id_;
    };

    using process_function = std::function<void(std::istream &, std::ostream &, int)>;
    using init_function = std::function<process_function(std::istream &)>;

    template <typename Task, typename... WorldArgs>
    auto add(WorldArgs... world_args) {
        auto world = std::make_tuple(world_args...);  // unwrap ref/cref
        init_function init = [world](std::istream &is) -> process_function {
            std::shared_ptr<Task> ptask = std::make_shared<Task>(is);

            process_function process = [world, ptask = std::move(ptask)](std::istream &is, std::ostream &os,
                                                                                            int rank) -> void {
                auto args = std::tuple_cat(std::tuple<std::istream&, std::ostream&, int>(is, os, rank), world);
                auto pcall = [&](auto&&... ts) {return ptask->process(std::forward<decltype(ts)>(ts)...);};
                call(pcall, args);
            };

            return process;
        };

        size_t job_id = inits_.size();
        inits_.push_back(init);
        return Job<Task, decltype(world)>(*this, job_id, world);
    }

    init_function get_init(size_t id) { return inits_[id]; }

    void stop() {
        assert(world_rank_ == 0);
        job_send(size_t(-1));
    }

    int world_rank() const {
        return world_rank_;
    }

    void job_send(size_t job_id) {
        assert(world_rank_ == 0);
        MPI_Bcast(&job_id, sizeof(job_id), MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    void listen() {
        assert(world_rank_ != 0);
        std::cout << "Listening started" << std::endl;
        while (true) {
            size_t job_id;
            MPI_Bcast(&job_id, sizeof(job_id), MPI_CHAR, 0, MPI_COMM_WORLD);
            std::cout << "Job got" << std::endl;
            if (job_id == size_t(-1)) {
                return;
            }

            auto init = get_init(job_id);
            std::cout << "Initializer obtained" << std::endl;
            std::string s;
            broadcast_string(s);
            std::stringstream ss(s);
            auto process = init(ss);
            std::cout << "Processor initialized" << std::endl;
            InputMPIStream is(0, MAP_TAG);
            std::cout << "Input stream constructed" << std::endl;
            OutputMPIStream os(0, REDUCE_TAG);
            std::cout << "Output stream constructed" << std::endl;
            process(is, os, world_rank_);
        }
    }

    ~TaskRegistry() {
        if (world_rank_ == 0) stop();
    }

private:
    std::vector<init_function> inits_;
    int world_rank_;
};

}  // namespace partask
