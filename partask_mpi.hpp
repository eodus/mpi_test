#pragma once

#include <mpi.h>
#include <omp.h>
#include <cassert>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>

namespace partask {

const size_t MPI_MAX_COUNT = std::numeric_limits<int>::max();

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
    MPI_Scatter(const_cast<int*>(all_num_threads.data()), 1, MPI_INT, &num_threads, 1, MPI_INT, root, MPI_COMM_WORLD);
    omp_set_num_threads(num_threads);
}

void all_set_num_threads(int num_threads, int root = 0) {
    omp_set_num_threads(num_threads);
}

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

}  // namespace partask
