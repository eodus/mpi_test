#pragma once

#include <mpi.h>
#include <omp.h>
#include <cassert>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

namespace partask {

bool init() {
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    return provided >= MPI_THREAD_MULTIPLE;
}

bool finalize() { return MPI_Finalize() == MPI_SUCCESS; }

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
    const size_t MAX_MPI_COUNT = std::numeric_limits<int>::max();
    assert(size <= MAX_MPI_COUNT);
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
        // #pragma omp critical(MPI)
        { MPI_Recv(&size, sizeof(size), MPI_CHAR, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
        std::vector<char> s;
        s.resize(size + 1);
        s[size] = '\0';
        // #pragma omp critical(MPI)
        { MPI_Recv(s.data(), size, MPI_CHAR, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
        this->str(s.data());
        std::cout << "Data recieved: " << this->str() << std::endl;
    }
};

template <typename MakeSplitters, typename Process, typename Reduce>
auto run(MakeSplitters &make_splitters, Process &process, Reduce &reduce) {
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    using return_type = decltype(reduce(std::vector<std::istream *>(), std::vector<int>()));
    std::unique_ptr<return_type> result;

    const int MAP_TAG = 13;
    const int REDUCE_TAG = 14;

    omp_set_num_threads(1);           // Section works even if we specify num_threads = 1
    omp_set_num_threads(world_size);  // Section works even if we specify num_threads = 1
#pragma omp sections
    {
        if (world_rank == 0) {
            auto splitters = make_splitters(world_size);
            assert(splitters.size() == world_size);
#pragma omp parallel for
            for (size_t rank = 0; rank < world_size; ++rank) {
                auto &splitter = splitters[rank];
                OutputMPIStream os(rank, MAP_TAG);
                bool more = false;
                do {
                    more = splitter(os, rank);
                } while (more);
            }
        }

#pragma omp section
        {
            InputMPIStream is(0, MAP_TAG);
            OutputMPIStream os(0, REDUCE_TAG);
            process(is, os, world_rank);
        }
#pragma omp section
        if (world_rank == 0) {
            std::vector<std::istream *> piss;
            std::vector<int> nodes;
#pragma omp parallel for
            for (int rank = 0; rank < world_size; ++rank) {
                auto pis = new InputMPIStream(rank, REDUCE_TAG);
#pragma omp critical
                {
                    piss.push_back(pis);
                    nodes.push_back(rank);
                }
            }

            result.reset(new return_type(reduce(piss, nodes)));
        }
    }
    return result;
}

}  // namespace partask
