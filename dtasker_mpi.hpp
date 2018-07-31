#pragma once

#include <mpi.h>
#include <cassert>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>
#include <omp.h>

template <typename T, typename Serialize, typename Deserialize>
void broadcast_mpi(T &data, Serialize &serialize, Deserialize &deserialize, int root = 0) {
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
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
    MPI_Bcast(&size, sizeof(size), MPI_BYTE, root, MPI_COMM_WORLD);

    if (world_rank != root) {
        s.resize(size + 1);
    }

    // Broadcast object itself
    const size_t MAX_MPI_COUNT = std::numeric_limits<int>::max();
    assert(size <= MAX_MPI_COUNT);
    MPI_Bcast(s.data(), size + 1, MPI_BYTE, root, MPI_COMM_WORLD);

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
        MPI_Send(&size, sizeof(size), MPI_BYTE, rank_, tag_, MPI_COMM_WORLD);
        MPI_Send(const_cast<char *>(this->str().data()), this->str().size(), MPI_BYTE, rank_, tag_, MPI_COMM_WORLD);
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
        MPI_Recv(&size, sizeof(size), MPI_BYTE, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<char> s;
        s.resize(size + 1);
        s[size] = '\0';
        MPI_Recv(s.data(), size, MPI_BYTE, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        this->str(s.data());
    }
};

template <typename MakeSplitters, typename Process, typename Reduce>
auto run_mpi(MakeSplitters &make_splitters, Process &process, Reduce &reduce, size_t n) {
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    assert(n <= world_size);
    using return_type = decltype(reduce(std::vector<std::istream *>(), std::vector<int>()));
    std::unique_ptr<return_type> result;

    const int MAP_TAG = 13;
    const int REDUCE_TAG = 14;

    omp_set_num_threads(1);  // Section works even if we specify num_threads = 1
#pragma omp sections
    {
        if (world_rank == 0) {
            auto splitters = make_splitters(n);
#pragma omp parallel
            for (size_t rank = 0; rank < splitters.size(); ++rank) {
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
            for (int rank = 0; rank < n; ++rank) {
                piss.push_back(new InputMPIStream(rank, REDUCE_TAG));
                nodes.push_back(rank);
            }

            result.reset(new return_type(reduce(piss, nodes)));
        }
    }
    return result;
}
