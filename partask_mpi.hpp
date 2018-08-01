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

template <typename MakeSplitter, typename Process, typename Reduce>
auto run(MakeSplitter &make_splitter, Process &process, Reduce &reduce) {
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    using return_type = decltype(reduce(std::vector<std::istream *>()));
    std::unique_ptr<return_type> result;

    const int MAP_TAG = 13;
    const int REDUCE_TAG = 14;

    // omp_set_num_threads(1);           // Section works even if we specify num_threads = 1
    auto plocal_map = std::make_shared<std::stringstream>();
    auto plocal_reduce = std::make_shared<std::stringstream>();
    if (world_rank == 0) {
        std::vector<std::shared_ptr<std::ostream>> oss;
        oss.push_back(plocal_map);
        for (size_t rank = 1; rank < world_size; ++rank) {
            oss.emplace_back(new OutputMPIStream(rank, MAP_TAG));
        }
        auto splitter = make_splitter(world_size * 10);
        for (size_t rank = 0; splitter(*oss[rank], rank); rank = (rank + 1) % world_size) { }
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
