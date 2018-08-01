#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include "dtasker.hpp"
#include "dtasker_mpi.hpp"

const size_t N = 100000;
int data[N];
int main(int argc, char *argv[]) {
    std::iota(data, data + N, 0);

    auto make_splitters = [&](size_t n) {
        std::vector<size_t> ends;
        for (size_t i = 0; i <= N; ++i) {
            ends.push_back(i * N / n);
        }

        std::vector<std::function<bool(std::ostream &, size_t)>> splitters;

        for (size_t i = 0; i < n; ++i) {
            size_t begin = ends[i], end = ends[i + 1];

            auto split_function = [begin, end](std::ostream &os, size_t node) -> bool {
                os << begin << " " << end;
                return false;
            };

            splitters.push_back(split_function);
        }

        return splitters;
    };

    auto process = [&](std::istream &is, std::ostream &os, size_t node) -> void {
        std::cout << "process run" << std::endl;
        long long int sum = 0;
        while (is.peek() != EOF) {
            size_t begin, end;
            is >> begin >> end;
            for (size_t i = begin; i < end; ++i) {
                sum += data[i];
            }
        }
        os << sum;
    };

    auto reduce = [&](const std::vector<std::istream *> &piss, const std::vector<int> &nodes) {
        long long int sum = 0;
        for (auto &pis : piss) {
            long long int local_sum;
            *pis >> local_sum;
            sum += local_sum;
        }

        return sum;
    };

    {
        auto result = run_omp(make_splitters, process, reduce, 4);
        std::cout << result << std::endl;
    }

    {
        // Initialize the MPI environment
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        std::cout << "Provided: " << provided << std::endl;
        // assert(provided >= MPI_THREAD_MULTIPLE);
        // Get the number of processes
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        auto presult = run_mpi(make_splitters, process, reduce, world_size);
        // Get the rank of the process
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        if (world_rank == 0) {
            std::cout << "MPI!!!!!!" << std::endl;
            std::cout << *presult << std::endl;
        }
        if (presult) {
            std::cout << "MPI!!!!!!" << std::endl;
            std::cout << *presult << std::endl;
        }
        // Finalize the MPI environment.
        MPI_Finalize();
    }

    return 0;
}
