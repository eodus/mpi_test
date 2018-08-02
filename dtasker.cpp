#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "partask_mpi.hpp"

const size_t N = 100000;
std::array<int, N> data;
int main(int argc, char *argv[]) {
    std::iota(data.begin(), data.end(), 1);
    size_t sum = std::accumulate(data.cbegin(), data.cend(), size_t(0));
    std::cout << "Actual sum: " << sum << std::endl;

    auto make_splitter = [&](size_t n) {
        auto splitter = [N = N, n = n, i = size_t(0)](std::ostream &os, size_t node) mutable -> bool {
            if (i == n) return false;
            size_t begin = i * N / n;
            size_t end = (i + 1) * N / n;
            ++i;
            os << begin << " " << end << " ";
            return true;
        };

        return splitter;
    };

    auto process = [&](std::istream &is, std::ostream &os, size_t node) -> void {
        std::cout << "process run" << std::endl;
        long long int sum = 0;
#pragma omp parallel reduction (+:sum)
        {
            while (true) {
                size_t begin, end;
                bool exit = false;
#pragma omp critical
                {
                    if (is.peek() == EOF || !(is >> begin >> end)) exit = true;
                    if (!exit) std::cout << "Extracted range: " << begin << " " << end << std::endl;
                }
                if (exit) break;
                for (size_t i = begin; i < end; ++i) {
                    sum += data[i];
                }
            }
        }
        std::cout << "Computed sum: " << sum << std::endl;
        os << sum;
    };

    auto reduce = [&](const std::vector<std::istream *> &piss) {
        long long int sum = 0;
        for (auto &pis : piss) {
            long long int local_sum;
            *pis >> local_sum;
            sum += local_sum;
        }

        return sum;
    };

    partask::init();
    auto presult = partask::run(make_splitter, process, reduce);
    partask::finalize();

    if (presult) {
        // We are on master node
        std::cout << *presult << std::endl;
    }

    return 0;
}
