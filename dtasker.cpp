#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <omp.h>

template <typename MakeSplitters, typename Process, typename Reduce>
auto run(MakeSplitters &make_splitters, Process &process, Reduce &reduce, size_t n) {
    auto splitters = make_splitters(n);

    std::vector<std::stringstream> oss(splitters.size());
    std::vector<std::istream *> piss;
    for (auto &ss : oss) {
        piss.push_back(&ss);
    }

#pragma omp parallel for
    for (size_t i = 0; i < splitters.size(); ++i) {
        auto &splitter = splitters[i];
        std::stringstream ss;
        bool more = false;
        do {
            more = splitter(ss, 0);
            process(ss, oss[i], 0);
        } while (more);
    }

    std::vector<size_t> nodes(oss.size(), 0);
    auto result = reduce(piss, nodes);

    return result;
}

const size_t N = 100000;
int data[N];
int main() {
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
        size_t begin, end;
        long long int sum = 0;
        while (is >> begin >> end) {
            for (size_t i = begin; i < end; ++i) {
                sum += data[i];
            }
        }
        os << sum;
    };

    auto reduce = [&](const std::vector<std::istream *> &piss, const std::vector<size_t> &nodes) {
        long long int sum = 0;
        for (auto &pis : piss) {
            long long int local_sum;
            *pis >> local_sum;
            sum += local_sum;
        }

        return sum;
    };

    auto result = run(make_splitters, process, reduce, 4);
    std::cout << result << std::endl;

    return 0;
}
