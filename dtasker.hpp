#pragma once

#include <sstream>
#include <vector>

template <typename MakeSplitters, typename Process, typename Reduce>
auto run_omp(MakeSplitters &make_splitters, Process &process, Reduce &reduce, size_t n) {
    auto splitters = make_splitters(n);

    std::vector<std::stringstream> oss(splitters.size());

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

    std::vector<std::istream *> piss;
    for (auto &ss : oss) {
        piss.push_back(&ss);
    }
    std::vector<int> nodes(oss.size(), 0);
    auto result = reduce(piss, nodes);

    return result;
}
