#pragma once

#include <omp.h>
#include <sstream>
#include <vector>
#include <memory>

namespace partask {

bool init() {
    return true;
}

bool finalize() {
    return true;
}

template <typename T, typename Serialize, typename Deserialize>
void broadcast(T &data, Serialize &serialize, Deserialize &deserialize, int root = 0) {
    // Do nothing
}

template <typename MakeSplitters, typename Process, typename Reduce>
auto run(MakeSplitters &make_splitters, Process &process, Reduce &reduce) {
    size_t num_threads = omp_get_num_threads();
    auto splitters = make_splitters(num_threads);
    assert(splitters.size() == num_threads);

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
    using return_type = decltype(reduce(std::vector<std::istream *>(), std::vector<int>()));
    return new return_type(reduce(piss, nodes));
}

}  // namespace partask
