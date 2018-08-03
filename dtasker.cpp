#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "partask_mpi.hpp"

class ArraySum {
public:
    ArraySum() = default;

    ArraySum(std::istream &) {}

    std::ostream &serialize(std::ostream &os) const { return os; }

    template <typename World>
    auto make_splitter(size_t n, World &world) {
        size_t N = world.size();
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

    template <typename World>
    void process(std::istream &is, std::ostream &os, int node, World &world) {
        std::cout << "process run" << std::endl;
        long long int sum = 0;
#pragma omp parallel reduction(+ : sum)
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
                sum += world[i];
            }
        }
        std::cout << "Computed sum: " << sum << std::endl;
        os << sum;
    }

    template <typename World>
    auto reduce(const std::vector<std::istream *> &piss, World &world) {
        long long int sum = 0;
        for (auto &pis : piss) {
            long long int local_sum;
            *pis >> local_sum;
            sum += local_sum;
        }

        return sum;
    };
};

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
#pragma omp parallel reduction(+ : sum)
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

    {
        partask::all_set_num_threads(1);

        partask::TaskRegistry reg;
        auto job = reg.add<ArraySum>(std::cref(data));
        // auto job = reg.add<partask::Task>(data);
        reg.listen();

        if (reg.master()) {
            auto res = job();
            std::cout << "JOB RESULT: " << res << std::endl;
            res = job();
            std::cout << "JOB RESULT: " << res << std::endl;
            res = job();
            std::cout << "JOB RESULT: " << res << std::endl;
            res = job();
            std::cout << "JOB RESULT: " << res << std::endl;
        }

        reg.stop();
        auto job2 = reg.add<ArraySum>(std::cref(data));
        reg.listen();
        if (reg.master()) {
            auto res = job();
            std::cout << "JOB RESULT: " << res << std::endl;
            res = job2();
            std::cout << "JOB RESULT: " << res << std::endl;
            res = job();
            std::cout << "JOB RESULT: " << res << std::endl;
            res = job();
            std::cout << "JOB RESULT: " << res << std::endl;
        }

        reg.stop();
        reg.stop();
        reg.stop();
    }

    // partask::all_set_num_threads(10);
    // auto nt = partask::collect_num_threads();
    // if (nt.size()) {
    //     std::cout << nt[0] << std::endl;
    // }
    // std::cout << "#threads after all_set: "<< omp_get_max_threads() << std::endl;
    // auto presult = partask::run(make_splitter, process, reduce);
    // if (presult) {
    //     // We are on master node
    //     std::cout << *presult << std::endl;
    // }

    partask::finalize();
    return 0;
}
