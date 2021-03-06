#include <mpi.h>
#include <stdio.h>
#include "dtasker_mpi.hpp"

int main(int argc, char **argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);

    // Find out rank, size
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int number;
    if (world_rank == 0) {
        number = -127;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (world_rank == 1) {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received number %d from process 0\n", number);
    }

    std::string s;
    if (world_rank == 0) {
        s = "TESTTEST TEEEEEEEEST";
    }

    auto serialize = [](auto &os, const auto &s) { os << s; };
    auto deserialize = [](auto &is, auto &s) { is >> s; };

    broadcast_mpi(s, serialize, deserialize, 0);

    std::cout << "===================================\n";

    std::cout << world_rank << " " << s << std::endl;

    // Finalize the MPI environment.
    MPI_Finalize();
}
