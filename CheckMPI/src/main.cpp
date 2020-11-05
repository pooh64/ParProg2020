#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>


int main(int argc, char** argv)
{
  int rank = 0;
  int size = 0;
  int buf = 0;
  MPI_Status status;
  std::string str;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> str;
    input.close();
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

  } else {
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (buf != 0)
    {
      return 1;
    }
  }

  if (rank != 0)
  {
    MPI_Recv(&status, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  }

  sleep(1);

  if (rank != size - 1)
  {
    MPI_Send(&status, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  }

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      return 1;
    }
    output << str << std::endl;
    output.close();
  }

  MPI_Finalize();
  return 0;
}
