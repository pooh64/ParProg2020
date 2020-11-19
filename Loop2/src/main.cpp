#include <iostream>
#include <iomanip>
#include <fstream>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <unistd.h>
#include <cmath>

static inline
double calc_elem(double in)
{
        return sin(0.00001*in);
}
#include "../../loop_common.h"

#if 0
void calc(double *arr, uint32_t ySize, uint32_t xSize, int rank, int size)
{
	BCAST(&xSize, 0);
	BCAST(&ySize, 0);

	for (uint32_t y = 0; y < ySize - 1; y++) {
		double *line_in  = &arr[(y+1)*xSize];
		double *line_out = &arr[y*xSize + 3];
		calc_line(rank, size, xSize - 3, line_in, line_out);
	}
}
#else
void process(int xSize)
{
	int ySize = g_self_len / xSize;

	for (int y = 0; y < ySize - 1; ++y) {
		for (int x = 3; x < xSize; ++x)
			g_task[y*xSize + x] = calc_elem(g_task[(y+1)*xSize + x-3]);
	}
}

void calc(double *arr, uint32_t ySize, uint32_t xSize, int rank, int size)
{
	BCAST(&xSize, 0);
	BCAST(&ySize, 0);
	
	calc_prep(rank, size, xSize * (ySize - 1), (int) xSize);
	calc_tasks_pad(rank, size, (int) xSize);
	calc_scatter(arr);

	process((int) xSize);
	calc_tasks_pad(rank, size, -xSize);
	calc_gather(arr);
}
#endif

int main(int argc, char** argv)
{
  int rank = 0, size = 0, buf = 0;
  uint32_t ySize = 0, xSize = 0;
  double* arr = 0;

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
    input >> ySize >> xSize;
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    arr = new double[ySize * xSize];

    for (uint32_t y = 0; y < ySize; y++)
    {
     for (uint32_t x = 0; x < xSize; x++)
      {
        input >> arr[y*xSize + x];
      }
    }
    input.close();
  } else {
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (buf != 0)
    {
      return 1;
    }
  }

  calc(arr, ySize, xSize, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete arr;
      return 1;
    }
    for (uint32_t y = 0; y < ySize; y++)
    {
      for (uint32_t x = 0; x < xSize; x++)
      {
        output << " " << arr[y*xSize + x];
      }
      output << std::endl;
    }
    output.close();
    delete arr;
  }

  MPI_Finalize();
  return 0;
}
