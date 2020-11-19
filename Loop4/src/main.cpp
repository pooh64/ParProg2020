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
        return sin(in);
}
#include "../../loop_common.h"

#if 0
void calc(double* arr, uint32_t zSize, uint32_t ySize, uint32_t xSize, int rank, int size)
{
	BCAST(&xSize, 0);
        BCAST(&ySize, 0);
	BCAST(&zSize, 0);

	for (uint32_t z = 1; z < zSize; z++) {
		for (uint32_t y = 0; y < ySize - 1; y++) {
			double *line_in  = &arr[(z-1)*ySize*xSize + (y+1)*xSize + 1];
			double *line_out = &arr[z*ySize*xSize + y*xSize];
			calc_line(rank, size, xSize-1, line_in, line_out);
		}
	}
}
#else

std::vector<double> g_restore_vec;

void process_save(double *arr, int xSize, int ySize)
{
	g_restore_vec.resize(ySize - 1 + xSize);
	for (int y = 0; y < ySize - 1; ++y)
		g_restore_vec[y] = arr[y*xSize + xSize - 1];
	for (int x = 0; x < xSize; ++x)
		g_restore_vec[ySize - 1 + x] = arr[ySize*(xSize-1) + x];
}

void process_restore(double *arr, int xSize, int ySize)
{
	for (int y = 0; y < ySize - 1; ++y)
		arr[y*xSize + xSize - 1] = g_restore_vec[y];
	for (int x = 0; x < xSize; ++x)
		arr[ySize*(xSize-1) + x] = g_restore_vec[ySize - 1 + x];
}

void process(int xSize)
{
        int ySize = g_self_len / xSize;
        for (int y = 0; y < ySize - 1; ++y) {
                for (int x = 0; x < xSize - 1; ++x)
                        g_task[y*xSize + x] = calc_elem(g_task[(y+1)*xSize + x+1]);
        }
}

void calc(double* arr, uint32_t zSize, uint32_t ySize, uint32_t xSize, int rank, int size)
{
	BCAST(&xSize, 0);
	BCAST(&ySize, 0);
	BCAST(&zSize, 0);

	calc_prep(rank, size, xSize * (ySize - 1), (int) xSize);

	for (uint32_t z = 1; z < zSize; ++z) {
		calc_tasks_pad(rank, size, (int) xSize);
		calc_scatter(&arr[(z-1)*ySize*xSize]);

		process((int) xSize);
		calc_tasks_pad(rank, size, -xSize);

		double *ptr = &arr[z*ySize*xSize];
		if (!rank) process_save(ptr, xSize, ySize);
		calc_gather(&arr[z*ySize*xSize]);
		if (!rank) process_restore(ptr, xSize, ySize);
	}
}
#endif

int main(int argc, char** argv)
{
  int rank = 0, size = 0, buf = 0;
  uint32_t zSize = 0, ySize = 0, xSize = 0;
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
    input >> zSize >> ySize >> xSize;
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    arr = new double[zSize * ySize * xSize];
    for (uint32_t z = 0; z < zSize; z++) {
      for (uint32_t y = 0; y < ySize; y++) {
        for (uint32_t x = 0; x < xSize; x++) {
          input >> arr[z*ySize*xSize + y*xSize + x];
        }
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

  calc(arr, zSize, ySize, xSize, rank, size);

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

    for (uint32_t z = 0; z < zSize; z++) {
      for (uint32_t y = 0; y < ySize; y++) {
        for (uint32_t x = 0; x < xSize; x++) {
          output << " " << arr[z*ySize*xSize + y*xSize + x];
        }
        output << std::endl;
      }
      output << std::endl;
    }
    output.close();
    delete arr;
  }

  MPI_Finalize();
  return 0;
}
