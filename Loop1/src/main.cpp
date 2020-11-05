#include <iostream>
#include <iomanip>
#include <fstream>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <unistd.h>
#include <cmath>
#include <vector>

#define BCAST(ptr, root) MPI_Bcast((ptr), sizeof(*(ptr)), MPI_CHAR, (root), MPI_COMM_WORLD)

#if 0
#define TRACE(fmt, ...) printf("[%d]: " fmt "\n", rank, ##__VA_ARGS__)
#else
#define TRACE(fmt, ...)
#endif

#define div_roundup(x, y) ({            \
        typeof(y) __y = y;              \
        (((x) + (__y - 1)) / __y); })

#define min(x, y) ({                    \
        typeof(x) __x = (x);            \
        typeof(y) __y = (y);            \
        (__x < __y) ? __x : __y;  })


void calc(double* arr, uint32_t ySize, uint32_t xSize, int rank, int size)
{
	BCAST(&xSize, 0);
	BCAST(&ySize, 0);
	int const len = xSize * ySize;
	int const task_sz = div_roundup(len, size);
	std::vector<double> task(task_sz);
	std::vector<int> offs_len;

	if (!rank) {
		offs_len.resize(2 * size);
		for (int i = 0; i < size; ++i) {
			offs_len[i] = min(task_sz * i, len);
			offs_len[i+size] = min(task_sz, len - offs_len[i]);
		}
	}
	TRACE("scattering");
	MPI_Scatterv(arr, &offs_len[size], &offs_len[0], MPI_DOUBLE,
			&task[0], task_sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	int truncated = min(task_sz, len - min(task_sz * rank, len));
	TRACE("truncated=%d", truncated);

	for (int i = 0; i < truncated; ++i)
		task[i] = sin(0.00001*task[i]);

	TRACE("gathering");
	MPI_Gatherv(&task[0], truncated, MPI_DOUBLE,
			arr, &offs_len[size], &offs_len[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

#if 0
	uint32_t len = xSize * ySize;
	for (uint32_t i = 0; i < len; ++i)
		arr[i] = sin(0.00001*arr[i]);
#endif
}

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
