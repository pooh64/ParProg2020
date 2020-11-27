#include <iostream>
#include <iomanip>
#include <fstream>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <numeric>

#include "../../split_buf.h"

static inline
void calc_line(double *out, double *line_in[3], int len, double &diff)
{
	double tmp = 0;
	for (int x = 1; x < len - 1; ++x) {
		out[x] = (line_in[2][x]   + line_in[0][x] +
			  line_in[1][x+1] + line_in[1][x-1]) / 4.0;
		tmp += std::abs(out[x] - line_in[1][x]);
	}
	diff += tmp;
}

void calc_iter(double *out, double *in, double *border[2], int y_max, int x_max, double &diff)
{
	for (int y = 0; y < y_max; y++) {
		double *line_in[3];
		if (y == 0)
			line_in[0] = border[0];
		else
			line_in[0] = &in[x_max*(y-1)];

		if (y == y_max - 1)
			line_in[2] = border[1];
		else
			line_in[2] = &in[x_max*(y+1)];

		line_in[1] = &in[x_max*y];
		calc_line(&out[x_max*y], line_in, x_max, diff);
	}
}

void init_border(double *border[2], double *frame, int ySize, int y_self, int xSize, int rank, int size)
{
	int len = sizeof(*frame) * xSize;
	if (rank == 0)
		memcpy(border[0], frame, len);
	if (size == 1) {
		memcpy(border[1], &frame[xSize*(ySize-1)], len);
		return;
	}

	if (rank == 0)
		SEND(size-1, &frame[(ySize-1)*xSize], len);
	if (rank == size - 1) {
		if (!y_self)
			RECV(0, border[0], len);
		else
			RECV(0, border[1], len);
	}
}

void sync_border(double *border[2], double *buf, int y_self, int xSize, int rank, int size)
{
	if (size == 1)
		return;
	int len = sizeof(*buf) * xSize;

	if (!y_self) {
		if (rank != size - 1)
			RECV(rank+1, border[0], len);
		SEND(rank-1, border[0], len);
		
		RECV(rank-1, border[1], len); // discard
		if (rank != size - 1)
			SEND(rank+1, border[1], len); // dummy
		return;
	}

	if (rank != 0)
		SEND(rank-1, &buf[0], len);

	if (rank != size - 1) {
		RECV(rank+1, border[1], len);
		SEND(rank+1, &buf[(y_self-1)*xSize], len);
	}

	if (rank != 0)
		RECV(rank-1, border[0], len);
}

void calc(double *frame, uint32_t ySize, uint32_t xSize, double delta, int rank, int size)
{
	BCAST(&xSize, 0);
	BCAST(&ySize, 0);
	BCAST(&delta, 0);

	split_buf<double> task(rank, size);
	split_buf<double> diff(rank, size);
	double *border[2] = {new double[xSize], new double[xSize]};
	std::vector<double> diff_scbuf(!rank? size : 0);

	diff.set_each(1);
	task.split((ySize-2)*xSize, xSize);
	task.scatter(&frame[xSize]);
	std::vector<double> tmp(task.buf);
	int y_self = task.buf_len / xSize;
	init_border(border, frame, ySize, y_self, xSize, rank, size);

	do {
		*diff.get() = 0;
		sync_border(border, &task.buf[0], y_self, xSize, rank, size);
		calc_iter(&tmp[0], &task.buf[0], border, y_self, xSize, *diff.get());
		task.buf.swap(tmp);

		diff.gather(&diff_scbuf[0]);
		if (!rank) {
			double sum = std::accumulate(diff_scbuf.begin(), diff_scbuf.end(), 0.0);
			std::fill(diff_scbuf.begin(), diff_scbuf.end(), sum);
		}
		diff.scatter(&diff_scbuf[0]);
	} while (*diff.get() > delta);

	task.gather(&frame[xSize]);
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, status = 0;
  double delta = 0;
  uint32_t ySize = 0, xSize = 0;
  double* frame = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> ySize >> xSize >> delta;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    frame = new double[ySize * xSize];

    for (uint32_t y = 0; y < ySize; y++)
    {
     for (uint32_t x = 0; x < xSize; x++)
      {
        input >> frame[y*xSize + x];
      }
    }
    input.close();
  } else {
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0)
    {
      return 1;
    }
  }

  calc(frame, ySize, xSize, delta, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete frame;
      return 1;
    }
    for (uint32_t y = 0; y < ySize; y++)
    {
      for (uint32_t x = 0; x < xSize; x++)
      {
        output << " " << frame[y*xSize + x];
      }
      output << std::endl;
    }
    output.close();
    delete frame;
  }

  MPI_Finalize();
  return 0;
}
