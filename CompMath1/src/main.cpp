#include <iostream>
#include <iomanip>
#include <fstream>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <unistd.h>
#include <cmath>
#include <cassert>

#include "../../split_buf.h"

static inline
double acceleration(double t)
{
  return sin(t);
}

struct calc_pass_data {
	double y0;
	double v0;
};

void calc_process(double *arr, int len, struct calc_pass_data &data, double t0, double dt)
{
	assert(len > 2);
	for (int i = 0; i < 2; ++i)
		arr[i] = 0;

	for (int i = 2; i < len; ++i)
		arr[i] = dt*dt*acceleration(t0+(i-1)*dt) + 2*arr[i-1] - arr[i-2];

	data.y0 = arr[len-1];
	//data.v0 = (arr[len-1] - arr[len-2]) / dt;
	data.v0 = (3*arr[len-1] - 4*arr[len-2] + arr[len-3]) / (2*dt);
}

void calc_aim(split_buf<double> &task, std::vector<calc_pass_data> &pass_buf,
		double y0, double y1, double dt, int traceSize)
{
	double v0 = 0;
	calc_pass_data pd = pass_buf[0];
	pass_buf[0].y0 = y0;
	pass_buf[0].v0 = 0;
	for (unsigned i = 1; i < pass_buf.size(); ++i) {
		calc_pass_data tmp = pass_buf[i];
		pass_buf[i].v0 = pass_buf[i-1].v0 + pd.v0;
		pass_buf[i].y0 = pass_buf[i-1].y0 + pd.y0 + pass_buf[i-1].v0 * task.blen[i-1] * dt;
		pd = tmp;
	}
	unsigned last = pass_buf.size() - 1;
	double y_pr = pass_buf[last].y0 + pd.y0 + pass_buf[last].v0 * task.blen[last] * dt;
	v0 = (y1 - y_pr) / (dt * traceSize);
	pass_buf[0].v0 = v0;
	double y_delta = 0;
	for (unsigned i = 1; i < pass_buf.size(); ++i) {
		pass_buf[i].v0 += v0;
		y_delta += v0 * task.blen[i-1] * dt;
		pass_buf[i].y0 += y_delta;
	}
}

void calc_correct(double *arr, int len, struct calc_pass_data data, double dt)
{
	data.v0 *= dt;
	for (int i = 0; i < len; ++i)
		arr[i] += data.y0 + data.v0 * i;
}

void calc(double *trace, uint32_t traceSize, double t0, double dt, double y0, double y1, int rank, int size)
{
	BCAST(&traceSize, 0);
	BCAST(&t0, 0);
	BCAST(&dt, 0);

	split_buf<double>		task(rank, size);
	split_buf<calc_pass_data>	pass(rank, size);
	std::vector<calc_pass_data>	pass_buf(size);

	task.split(traceSize, 0);
	pass.set_each(1);

	calc_process(task.get(), task.buf_len, *pass.get(), t0+task.buf_off*dt, dt);

	pass.gather(&pass_buf[0]);
	if (!rank)
		calc_aim(task, pass_buf, y0, y1, dt, traceSize);
	pass.scatter(&pass_buf[0]);

	calc_correct(task.get(), task.buf_len, *pass.get(), dt);
	task.gather(trace);
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, status = 0;
  uint32_t traceSize = 0;
  double t0 = 0, t1 = 0, dt = 0, y0 = 0, y1 = 0;
  double* trace = 0;

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
    input >> t0 >> t1 >> dt >> y0 >> y1;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    traceSize = (t1 - t0)/dt;
    trace = new double[traceSize];

    input.close();
  } else {
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0)
    {
      return 1;
    }
  }

  calc(trace, traceSize, t0, dt, y0, y1, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete trace;
      return 1;
    }

    for (uint32_t i = 0; i < traceSize; i++)
    {
      output << " " << trace[i];
    }
    output << std::endl;
    output.close();
    delete trace;
  }

  MPI_Finalize();
  return 0;
}
