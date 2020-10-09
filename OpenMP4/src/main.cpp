#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>

#include <assert.h>
struct __attribute__((aligned(64))) calc_info {
	double sum;
	double last;
};

#define DIV_ROUND_UP(a, b) ({	\
	typeof((a)) __a = (a);	\
	typeof((b)) __b = (b);	\
	(__a + __b - 1) / __b;})

calc_info calc_thread(uint32_t x_beg, uint32_t x_end)
{
	calc_info cinf;
	if (x_end <= x_beg) {
		cinf.sum = 0;
		cinf.last = 0;
		return cinf;
	};
	double term = 1.0 / x_beg;
	double sum = term;
	for (uint32_t i = x_beg + 1; i < x_end; ++i) {
		term /= i;
		sum += term;
	}
	cinf.sum = sum;
	cinf.last = term;
	return cinf;
}

double calc(uint32_t x_last, uint32_t num_threads)
{
	if (!num_threads)
		return 0;

	calc_info *arr = (calc_info*) malloc(sizeof(*arr) * num_threads);
	assert(arr);

	uint32_t batch_sz = DIV_ROUND_UP(x_last, num_threads);

	#pragma omp parallel num_threads(num_threads)
	{
		int tid = omp_get_thread_num();

		uint32_t x_beg = 1 + batch_sz * tid;
		uint32_t x_end = x_beg + batch_sz;
		x_end = (x_end > x_last) ? x_last : x_end;
		arr[tid] = calc_thread(x_beg, x_end);
#pragma omp critical
		printf("[%d] %8.8u %8.8u %8.8lg %8.8lg\n", tid, x_beg, x_end, arr[tid].sum, arr[tid].last);
	}

	double res = 1.0 + arr[0].sum;
	double multp = 1;
	for (uint32_t i = 1; i < num_threads; ++i) {
		multp *= arr[i-1].last;
		res += arr[i].sum * multp;
	}

	free(arr);
	return res;
}

int main(int argc, char** argv)
{
  // Check arguments
  if (argc != 3)
  {
    std::cout << "[Error] Usage <inputfile> <output file>\n";
    return 1;
  }

  // Prepare input file
  std::ifstream input(argv[1]);
  if (!input.is_open())
  {
    std::cout << "[Error] Can't open " << argv[1] << " for write\n";
    return 1;
  }

  // Prepare output file
  std::ofstream output(argv[2]);
  if (!output.is_open())
  {
    std::cout << "[Error] Can't open " << argv[2] << " for read\n";
    input.close();
    return 1;
  }

// Read arguments from input
  uint32_t x_last = 0, num_threads = 0;
  input >> x_last >> num_threads;

  // Calculation
  double res = calc(x_last, num_threads);

  // Write result
  output << std::setprecision(16) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
