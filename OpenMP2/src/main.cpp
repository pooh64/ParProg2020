#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>

#include <assert.h>
struct __attribute__((aligned(64))) double_al {
	double val;
};

#define DIV_ROUND_UP(a, b) ({	\
	typeof((a)) __a = (a);	\
	typeof((b)) __b = (b);	\
	(__a + __b - 1) / __b;})

#if 0
double calc_thread(uint32_t x_beg, uint32_t x_end)
{
	double res = 0;
	for (uint32_t i = x_end - 1; i >= x_beg; --i)
		res += (1.0 / i);
	return res;
}
#else
/* kahan-sum alternative: */
#define N_BATCH 64
double calc_thread(uint32_t x_beg, uint32_t x_end)
{
	if (x_beg >= x_end)
		return 0;

	uint32_t batch_sz = DIV_ROUND_UP((x_end - x_beg), N_BATCH);
	double res = 0;
	uint32_t tbeg = x_beg + batch_sz * (N_BATCH - 1);

	for (uint32_t b = 0; b < N_BATCH; ++b) {
		double tres = 0;
		uint32_t tend = tbeg + batch_sz;
		tend = (tend > x_end) ? x_end : tend;

		for (uint32_t i = tend - 1; i >= tbeg; --i)
			tres += (1.0 / i);

		tbeg -= batch_sz;
		res += tres;
	}
	return res;
}
#endif

double calc(uint32_t x_last, uint32_t num_threads)
{
	double_al *arr = (double_al*) malloc(sizeof(*arr) * num_threads);
	assert(arr);

	uint32_t batch_sz = DIV_ROUND_UP(x_last, num_threads);

	#pragma omp parallel num_threads(num_threads)
	{
		int tid = omp_get_thread_num();

		uint32_t x_beg = 1 + batch_sz * tid;
		uint32_t x_end = ((uint32_t) tid == num_threads - 1) ? 
			x_last + 1 : x_beg + batch_sz;
		arr[tid].val = calc_thread(x_beg, x_end);
	}

	double res = 0;
	for (int32_t i = num_threads; i >= 0; --i)
		res += arr[i].val;

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
  output << std::setprecision(15) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
