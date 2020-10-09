#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <cmath>

inline double func(double x)
{
  return sin(x);
}

#include <assert.h>
struct __attribute__((aligned(64))) double_al {
	double val;
};

#define DIV_ROUND_UP(a, b) ({	\
	typeof((a)) __a = (a);	\
	typeof((b)) __b = (b);	\
	(__a + __b - 1) / __b;})

inline double __attribute__((optimize("O0")))
kahan_add(double sum, double v)
{
	double c = 0;
	double y = v - c;
	double t = sum + y;
	c = (t - sum) - y;
	return t;
}

#define KAHAN

double calc_thread(double x0, uint64_t s_beg, uint64_t s_end, double dx)
{
	if (s_beg > s_end)
		return 0;

	double res = 0;
	s_end++;
	for (uint64_t s = s_beg; s < s_end; ++s) {
#ifdef KAHAN
		res = kahan_add(res, func(x0 + s * dx));
#else
		res += func(x0 + s * dx);
#endif
	}
	double tmp = -0.5 * (func(x0 + s_beg * dx) + func(x0 + (--s_end) * dx));
#ifdef KAHAN
	res = kahan_add(res, tmp);
#else
	res += tmp;
#endif
	return res * dx;
}

/* slow, but needed only for last step */
double calc_step(double x0, double x1)
{
	return 0.5 * (func(x0) + func(x1)) * (x1 - x0);
}

double __attribute__((optimize("O0"))) 
kahan_sum(double_al *arr, size_t arr_sz) {
	double sum = 0;
	double c = 0;
	double y, t;
	for (size_t i = 0; i < arr_sz; ++i) {
		y = arr[i].val - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}
	return sum;
}

double calc(double x0, double x1, double dx, uint32_t num_threads)
{
	double_al *arr = (double_al*) malloc(sizeof(*arr) * num_threads);
	assert(arr);

	uint64_t n_steps = (x1 - x0) / dx;
	uint64_t batch_sz = DIV_ROUND_UP(n_steps, num_threads);

	#pragma omp parallel num_threads(num_threads)
	{
		int tid = omp_get_thread_num();

		uint64_t s_beg = batch_sz * tid;
		uint64_t s_end = ((uint32_t) tid == num_threads - 1) ? 
			n_steps: s_beg + batch_sz;
		arr[tid].val = calc_thread(x0, s_beg, s_end, dx);
	}

	double res = calc_step(x0 + n_steps * dx, x1);
	for (uint32_t i = 0; i < num_threads; ++i)
		res = kahan_add(res, arr[i].val);
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
  double x0 = 0.0, x1 =0.0, dx = 0.0;
  uint32_t num_threads = 0;
  input >> x0 >> x1 >> dx >> num_threads;

  // Calculation
  double res = calc(x0, x1, dx, num_threads);

  // Write result
  output << std::setprecision(13) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
