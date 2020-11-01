#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>

#define CACHE_LINE_SZ 64

#define rounddown(x, y) ({	\
	typeof(x) __x = (x);	\
	__x - (__x % (y)); })	\

#define div_roundup(x, y) ({		\
	typeof(y) __y = y;		\
	(((x) + (__y - 1)) / __y); })

#define min(x, y) ({			\
	typeof(x) __x = (x);		\
	typeof(y) __y = (y);		\
	__x = (__x < __y) ? __x : __y;	})

static inline
uint8_t sum_near(uint8_t *line[3], int ind1, int ind2, int ind3)
{
	uint8_t sum = 0;
	for (int l = 0; l < 3; ++l) {
		sum += line[l][ind1];
		sum += line[l][ind2];
		sum += line[l][ind3];
	}
	return sum;
}

static inline
uint8_t upd_cell(uint8_t in, uint8_t sum)
{
	if (!in && sum == 3)
		return 1;
	if (in && (sum == 3 || sum == 4))
		return 1;
	return 0;
}

static inline
int modulo(int x, int y)
{
	return (x % y + y) % y;
}

static inline void
eval_line(uint8_t *lin[3], uint8_t *lout[3], int len, int beg, int end)
{
#define EVAL_ITER(i, ...) do {							\
		lout[1][i] = upd_cell(lin[1][i], sum_near(lin, __VA_ARGS__));	\
	} while (0)

	bool proc_last = 0;
	if (end == len) {
		proc_last = 1;
		--end;
	}

	if (beg == 0) {
		EVAL_ITER(0, len - 1, 0, modulo(1, len));
		++beg;
	}
	for (int i = beg; i < end; ++i)
		EVAL_ITER(i, i - 1, i, i + 1);
	if (proc_last)
		EVAL_ITER(len - 1, modulo(len - 2, len), len - 1, 0);
#undef EVAL_ITER
}

void calc_iter(int len, int h, uint32_t num_threads, uint8_t *in_fr, uint8_t *out_fr)
{
#if 0
	#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < h; ++i) {
		uint8_t *lin[3], *lout[3];
		int ind[3] = { modulo(i - 1, h), i, modulo(i + 1, h) };
		for (int n = 0; n < 3; ++n) {
			lin[n]  = in_fr  + len * ind[n];
			lout[n] = out_fr + len * ind[n];
		}
		eval_line(lin, lout, len, 0, len);
	}
#else
#define BATCH_SZ (CACHE_LINE_SZ * 16)
	int total_sz = h * len;
	int n_batch = div_roundup(total_sz, BATCH_SZ);

#pragma omp parallel for num_threads(num_threads) proc_bind(close) schedule(static, 1)
	for (int b = 0; b < n_batch; ++b) {
		int beg = b * BATCH_SZ;
		int end = min(beg + BATCH_SZ, total_sz);
		for (int p = beg; p != end;) {
			int next = min(rounddown(p, len) + len, end);
			int y = p / len;
			uint8_t *lin[3], *lout[3];
			int ind[3] = { modulo(y - 1, h), y, modulo(y + 1, h) };
			for (int n = 0; n < 3; ++n) {
				lin[n]  = in_fr  + len * ind[n];
				lout[n] = out_fr + len * ind[n];
			}
			eval_line(lin, lout, len, p - y * len, next - y * len);
			p = next;
		}
	}
#endif
}

void calc(uint32_t xSize, uint32_t ySize, uint32_t iterations,
		uint32_t num_threads, uint8_t *&inFrame, uint8_t *&outFrame)
{
	int len = (int) xSize;
	int h = (int) ySize;
	int iter = (int) iterations;

	for (int i = 0; i < iter; ++i) {
		calc_iter(len, h, num_threads, inFrame, outFrame);
		std::swap(inFrame, outFrame);
	}
	std::swap(inFrame, outFrame);
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
  uint32_t xSize = 0, ySize = 0, iterations = 0, num_threads = 0;
  input >> xSize >> ySize >> iterations >> num_threads;
  uint8_t* inputFrame =  (uint8_t*) aligned_alloc(CACHE_LINE_SZ, xSize*ySize);
  uint8_t* outputFrame = (uint8_t*) aligned_alloc(CACHE_LINE_SZ, xSize*ySize);
  for (uint32_t y = 0; y < ySize; y++)
  {
    for (uint32_t x = 0; x < xSize; x++)
    {
      input >> inputFrame[y*xSize + x];
      inputFrame[y*xSize + x] -= '0';
    }
  }

  // Calculation
  calc(xSize, ySize, iterations, num_threads, inputFrame, outputFrame);

  // Write result
  for (uint32_t y = 0; y < ySize; y++)
  {
    for (uint32_t x = 0; x < xSize; x++)
    {
      outputFrame[y*xSize + x] += '0';
      output << " " << outputFrame[y*xSize + x];
    }
    output << "\n";
  }

  // Prepare to exit
  delete outputFrame;
  delete inputFrame;
  output.close();
  input.close();
  return 0;
}
