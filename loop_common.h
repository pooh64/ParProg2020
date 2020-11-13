#pragma once
#include <vector>

#define BCAST(ptr, root) MPI_Bcast((ptr), sizeof(*(ptr)), MPI_CHAR, (root), MPI_COMM_WORLD)

#define div_roundup(x, y) ({            \
        typeof(y) __y = y;              \
        (((x) + (__y - 1)) / __y); })

#define min(x, y) ({                    \
        typeof(x) __x = (x);            \
        typeof(y) __y = (y);            \
        (__x < __y) ? __x : __y;  })

int			g_task_sz;
std::vector<double> 	g_task;
std::vector<int>	g_toff;
std::vector<int>	g_tlen;

static
void calc_truncate(int len, int rank, int size)
{
        if (!rank) {
                g_toff.resize(size);
		g_tlen.resize(size);
                for (int i = 0; i < size; ++i) {
                        g_toff[i] = min(g_task_sz * i, len);
                        g_tlen[i] = min(g_task_sz, len - g_toff[i]);
                }
        }
}

static
void calc_prep(int len, int rank, int size)
{
        g_task_sz = div_roundup(len, size);
        g_task.resize(g_task_sz);
	calc_truncate(len, rank, size);
}

static
void calc_scatter(int len, int rank, int size, double *arr)
{
	(void) size;
	int truncated = min(g_task_sz, len - min(g_task_sz * rank, len));
	if (!truncated)
		return;
	MPI_Scatterv(arr, &g_tlen[0], &g_toff[0], MPI_DOUBLE,
			&g_task[0], g_task_sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

static
void calc_process(int len, int rank)
{
	int truncated = min(g_task_sz, len - min(g_task_sz * rank, len));
        for (int i = 0; i < truncated; ++i)
                g_task[i] = calc_elem(g_task[i]);
}

static
void calc_gather(int len, int rank, int size, double *arr)
{
	(void) size;
	int truncated = min(g_task_sz, len - min(g_task_sz * rank, len));
	if (!truncated)
		return;
        MPI_Gatherv(&g_task[0], truncated, MPI_DOUBLE,
                        arr, &g_tlen[0], &g_toff[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
