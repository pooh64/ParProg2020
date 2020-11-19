#pragma once
#include <vector>

#define BCAST(ptr, root) MPI_Bcast((ptr), sizeof(*(ptr)), MPI_CHAR, (root), MPI_COMM_WORLD)

#define div_roundup(x, y) ({            \
        typeof(y) __y = y;              \
        ((x) + __y - 1) / __y; })

#define roundup(x, y) ({			\
	typeof(y) __y = y;			\
	(((x) + __y - 1) / __y) * __y; })

#define min(x, y) ({                    \
        typeof(x) __x = (x);            \
        typeof(y) __y = (y);            \
        (__x < __y) ? __x : __y;  })

int			g_task_sz;
int			g_self_len;
std::vector<double> 	g_task;
std::vector<int>	g_toff;
std::vector<int>	g_tlen;

void calc_truncate(int rank, int size, int len)
{
        if (!rank) {
                g_toff.resize(size);
		g_tlen.resize(size);
                for (int i = 0; i < size; ++i) {
                        g_toff[i] = min(g_task_sz * i, len);
                        g_tlen[i] = min(g_task_sz, len - g_toff[i]);
                }
        }
	g_self_len = min(g_task_sz, len - min(g_task_sz * rank, len));
        g_task.resize(g_self_len);
}

void calc_tasks_pad(int rank, int size, int pad)
{
	if (!rank) {
		for (int i = 0; i < size; ++i)
			g_tlen[i] += pad;
	}
	g_self_len += pad;
	g_task.resize(g_self_len);
}

void calc_prep(int rank, int size, int len, int align)
{
        g_task_sz = div_roundup(len, size);
	if (align)
		g_task_sz = roundup(g_task_sz, align);
	calc_truncate(rank, size, len);
}

void calc_scatter(double *arr)
{
	if (!g_self_len)
		return;
	MPI_Scatterv(arr, &g_tlen[0], &g_toff[0], MPI_DOUBLE,
			&g_task[0], g_self_len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void calc_process()
{
	for (int i = 0; i < g_self_len; ++i)
		g_task[i] = calc_elem(g_task[i]);
}

void calc_gather(double *arr)
{
	if (!g_self_len)
		return;
        MPI_Gatherv(&g_task[0], g_self_len, MPI_DOUBLE,
                        arr, &g_tlen[0], &g_toff[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void calc_line(int rank, int size, int len, double *arr_in, double *arr_out)
{
        calc_prep(rank, size, len, 0);
        calc_scatter(arr_in);
        calc_process();
        calc_gather(arr_out);
}
