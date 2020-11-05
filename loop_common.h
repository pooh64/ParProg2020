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


static inline
double calc_elem(double in)
{
        return sin(0.00001*in);
}

static
void calc_line(double *arr_in, double *arr_out, int len, int rank, int size)
{
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
        int truncated = min(task_sz, len - min(task_sz * rank, len));
        if (!truncated)
                return;
        MPI_Scatterv(arr_in, &offs_len[size], &offs_len[0], MPI_DOUBLE,
                        &task[0], task_sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int i = 0; i < truncated; ++i)
                task[i] = calc_elem(task[i]);

        MPI_Gatherv(&task[0], truncated, MPI_DOUBLE,
                        arr_out, &offs_len[size], &offs_len[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
