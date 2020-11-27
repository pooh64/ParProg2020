#pragma once
#include <vector>

#define BCAST(ptr, root) MPI_Bcast((ptr), sizeof(*(ptr)), MPI_CHAR, (root), MPI_COMM_WORLD)

#define SEND(dest, buf, size) MPI_Send(buf, size, MPI_CHAR, dest, 0, MPI_COMM_WORLD)
#define RECV(src,  buf, size) MPI_Send(buf, size, MPI_CHAR, src,  0, MPI_COMM_WORLD)

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

template<typename T>
struct split_buf {
public:
	int			buf_len;
	int			buf_off;
	std::vector<T>		buf;

	std::vector<int>	boff;
	std::vector<int>	blen;
private:
	int			buf_sz;
	MPI_Datatype dtype;
	int rank, size;
public:

	split_buf(int _rank, int _size)
		: rank(_rank), size(_size)
	{
		MPI_Type_contiguous(sizeof(T), MPI_CHAR, &dtype);
		MPI_Type_commit(&dtype);
		if (!rank) {
			boff.resize(size);
			blen.resize(size);
		}
	}
	T *get()
	{
		return &buf[0];
	}
	void truncate_len(int len)
	{
		if (!rank) {
			for (int i = 0; i < size; ++i) {
				boff[i] = min(buf_sz * i, len);
				blen[i] = min(buf_sz, len - boff[i]);
			}
		}
		buf_off = min(buf_sz * rank, len); 
		buf_len = min(buf_sz, len - buf_off);
		buf.resize(buf_len);
	}
	void pad_each(int pad)
	{
		if (!rank) {
			for (int i = 0; i < size; ++i)
				blen[i] += pad;
		}
		buf_len += pad;
		buf.resize(buf_len);
	}
	void split(int len, int align)
	{
		buf_sz = div_roundup(len, size);
		if (align)
			buf_sz = roundup(buf_sz, align);
		truncate_len(len);
	}
	void set_each(int _buf_sz)
	{
		buf_sz = _buf_sz;
		truncate_len(buf_sz * size);
	}
	void scatter(T *arr)
	{
		if (!buf_len)
			return;
		MPI_Scatterv(arr, &blen[0], &boff[0], dtype,
				&buf[0], buf_len, dtype, 0, MPI_COMM_WORLD);
	}
	void gather(T *arr)
	{
		if (!buf_len)
			return;
		MPI_Gatherv(&buf[0], buf_len, dtype,
				arr, &blen[0], &boff[0], dtype, 0, MPI_COMM_WORLD);
	}
};
