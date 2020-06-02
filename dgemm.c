#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <mkl.h>

double mysecond()
{
	struct timeval tp;
	struct timezone tzp;
	int i;
	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void multiply_matrices(double *C, double *A, double *B, const int dim_M, const int dim_N, const int dim_K)
{

	memset(C, 0, dim_M * dim_N * sizeof(*C));
#pragma omp parallel for
	for (int i = 0 ; i < dim_M ; i++) {
		for (int k = 0 ; k < dim_K ; k++) {
			for (int j = 0 ; j < dim_N ; j++) {
				C[i*dim_N+j] += A[i*dim_K+k] * B[k*dim_N+j];
			}
		}
	}
}

int main(int argc, char *argv[])
{
	if (argc != 4) {
		printf("Usage: %s [M] [K] [N]\n", argv[0]);
		exit(1);
	}

	int A_dims[] = {atoi(argv[1]), atoi(argv[2])};
	int B_dims[] = {atoi(argv[2]), atoi(argv[3])};
	double *A = (double*)malloc(sizeof(double) * (A_dims[0] * A_dims[1]));
	assert(A != NULL);
	double *B = (double*)malloc(sizeof(double) * (B_dims[0] * B_dims[1]));
	assert(B != NULL);
	double *C = (double*)malloc(sizeof(double) * (A_dims[0] * B_dims[1]));
	assert(C != NULL);
	double *C_nav = (double*)calloc(sizeof(double), (A_dims[0] * B_dims[1]));
	assert(C_nav != NULL);

	double start_random = mysecond();
	for (int i = 0; i < A_dims[0] * A_dims[1]; i++) {
		A[i] = (double)rand() / (double)RAND_MAX;
	}
	for (int i = 0; i < B_dims[0] * B_dims[1]; i++) {
		B[i] = (double)rand() / (double)RAND_MAX;
	}
	double stop_random = mysecond();

	double start_blas = mysecond();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                        A_dims[0], B_dims[1], A_dims[1],
                                        1.0, A, A_dims[1], B, B_dims[1],
                                        0.0, C, B_dims[1]);
	double stop_blas = mysecond();

	double start_naive = mysecond();
	multiply_matrices(C_nav, A, B, A_dims[0], B_dims[1], A_dims[1]);
	double stop_naive = mysecond();

#pragma omp parallel for
	for (int i = 0; i < A_dims[0] * B_dims[1]; i++) {
		double err = fabs(C_nav[i] - C[i]);
		if (err > 10e-6) {
			printf("error (%d): %f %f\n", i, C_nav[i], C[i]);
			exit(1);
		}
	}

	printf("MKL Max threads: %d\n", mkl_get_max_threads());
	printf("Generate A B: %f s\n", stop_random - start_random);
	printf("BLAS: %f s\n", stop_blas - start_blas);
	printf("Naive matmul: %f s\n", stop_naive - start_naive);

	free(A);
	free(B);
	free(C);
	free(C_nav);
}
