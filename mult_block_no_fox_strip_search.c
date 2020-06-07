#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>

#define SEED     921


int min(int a, int b) {
    if (a > b)
        return b;
    return a;
}


double mysecond()
{
	struct timeval tp;
	struct timezone tzp;
	int i;
	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


void show_matrice(double *M, const int dim)
{

	printf("-----------------------------------------\n");
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			printf("%f ", M[i*dim + j]);
		}
		printf("\n");
	}
	printf("-----------------------------------------\n");
}


void initialize_matrice_rand(double *M, const int dim)
{
	for (int i = 0; i < dim * dim; i++) {
		M[i] = (double)rand() / (double)RAND_MAX;
	}
}


void initialize_matrice_identity(double *M, const int dim)
{
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++) 
		{
			if (i==j)
				M[i*dim + j] = 1;
			else
				M[i*dim + j] = 0;
		}
	}

}


void multiply_matrices(double *C, double *A, double *B, const int dim)
{
//#pragma omp parallel for
	for (int i = 0 ; i < dim ; i++) 
	{
		for (int k = 0 ; k < dim ; k++) 
		{
			for (int j = 0 ; j < dim ; j++) 
			{
				C[i*dim+j] += A[i*dim+k] * B[k*dim+j];
			}
		}
	}
}


void multiply_matrices_block(double *C, double *A, double *B, const int dim, const int STRIP)
{
	for (int ii = 0; ii < dim; ii+=STRIP)
	{
		for (int jj = 0; jj < dim; jj+=STRIP)
		{
			for (int kk = 0; kk < dim; kk+=STRIP)
			{
				int imax = min(ii+STRIP, dim);
				for (int i = ii; i < imax; i++)
				{
					int jmax = min(jj+STRIP, dim);
					for (int j = jj; j < jmax; j++)
					{
						double sum = 0;
						int kmax = min(kk+STRIP, dim);
						for (int k = kk; k < kmax; k++)
						{
							sum += A[i*dim+k] * B[k*dim+j];
						}
						C[i*dim+j] += sum;
					}
				}
			}
		}
	}
}


int main (int argc, char* argv[])
{

	srand(SEED);
	int provided;

	// Check if the program is correctly called
	if (argc != 2) 
	{
		printf("Usage: %s [M] \n", argv[0]);
		exit(1);
	}

	int dim = atoi(argv[1]);

    //printf("%d %d\n", rank, size);

	// Initialize matrices
	double *A = NULL, *B = NULL, *C = NULL, *C_nav = NULL;
	double start_naive, stop_naive, start_block, stop_block;
	A = (double*)malloc(sizeof(double) * (dim * dim));
	assert(A != NULL);
	B = (double*)malloc(sizeof(double) * (dim * dim));
	assert(B != NULL);
	C = (double*)malloc(sizeof(double) * (dim * dim));
	assert(C != NULL);
	C_nav = (double*)calloc(sizeof(double), (dim * dim));
	assert(C_nav != NULL);
	
	initialize_matrice_rand(A, dim);
	initialize_matrice_rand(B, dim);
	memset(C_nav, 0, (dim * dim) * sizeof(*C));
	memset(C, 0, (dim * dim) * sizeof(*C));

	int STRIP = 30;
	// Naive multiplication as a reference
	start_naive = mysecond();
	multiply_matrices(C_nav, A, B, dim);
	stop_naive = mysecond();
	
	printf("Naive matmul: %f s\n", stop_naive - start_naive);

	// Block matrix algorithm
	for (int STRIP = 10; STRIP < 50; STRIP+=2)
	{
		start_block = mysecond();
		multiply_matrices_block(C, A, B, dim, STRIP);
		stop_block = mysecond();

		for (int i = 0; i < dim * dim; i++) {
			double err = fabs(C_nav[i] - C[i]);
			if (err > 10e-6) {
				printf("error (%d): %f %f\n", i, C_nav[i], C[i]);
				exit(1);
			}
		}

		printf("No errors found\n");
		printf("STRIP : %d, Time : %f\n", STRIP, stop_block - start_block);
		memset(C, 0, (dim * dim) * sizeof(*C));
	}
	
	free(A);
	free(B);
	free(C);
	free(C_nav);
	
}