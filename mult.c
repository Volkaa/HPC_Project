#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <mpi.h>

#define SEED     921

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


void fox_algorithm(double *C, double *A, double *B, const int dim, int nb_proc)
{

	// First, create the grid and communicators
	int rank, size;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("%d %d\n", rank, size);

    MPI_Comm grid_comm;
    int dim_sizes[2];
    int wrap_around[2];
    int reorder = 1; 

    dim_sizes[0] = (int) sqrt(nb_proc); 
    dim_sizes[1] = (int) sqrt(nb_proc); 
    wrap_around[0] = 1; 
    wrap_around[1] = 0;  

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim_sizes, wrap_around, reorder, &grid_comm); 
}


int main (int argc, char* argv[])
{

	srand(SEED);
	int provided;

	// Check if the program is correctly called
	if (argc != 3) 
	{
		printf("Usage: %s [M] [nb_proc] \n", argv[0]);
		exit(1);
	}

	int dim = atoi(argv[1]);
	int nb_proc = atoi(argv[2]);

	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

	// Check if the given number of processes is a square
	if (((int)sqrt(nb_proc))*((int)sqrt(nb_proc)) != nb_proc)
	{
		printf("The number of processes must be a square\n");
		exit(1);
	}

	// Initialize matrices
	double *A = (double*)malloc(sizeof(double) * (dim * dim));
	assert(A != NULL);
	double *B = (double*)malloc(sizeof(double) * (dim * dim));
	assert(B != NULL);
	double *C = (double*)malloc(sizeof(double) * (dim * dim));
	assert(C != NULL);
	double *C_nav = (double*)calloc(sizeof(double), (dim * dim));
	assert(C_nav != NULL);
	
	initialize_matrice_rand(A, dim);
	initialize_matrice_identity(B, dim);
	memset(C_nav, 0, (dim * dim) * sizeof(*C));
	memset(C, 0, (dim * dim) * sizeof(*C));

	/*
	// Show matrices if needed
	printf("Matrice A : \n");
	show_matrice(A, dim);
	printf("Matrice B : \n");
	show_matrice(B, dim);*/

	// Naive multiplication as a reference
	double start_naive = mysecond();
	multiply_matrices(C_nav, A, B, dim);
	double stop_naive = mysecond();

	// Fox algorithm
	double t1 = MPI_Wtime();
	fox_algorithm(C, A, B, dim, nb_proc);
	double t2 = MPI_Wtime();

	/*printf("Matrice C = AB : \n");
	show_matrice(C_nav, dim);

	// Checking the correctness of our Fox algorithm implementation
//#pragma omp parallel for
	for (int i = 0; i < dim * dim; i++) {
		double err = fabs(C_nav[i] - C[i]);
		if (err > 10e-6) {
			printf("error (%d): %f %f\n", i, C_nav[i], C[i]);
			exit(1);
		}
	}*/

	printf("Fox algorithm : %f s\n", t2-t1);
	printf("Naive matmul: %f s\n", stop_naive - start_naive);
	
	free(A);
	free(B);
	free(C);
	free(C_nav);

	MPI_Finalize();
	
}