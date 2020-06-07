#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <mpi.h>

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


void fox_algorithm(double *C, double *A, double *B, const int dim, int nb_proc, const int STRIP)
{
	// -----------------------------------------
	// First, create the grid, communicators, and 
	// store informations for each process
	// -----------------------------------------

    // Create the grid
    MPI_Comm grid_comm;
    int dim_sizes[2];
    int wrap_around[2];
    int reorder = 1; 

    int sqrt_nb_proc = (int) sqrt(nb_proc);
    int block_dim = dim / sqrt_nb_proc;
    dim_sizes[0] = sqrt_nb_proc; 
    dim_sizes[1] = sqrt_nb_proc; 
    wrap_around[0] = 1; 
    wrap_around[1] = 0;  

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim_sizes, wrap_around, reorder, &grid_comm); 

    // Locate each process in the grid
    int coordinates[2];
    int grid_rank;
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_coords( grid_comm, grid_rank, 2, coordinates);  
    int row = coordinates[0];
    int column = coordinates[1];
    //printf("Grid_rank : %d, Coordinates : %d %d\n", grid_rank, row, column);

    // Create row and column communicators
    int free_coords[2];
    MPI_Comm row_comm;
    free_coords[0] = 0; 
    free_coords[1] = 1;
    MPI_Cart_sub(grid_comm, free_coords, &row_comm);
    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);

    MPI_Comm col_comm;
    free_coords[0] = 1; 
    free_coords[1] = 0;
    MPI_Cart_sub(grid_comm, free_coords, &col_comm);
    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);

    //printf("Grid_rank : %d, Row_rank=column : %d, Col_rank=row : %d, Coordinates : %d %d\n", grid_rank, row_rank, col_rank, row, column);

    
    //--------------------------------------------------
    // Then, decompose the global matrices A and B in tiles 
    // for each process (taken from https://stackoverflow.com/questions/9269399/
    // sending-blocks-of-2d-array-in-c-using-mpi/9271753#9271753)
    //--------------------------------------------------
    double *block_A = (double*)malloc(sizeof(double)*(block_dim*block_dim));
    double *block_B = (double*)malloc(sizeof(double)*(block_dim*block_dim));
    double *block_C = (double*)malloc(sizeof(double)*(block_dim*block_dim));
    double *temp_A = (double*)malloc(sizeof(double)*(block_dim*block_dim));
    double *temp_B = (double*)malloc(sizeof(double)*(block_dim*block_dim));

    for (int i = 0; i < block_dim*block_dim; i++) block_C[i] = 0.0;
    //memset(block_C, 0.0, (block_dim * block_dim) * sizeof(*block_C));

    int sizes[2] = {dim, dim};
    int subsizes[2] = {block_dim, block_dim};
    int starts[2] = {0,0}; 
    MPI_Datatype type, subarrtype;
    
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_create_resized(type, 0, block_dim*sizeof(double), &subarrtype);
    MPI_Type_commit(&subarrtype);

    //int type_size;
    //MPI_Type_size(subarrtype, &type_size);
    //printf("New type size : %d, end size : %d\n", type_size, block_dim*block_dim*sizeof(double));
    
    int sendcounts[nb_proc];
    int displs[nb_proc];

    if (grid_rank == 0) 
    {
    	for (int i = 0; i < nb_proc; i++) sendcounts[i] = 1;

        int disp = 0;
        for (int i = 0; i < sqrt_nb_proc; i++) 
        {
            for (int j = 0; j < sqrt_nb_proc; j++) 
            {
                displs[i*sqrt_nb_proc+j] = disp;
                disp += 1;
            }
            disp += (block_dim-1)*sqrt_nb_proc;
        }
    }


    MPI_Scatterv(A, sendcounts, displs, subarrtype, block_A, 
    	block_dim*block_dim, MPI_DOUBLE, 0, grid_comm);
    MPI_Scatterv(B, sendcounts, displs, subarrtype, block_B, 
    	block_dim*block_dim, MPI_DOUBLE, 0, grid_comm);

   	MPI_Barrier(grid_comm);

   	//----------------------------------------
   	// Now, the main loop of the fox algorithm
   	//----------------------------------------
	for (int i = 0; i < sqrt_nb_proc; i++)
	{
		/*if (grid_rank == 2)
		{
			printf("Grid_rank %d block_A before Bcast\n", grid_rank);
			show_matrice(block_A, block_dim);
			printf("Grid_rank %d temp_A before Bcast\n", grid_rank);
			show_matrice(temp_A, block_dim);
		}*/

		int diag_col = (row + i)%sqrt_nb_proc;
		if (diag_col == column)
		{
			MPI_Bcast(block_A, block_dim*block_dim, MPI_DOUBLE, diag_col, row_comm);
			multiply_matrices_block(block_C, block_A, block_B, block_dim, STRIP);
		} 
		else
		{
			MPI_Bcast(temp_A, block_dim*block_dim, MPI_DOUBLE, diag_col, row_comm);
			multiply_matrices_block(block_C, temp_A, block_B, block_dim, STRIP);
		}

		/*if (grid_rank == 2)
		{
			printf("Grid_rank %d block_A after Bcast\n", grid_rank);
			show_matrice(block_A, block_dim);
			printf("Grid_rank %d temp_A after Bcast\n", grid_rank);
			show_matrice(temp_A, block_dim);
			printf("Grid_rank %d block_B before shift\n", grid_rank);
			show_matrice(block_B, block_dim);
		}*/

		MPI_Status s;
		MPI_Request req_s, req_r;
		
		//printf("Grid_rank : %d, col_rank : %d, receiving from %d\n", grid_rank, column, (row + 1)%sqrt_nb_proc);
		MPI_Irecv(temp_B, block_dim*block_dim, MPI_DOUBLE, (row + 1)%sqrt_nb_proc, i, col_comm, &req_r);
		//printf("Grid_rank : %d, col_rank : %d, sending to %d\n", grid_rank, column, (row - 1 + sqrt_nb_proc)%sqrt_nb_proc);
		MPI_Isend(block_B, block_dim*block_dim, MPI_DOUBLE, (row - 1 + sqrt_nb_proc)%sqrt_nb_proc, i, col_comm, &req_s);
		
		MPI_Wait(&req_s, &s);
		MPI_Wait(&req_r, &s);

		memcpy(block_B, temp_B, block_dim*block_dim*sizeof(double));

		/*if (grid_rank == 2)
		{
			printf("Grid_rank %d block_B after shift\n", grid_rank);
			show_matrice(block_B, block_dim);
		}*/

		MPI_Barrier(grid_comm);
	}

	MPI_Gatherv(block_C, block_dim*block_dim, MPI_DOUBLE, C, sendcounts, displs, subarrtype, 0, grid_comm);

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

	// Check if the number of processes enables to equally 
	// divide the matrices in smaller square matrices
	if (dim % ((int)sqrt(nb_proc)) != 0)
	{
		printf("The square root of the number of processes must divide the dimension of the matrices\n");
		exit(1);
	}

	//printf("Dim : %d, block_dim : %d\n", dim, (int)(dim/sqrt(nb_proc)));
	int rank, size;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //printf("%d %d\n", rank, size);

	// Initialize matrices
	double *A = NULL, *B = NULL, *C = NULL, *C_nav = NULL;
	double start_naive, stop_naive, t1, t2;
	if (rank == 0)
	{
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

		
		/*// Show matrices if needed
		printf("Matrice A : \n");
		show_matrice(A, dim);
		printf("Matrice B : \n");
		show_matrice(B, dim);*/

		// Naive multiplication as a reference
		start_naive = mysecond();
		multiply_matrices(C_nav, A, B, dim);
		stop_naive = mysecond();
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	int STRIP = 32;
	// Fox algorithm
	t1 = MPI_Wtime();
	fox_algorithm(C, A, B, dim, nb_proc, STRIP);
	t2 = MPI_Wtime();

	if (rank == 0)
	{
		/*printf("Matrice C = AB : \n");
		show_matrice(C_nav, dim);
		printf("Matrice C = AB fox\n");
		show_matrice(C, dim);*/

		// Checking the correctness of our Fox algorithm implementation
		//#pragma omp parallel for
		for (int i = 0; i < dim * dim; i++) {
			double err = fabs(C_nav[i] - C[i]);
			if (err > 10e-6) {
				printf("error (%d): %f %f\n", i, C_nav[i], C[i]);
				exit(1);
			}
		}

		printf("No errors found\n");
		printf("Fox algorithm : %f s\n", t2-t1);
		printf("Naive matmul: %f s\n", stop_naive - start_naive);

		free(A);
		free(B);
		free(C);
		free(C_nav);
	}
	
	MPI_Finalize();
	
}