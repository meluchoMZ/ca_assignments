/*
   * Arquitectura de computadores
   * Scalable Uniform Matrix Multiplication Algorithm (SUMMA)
   * Miguel Blanco Godón
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define TRUE 1
#define FALSE 0

// Comunicadores:
// Se comentado, os comunicadores créanse con MPI_Comm_split
// Noutro caso, os comunicadores créanse mediante topoloxías virtuais
//#define CARTESIAN_TOPOLOGY

int debug = FALSE;
void print_usage(void);
void init_matrix(float *M, int rows, int cols, int zeros);
void print_matrix(float *M, int rows, int cols);
void sequential_matrix_product(float *A, float *B, float *C, float alpha, int m, int n, int k);
void compare_matrix(float *A, float *B, int A_rows, int A_cols, int B_rows, int B_cols);

int main(int argc, char **argv)
{
	float alpha, *A, *B, *C, *C_gathered, *recvbuffer, *local_A, *local_B, *local_C, *bcast_A, *bcast_B;
	int m, n, k, procs, rank, pack_position = 0, pack_buffer_size, pack_buffer_size_delta, recvcount, rows_rank, columns_rank, rows_procs, columns_procs;
	int *sendcounts, *displs;
	char *pack_buffer;
	MPI_Comm rows_communicator, columns_communicator;
	MPI_Datatype submatrix_A, submatrix_B, submatrix_C, submatrix_A_resized, submatrix_B_resized, submatrix_C_resized;

	// Inicio de MPI
	MPI_Init(&argc, &argv);
	// Obtención de número de procesos e o rango de cada un
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// O proceso 0 (root) encargase do parsing e inicializa as matrices
	if (rank == 0) {
		if (argc < 5 || argc > 6) {
			print_usage();
			return EXIT_FAILURE;
		}
		if (argc == 6) {
			if (strcmp(argv[argc-1], "-debug") == 0) {
				debug = TRUE;
				printf("Modo DEBUG\n");
			} else {
				print_usage();
				return EXIT_FAILURE;
			}
		}
		// parsing de parámetros
		m = atoi(argv[1]); n = atoi(argv[2]); 
		k = atoi(argv[3]); alpha = atof(argv[4]);
		if (m % (int)sqrt(procs) != 0 || n % (int)sqrt(procs) != 0 || k % (int)sqrt(procs) != 0 || m <= 0 || n <= 0 || k <= 0) {
			printf("\nERRO!\nOs tamaños escollidos non son válidos. Deben ser valores positivos mútiplos da raíz do número de procesos.\n");
			MPI_Abort(MPI_COMM_WORLD, 0);
	 		return EXIT_FAILURE;
		}		
		// reserva de memoria
		A = malloc(m*n*sizeof(float));
		B = malloc(n*k*sizeof(float));
		C = malloc(m*k*sizeof(float));
		// inicialización de matrices
		init_matrix(A, m, n, FALSE);
		init_matrix(B, n, k, FALSE);
		init_matrix(C, m, k, TRUE);
		if (debug == TRUE) {
			printf("Matriz A:\n");
			print_matrix(A, m, n);
			printf("Matriz B:\n");
			print_matrix(B, n, k);
			printf("Matriz C:\n");
			print_matrix(C, m, k);
		}
		// Calcula a multiplicación de xeito secuencial para comprobar se SUMMA está ben implementado
		sequential_matrix_product(A, B, C, alpha, m, n, k);
	}

	// Para enviar os parámetros de entrada do programa aos procesos non root empaquétanse

	// calculase o que ocupan m,n e k en MPI_PACKED
	MPI_Pack_size(3, MPI_INT, MPI_COMM_WORLD, &pack_buffer_size_delta);
	pack_buffer_size = pack_buffer_size_delta;
	// calculase canto ocupa alpha no empaquetamento
	MPI_Pack_size(1, MPI_FLOAT, MPI_COMM_WORLD, &pack_buffer_size_delta);
	// Engádese ao tamanho requirido anteior
	pack_buffer_size += pack_buffer_size_delta;
	// Reservase a memoria necesaria
	pack_buffer = malloc(pack_buffer_size);

	if (rank == 0) {
		// O proceso 0 empaqueta os datos
		MPI_Pack(&m, 1, MPI_INT, pack_buffer, pack_buffer_size, &pack_position, MPI_COMM_WORLD);
		MPI_Pack(&n, 1, MPI_INT, pack_buffer, pack_buffer_size, &pack_position, MPI_COMM_WORLD);
		MPI_Pack(&k, 1, MPI_INT, pack_buffer, pack_buffer_size, &pack_position, MPI_COMM_WORLD);
		MPI_Pack(&alpha, 1, MPI_FLOAT, pack_buffer, pack_buffer_size, &pack_position, MPI_COMM_WORLD);
		// Broadcast a todos os outros procesos
		MPI_Bcast(pack_buffer, pack_buffer_size, MPI_PACKED, 0, MPI_COMM_WORLD);
	} else {
		// Broadcast replicado ao ser unha operación colectiva
		MPI_Bcast(pack_buffer, pack_buffer_size, MPI_PACKED, 0, MPI_COMM_WORLD);
		// Desempaquetamento de datos
		MPI_Unpack(pack_buffer, pack_buffer_size, &pack_position, &m, 1, MPI_INT, MPI_COMM_WORLD);
		MPI_Unpack(pack_buffer, pack_buffer_size, &pack_position, &n, 1, MPI_INT, MPI_COMM_WORLD);
		MPI_Unpack(pack_buffer, pack_buffer_size, &pack_position, &k, 1, MPI_INT, MPI_COMM_WORLD);
		MPI_Unpack(pack_buffer, pack_buffer_size, &pack_position, &alpha, 1, MPI_FLOAT, MPI_COMM_WORLD);
	}

	// creación de tipos derivados
	MPI_Type_vector(m/(int)sqrt(procs), n/(int)sqrt(procs), n, MPI_FLOAT, &submatrix_A);
	MPI_Type_vector(n/(int)sqrt(procs), k/(int)sqrt(procs), k, MPI_FLOAT, &submatrix_B);
	MPI_Type_vector(m/(int)sqrt(procs), k/(int)sqrt(procs), k, MPI_FLOAT, &submatrix_C);
	MPI_Type_commit(&submatrix_A);
	MPI_Type_commit(&submatrix_B);
	MPI_Type_commit(&submatrix_C);
	// para usar operacións colectivas é necesario usar types resized
	MPI_Type_create_resized(submatrix_A, 0, n/(int)sqrt(procs)*sizeof(float), &submatrix_A_resized);
	MPI_Type_create_resized(submatrix_B, 0, k/(int)sqrt(procs)*sizeof(float), &submatrix_B_resized);
	MPI_Type_create_resized(submatrix_C, 0, k/(int)sqrt(procs)*sizeof(float), &submatrix_C_resized);
	MPI_Type_commit(&submatrix_A_resized);
	MPI_Type_commit(&submatrix_B_resized);
	MPI_Type_commit(&submatrix_C_resized);

	// calcúlanse os desprazamentos
	sendcounts = malloc(procs*sizeof(int));
	displs = malloc(procs*sizeof(int));
	
	local_A = malloc(m/sqrt(procs) * n/sqrt(procs) * sizeof(float));
	recvcount = m/(int)sqrt(procs) * n/(int)sqrt(procs);

	for (int i = 0; i < procs; i++) {
		sendcounts[i] = 1;
	}

	for (int i = 0; i < (int)sqrt(procs); i++) {
		for (int j = 0; j < (int)sqrt(procs); j++) {
			displs[i*(int)sqrt(procs)+j] = j + i*((n*m/(int)sqrt(procs)) / (n/(int)sqrt(procs)));
		}
	}

	if (rank == 0) {
	for (int i = 0; i < procs; i++) {
		printf("process %d: offset -> %d\n", i, displs[i]);
	}
	}

	if (debug == TRUE && rank == 0) {
		printf("M = %d; N = %d; K = %d; procs = %d\n", m, n, k, procs);
		printf("TypeVector: param1 = %d; param2 = %d; param3 = %d\n", m/(int)sqrt(procs), m/(int)sqrt(procs), m);
	}
	// compártense as submatrices entre os procesos
	MPI_Scatterv(A, sendcounts, displs, submatrix_A_resized, local_A, recvcount, MPI_FLOAT, 0, MPI_COMM_WORLD);


	for (int i = 0; i < procs; i++) {
		if (rank == i) {
			printf("\n RANK = %d\n", rank);
			print_matrix(local_A, m/(int)sqrt(procs), n/(int)sqrt(procs));
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	// compartese B
	for (int i = 0; i < (int)sqrt(procs); i++) {
		for (int j = 0; j < (int)sqrt(procs); j++) {
			displs[i*(int)sqrt(procs)+j] = j + i*((k*n/(int)sqrt(procs)) / (k/(int)sqrt(procs)));
		}
	}
	local_B = malloc(n/sqrt(procs) * k/sqrt(procs) * sizeof(float));
	recvcount = n/(int)sqrt(procs) * k/(int)sqrt(procs);
	MPI_Scatterv(B, sendcounts, displs, submatrix_B_resized, local_B, recvcount, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (rank == 0) printf("\nMatrix B\n");
	for (int i = 0; i < procs; i++) {
		if (rank == i) {
			printf("\n RANK = %d\n", rank);
			print_matrix(local_B, n/(int)sqrt(procs), k/(int)sqrt(procs));
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

#ifndef CARTESIAN_TOPOLOGY
	// creación de comunicadores con MPI_Comm_split
	MPI_Comm_split(MPI_COMM_WORLD, rank%(int)sqrt(procs), rank/(int)sqrt(procs), &columns_communicator);
	MPI_Comm_split(MPI_COMM_WORLD, rank/(int)sqrt(procs), rank%(int)sqrt(procs), &rows_communicator);

#else
	// creación de comunicadores con topoloxías virtuais

#endif

	// Algoritmo SUMMA

	// obtéñense os procesos e rangos por comunicador
	MPI_Comm_size(rows_communicator, &rows_procs);
	MPI_Comm_rank(rows_communicator, &rows_rank);
	MPI_Comm_size(columns_communicator, &columns_procs);
	MPI_Comm_rank(columns_communicator, &columns_rank);
	
	bcast_A = malloc(m/sqrt(procs) * n/sqrt(procs) * sizeof(float));
	bcast_B = malloc(n/sqrt(procs) * k/sqrt(procs) * sizeof(float));
	local_C = calloc(m/sqrt(procs) * k/sqrt(procs), sizeof(float));
	if (rank == 0) {
		printf("local_C before computation\n");
		print_matrix(local_C, m/sqrt(procs), k/sqrt(procs));
	}

	for (int p = 0; p < sqrt(procs); p++) {
		if (rows_rank == p) {
			memcpy(bcast_A, local_A, m/sqrt(procs)*n/sqrt(procs)*sizeof(float));
		}
		MPI_Bcast(bcast_A, m/sqrt(procs)*n/sqrt(procs), MPI_FLOAT, p, rows_communicator);
		if (columns_rank == p) {
			memcpy(bcast_B, local_B, n/sqrt(procs)*k/sqrt(procs)*sizeof(float));
		}
		MPI_Bcast(bcast_B, n/sqrt(procs)*k/sqrt(procs), MPI_FLOAT, p, columns_communicator);

		for (int i = 0; i < m/sqrt(procs); i++) {
			for (int j = 0; j < k/sqrt(procs); j++) {
				for (int l = 0; l < n/sqrt(procs); l++) {
					//C[i*k+j] += alpha * A[i*n+l] * B[l*k+j];
					local_C[i*(k/(int)sqrt(procs))+j] += alpha * bcast_A[i*(n/(int)sqrt(procs))+l] * bcast_B[l*(k/(int)sqrt(procs))+j];
					//printf("bcast_A[i] = %f; bcast_B[i] = %f\n", bcast_A[i*(n/(int)sqrt(procs))+l], bcast_B[l*(k/(int)sqrt(procs))+j]);
				}
			}
		}
	}
	// fin SUMMA
	if (rank == 0) {
		printf("local_C after computation\n");
		print_matrix(local_C, m/sqrt(procs), k/sqrt(procs));
	}

	for (int i = 0; i < (int)sqrt(procs); i++) {
		for (int j = 0; j < (int)sqrt(procs); j++) {
			displs[i*(int)sqrt(procs)+j] = j + i*((k*m/(int)sqrt(procs)) / (k/(int)sqrt(procs)));
		}
	}
	if (rank == 0) {
		C_gathered = malloc(m*k*sizeof(float));
	}
	recvcount = m/(int)sqrt(procs) * k/(int)sqrt(procs);
	MPI_Gatherv(local_C, recvcount, MPI_FLOAT, C_gathered, sendcounts, displs, submatrix_C_resized, 0, MPI_COMM_WORLD);


	if (debug == TRUE && rank == 0) {
		printf("Calculated matrix A*B = C:\n");
		print_matrix(C, m, k);
		printf("\nGathered matrix: \n");
		print_matrix(C_gathered, m, k);
		printf("\n");
		compare_matrix(C, C_gathered, m, k, m, k);
	}

	// desfeita de comunicadores
	MPI_Comm_free(&rows_communicator);
	MPI_Comm_free(&columns_communicator);

	// desfanse os tipos derivados
	MPI_Type_free(&submatrix_A); MPI_Type_free(&submatrix_A_resized);
	MPI_Type_free(&submatrix_B); MPI_Type_free(&submatrix_B_resized);
	MPI_Type_free(&submatrix_C); MPI_Type_free(&submatrix_C_resized);

	free(pack_buffer); pack_buffer = NULL;
	if (rank == 0) {
		free(A); free(B); free(C);
		A = NULL; B = NULL; C = NULL;
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}

void print_usage(void)
{
	printf("Erro sintactico. Uso:\n\t./p3 <n_filas_A> <num_columnas_A> <num_columnas_B> <factor_de_escalado> [-debug]\n");
	return;
}

void init_matrix(float *M, int rows, int cols, int zeros)
{
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (zeros == TRUE) {
				M[i*cols+j] = 0;
			} else {
				M[i*cols+j] = i*cols+j;
			}
		}
	}
	return;
}

void print_matrix(float *M, int rows, int cols)
{
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf(" %2f ", M[i*cols+j]);
		}
		printf("\n");
	}
	printf("\n");
	return;
}

void sequential_matrix_product(float *A, float *B, float *C, float alpha, int m, int n, int k)
{
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k; j++) {
			for (int l = 0; l < n; l++) {
				C[i*k+j] += alpha * A[i*n+l] * B[l*k+j];
			}
		}
	}
	return;
}

void compare_matrix(float *A, float *B, int A_rows, int A_cols, int B_rows, int B_cols)
{
	int diff = 0;
	if (A_cols != B_cols || A_rows != B_rows) {
		printf("Non se poden comparar. Non cadran os tamaños\n");
		printf("A(%dx%d) e b(%dx%d)\n", A_rows, A_cols, B_rows, B_cols);
		return;
	}
	for (int i = 0; i < A_rows; i++) {
		for (int j = 0; j < A_cols; j++) {
			if (A[i*A_cols+j] != B[i*A_cols+j]) {
				diff++;
			}
		}
	}
	if (diff == 0) {
		printf("Sen erros na execución paralela. Todos os calculos son correctos\n");
	} else {
		printf("ERROR! CALCULO PARALELO INCORRECTO! Os resultados difiren en %d posicións da matriz\n", diff);
	}
}
