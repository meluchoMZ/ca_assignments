/*
   * Arquitectura de computadores
   * Miguel Blanco Godón
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

#define TRUE 1
#define FALSE 0

void print_usage(void);
void init_matrix(float *M, int rows, int cols, int zeros);
void print_matrix(float *M, int rows, int cols);
void sequential_matrix_product(float *A, float *B, float *C, float alpha, int m, int n, int k);
void compare_matrix(float *A, float *B, int A_rows, int A_cols, int B_rows, int B_cols);

int main(int argc, char **argv)
{
	float alpha, *A, *Ap, *B, *C, *Cp, *Cgathered;
	int debug = FALSE, m, n, k, procs, rank, rows_x_procs, *sendcnts, *displs, *kcnts, *kdispls;
	struct timeval start, finish;
	double mpi_time, mpi_endtime, *mpi_time_array;

	// Inicio da libraria.
	MPI_Init(&argc, &argv);
	// Obtenhense o numero de procesos invocados
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	// Obtense o rango de cada proceso
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// O proceso con rango 0 sera o que realice o parsing de datos do usuario e inicialice as matrices
	if (rank == 0) {
		if (argc < 5 || argc > 6) {
			if (rank == 0) {
				print_usage();
			}
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		if (argc == 6) {
			if (strcmp(argv[argc-1], "-debug") == 0) {
				debug = TRUE;
				if (rank == 0) {
					printf("Modo DEBUG\n");
				}
			} else {
				if (rank == 0) { 
					print_usage();
				}
				MPI_Finalize();
				return EXIT_FAILURE;
			}
		}
		m = atoi(argv[1]); n = atoi(argv[2]); 
		k = atoi(argv[3]); alpha = atof(argv[4]);
	}
	// O proceso 0 inicializa as matrices e realiza o calculo secuencial para comparacion e verificacion futura
	if (rank == 0) {
		A = malloc(m*n*sizeof(float));
		B = malloc(n*k*sizeof(float));
		C = malloc(m*k*sizeof(float));
		// Inicializanse as matrices (TRUE se se inicializan con ceros)
		init_matrix(A, m, n, FALSE);
		init_matrix(B, n, k, FALSE);
		init_matrix(C, m, k, TRUE);
		if (debug == TRUE && m<=10 && n<=10 && k<=10) {
			printf("Matriz A:\n");
			print_matrix(A, m, n);
			printf("Matriz B:\n");
			print_matrix(B, n, k);
			printf("Matriz C:\n");
			print_matrix(C, m, k);
		}
		// O proceso 0 tamen resolve o problema secuencialmente para posiblitar a posterior comparacion de resultados
		gettimeofday(&start, NULL);
		sequential_matrix_product(A, B, C, alpha, m, n, k);
		gettimeofday(&finish, NULL);
		if (debug == TRUE && m<=10 && m<=10 && k<=10) {
			printf("Matriz calculada A*B = C:\n");
			print_matrix(C, m, k);
		}
		printf("Problema resolto secuencialmente. t = %f segundos\n",(double) ((finish.tv_sec-start.tv_sec)*1000+(finish.tv_usec-start.tv_usec)/1000)/1000);
		printf("A continuacion resolverase o problema aplicando paralelismo.\n");
	}

	// A partires daqui comeza o codigo paralelo
	// Cada nucleo comeza a medir o seu tempo
	// Ponse unha barreira para que todos comecen ao mesmo tempo
	MPI_Barrier(MPI_COMM_WORLD);
	mpi_time = MPI_Wtime();
	if (debug == TRUE && rank == 0) {
			printf("Creados %d elementos de proceso\n", procs);
	}
	// Envianselle aos outros procesos os datos de entrada do problema
	MPI_Bcast((void *) &m, 1, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Bcast((void *) &n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void *) &k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void *) &alpha, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void *) &debug, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (debug == TRUE && rank == 0) {
		printf("Enviados parametros de entrada\n");
	}
	// Faise unha distribución por bloques da matriz A para cada elemento de proceso
	// Pra iso o primeiro que hai que facer é calcular o número de filas das que cada elemento e responsabel.
	rows_x_procs = (int) round(m/procs);
	if (debug == TRUE && rank == 0) {
		printf("Cada elemento de proceso procesa %d filas de %d, excepto o ultimo que procesa %d filas\n", rows_x_procs, m, m-(procs-1)*rows_x_procs);
	}
	// Todos os procesos levan m/procs elementos menos o ultimo que leva o que queda (m - (procs-1)*rows_x_procs)	
	// Reserva memoria en cada elemento de proceso menos o root
	if (rank != procs - 1) {
		Ap = malloc(rows_x_procs*n*sizeof(float));
		Cp = malloc(rows_x_procs*k*sizeof(float));
		init_matrix(Cp, rows_x_procs, k, TRUE);
	} else {
		Ap = malloc((m-(procs-1)*rows_x_procs)*n*sizeof(float));
		Cp = malloc((m-(procs-1)*rows_x_procs)*k*sizeof(float));
		init_matrix(Cp, (m-(procs-1)*rows_x_procs), k, TRUE);
	}
	if (rank != 0) {
		B = malloc(n*k*sizeof(float));
	}
	if (debug == TRUE && rank == 0) {
		printf("%d: Reservada memoria para A, B e C\n", rank);
	}
	// Envianse os anacos de A e C correspondentes para cada elemento de proceso	
	// Para iso hai que crear un vector co numero de filas por elemento de proceso, posto que non se garante que os tamanhos
	// das matrices sexan multiplo do numero de procesos. Tamen hai que facer o vector desprazamento.
	sendcnts = malloc(procs*sizeof(int));
	displs = malloc(procs*sizeof(int));
	kcnts = malloc(procs*sizeof(int));
	kdispls = malloc(procs*sizeof(int));
	if (rank == 0) {
		for (int i = 0; i < procs-1; i++) {
			sendcnts[i] = rows_x_procs*n;
		}
		sendcnts[procs-1] = (m-(procs-1)*rows_x_procs)*n;
		displs[0] = 0;
		for (int i = 1; i < procs; i++) {
			displs[i] = displs[i-1]+sendcnts[i-1];
		}	
		for (int i = 0; i < procs-1; i++) {
			kcnts[i] = rows_x_procs*k;
		}
		kcnts[procs-1] = (m-(procs-1)*rows_x_procs)*k;
		kdispls[0] = 0;
		for (int i = 1; i < procs; i++) {
			kdispls[i] = kdispls[i-1]+kcnts[i-1];
		}
	}
	// Compartense os arrays
	MPI_Bcast((void *) sendcnts, procs, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void *) displs, procs, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void *) kcnts, procs, MPI_INT, 0, MPI_COMM_WORLD);
	if (debug == TRUE) {
		if (rank == 0) {
			printf("%d: Creados arrays de contadores e desprazamentos\n", rank);
		}
		if (rank == 0 && procs <= 40) {
			printf("%d: Contadores scatter:\n", rank);
			for (int i = 0; i < procs; i++) {
				printf(" %d ", sendcnts[i]);
			}
			printf("\n%d: Offsets scatter:\n", rank);
			for (int i = 0; i < procs; i++) {
				printf(" %d ", displs[i]);
			}
			printf("\n");
			printf("%d: Contadores gather:\n", rank);
			for (int i = 0; i < procs; i++) {
				printf(" %d ", kcnts[i]);
			}
			printf("\n%d: Offsets gather:\n", rank);
			for (int i = 0; i < procs; i++) {
				printf(" %d ", kdispls[i]);
			}
			printf("\n");
		}
	}
	// Envianse as partes correspondentes de A a cada elemento de proceso
	MPI_Scatterv((void *) A, sendcnts, displs, MPI_FLOAT, (void *) Ap, sendcnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (debug == TRUE && rank == 0 && m <= 10 && n <= 10 && k <= 10) {
		printf("Compartidas matrices A e C\n");
		printf("Ap\n");
		print_matrix(Ap, sendcnts[rank]/n, n);
		printf("Cp\n");
		print_matrix(Cp, kcnts[rank]/k, k);
	}
	// Enviase a matriz B completa a cada elemento de proceso
	MPI_Bcast((void *) B, n*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (debug == TRUE && rank == 0) {
		printf("Compartida matriz B\n");
	}
	// Calculase o produto en cada elemento de proceso
	if (debug == TRUE) {
		printf("%d: Recibidas as matrices A e C\n", rank);
		printf("%d: Recibida matriz B\n", rank);
		for (int i = 0; i < procs; i++) {
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == i) {
				printf("Elemento de proceso %d de %d: multiplicacion parcial realizada\n", rank, procs);
			}
		}
	}
	// Recollense os computos individuais e metese nunha unica matriz
	if (rank == 0) {
		Cgathered = malloc(m*k*sizeof(float));
	}
	// calculase o produto. Vale o mesmo algoritmo porque a distribucion e por bloques de filas
	sequential_matrix_product(Ap, B, Cp, alpha, sendcnts[rank]/n, n, k);
	if (debug == TRUE) {
		for (int i = 0; i < procs; i++) {
			MPI_Barrier(MPI_COMM_WORLD);
			if (rank == i) {
				printf("Proceso %d: valor de sendcnts[%d] = %d\n", rank, rank, sendcnts[rank]);
			}
		}
	}
	MPI_Gatherv((void *) Cp, kcnts[rank], MPI_FLOAT, (void *) Cgathered, kcnts, kdispls, MPI_FLOAT, 0, MPI_COMM_WORLD);
	// Neste punto cada elemento de proceso remata a sua parte
	mpi_time = MPI_Wtime() - mpi_time;
	if (rank == 0) {
		printf("Compartindo datos de tempos...\n");
		mpi_time_array = malloc(procs*sizeof(double));
	}
	MPI_Gather(&mpi_time, 1, MPI_DOUBLE, mpi_time_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// Sincroniza para que  o if seguinte tenha sentido
	MPI_Barrier(MPI_COMM_WORLD);
	if (debug == TRUE && rank == 0) {
		printf("Computacion paralela finalizada\n");
	}
	if (debug == TRUE && rank == 0 && m <= 10 && k <= 10) {
		printf("Cgathered:\n");
		print_matrix(Cgathered, m, k);
	}
	if (rank == 0) {
		printf("Problema resolto aplicando paralelismo con %d elementos de proceso. Tempo utilizado:\n", procs);
		mpi_endtime = 0;
		for (int i = 0; i < procs; i++) {
			if (mpi_endtime < mpi_time_array[i]) {
				mpi_endtime = mpi_time_array[i];
			}
			printf("EP %d (de %d): %f segundos\n", i, procs, mpi_time_array[i]);
		}
		printf("Tempo de resolucion: %f segundos\n", mpi_endtime);
		if (mpi_endtime  == 0) {
			printf("Factor de aceleracion incalculable debido a resolucion demasiado rapida. Probe cun problema de maior dimension\n");
		} else {
			printf("Factor de aceleracion (t_secuencial / t_paralelo): %f\n", ((finish.tv_sec-start.tv_sec)*1000+(finish.tv_usec-start.tv_usec)/1000)/(1000*mpi_endtime));
		}
		compare_matrix(C, Cgathered, m, k, m, k);
	}	

	// Liberase a memoria 
	free(Ap); Ap = NULL;
	free(Cp); Cp = NULL;
	free(B); B = NULL;
	free(sendcnts); sendcnts = NULL;
	free(displs); displs = NULL;
	free(kcnts); kcnts = NULL;
	free(kdispls); kdispls = NULL;
	if (rank == 0) {
		free(A); A = NULL;
		free(C); C = NULL;
		free(Cgathered); Cgathered = NULL;
		free(mpi_time_array); mpi_time_array = NULL;
	}
	if (debug == TRUE && rank == 0) {
		printf("Memoria liberada\n");
	}
	MPI_Finalize();
	return EXIT_SUCCESS;
}

void print_usage(void)
{
	printf("Uso:\n\t./p2 <n_filas_A> <num_columnas_A> <num_columnas_B> <alfa> [-debug]\n");
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
		printf("A(%dx%d) e B(%dx%d)\n", A_rows, A_cols, B_rows, B_cols);
		return;
	}
	for (int i = 0; i < A_rows; i++) {
		for (int j = 0; j < A_cols; j++) {
			 if(A[i*A_cols+j] != B[i*A_cols+j]) {
				 diff++;
			 }
		}
	}
	if (diff == 0) {
		printf("Sen erros na execucion paralela. Todos os calculos son correctos\n");
	} else {
		printf("ERROR! CALCULO PARALELO INCORRECTO! Os resultados difiren en %d posicions da matriz\n", diff);
	}
}
