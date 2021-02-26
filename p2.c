/*
   * Arquitectura de computadores
   * Miguel Blanco Godón
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#define TRUE 1
#define FALSE 0

int debug = FALSE;
void print_usage(void);
void init_matrix(float *M, int rows, int cols, int zeros);
void print_matrix(float *M, int rows, int cols);
void sequential_matrix_product(float *A, float *B, float *C, float alpha, int m, int n, int k);

int main(int argc, char **argv)
{
	float alpha, *A, *B, *C;
	int m, n, k;

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
	m = atoi(argv[1]); n = atoi(argv[2]); 
	k = atoi(argv[3]); alpha = atof(argv[4]);
	A = malloc(m*n*sizeof(float));
	B = malloc(n*k*sizeof(float));
	C = malloc(m*k*sizeof(float));
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
	sequential_matrix_product(A, B, C, alpha, m, n, k);
	if (debug == TRUE) {
		printf("Calculated matrix A*B = C:\n");
		print_matrix(C, m, k);
	}
	return EXIT_SUCCESS;
}

void print_usage(void)
{
	printf("Erro sintactico. Uso:\n\t./p2 <n_filas_A> <num_columnas_A> <num_columnas_B> [-debug]\n");
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
