#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <pmmintrin.h>

#define HADD
#define NOHADD

int main(int argc, char *argv[])
{
	int m, _m, n, _n, test, i, j, errors;
	float alpha, *x, *A, *y, *testy;
	struct timeval t0, t1, t;
	__m128 alpha_reg, x_reg, a_reg[4];

	if (argc == 5) {
		m = atoi(argv[1]);
		n = atoi(argv[2]);
		alpha = atof(argv[3]);
		test = atoi(argv[4]);
	} else {
		printf("NUMERO DE PARAMETROS INCORRECTO\n");
		exit(EXIT_FAILURE);
	}

	_m = m; _n = n;

	if ((float) m/4 > (int) m/4) {
		_m = (int) (m/4 + 1) * 4;
	}
	if ((float) n/4 > (int) n/4) {
		_n = (int) (n/4 + 1) * 4;
	}
	printf("_m = %d; _n = %d\n", _m, _n);

	x = (float *) _mm_malloc(_n*sizeof(float), 16);
	A = (float *) _mm_malloc(_m*_n*sizeof(float), 16);
	y = (float *) _mm_malloc(_m*sizeof(float), 16);

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			A[i*_n+j] = 1+i+j;
		}
	}


	for (i = 0; i < n; i++) {
		x[i] = (1+i);
	}
	

	for (i = 0; i < m; i++) {
		y[i] = (1-i);
	}

	if (test) {
		printf("\nMatriz A é ...\n");
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				printf("%f ", A[i*_n+j]);
			}
			printf("\n");
		}

		printf("\nVector X é ...\n");
		for (i = 0; i < n; i++) {
			printf("%f ", x[i]);
		}
		printf("%f ", x[i]);

		printf("\nVector Y ao comezo é ...\n");
		for (i = 0; i < m; i++) {
			printf("%f ", y[i]);
		}
		printf("%f ", y[i]);
	}

	assert(gettimeofday(&t0, NULL) == 0);
	// replícase o factor de escalado en todo o rexistro pra paralelizar a multiplicación
	alpha_reg = _mm_set_ps(alpha, alpha, alpha, alpha);
#ifdef HADD
	for (i = 0; i < m; i+=4) {
		for (j = 0; j < n; j+=4) {
			x_reg = _mm_load_ps(&x[j]);
			for (char c = 0x0; c < 0x4; c++) {
				a_reg[c] = _mm_load_ps(&A[(i+c)*_n+j]);
				a_reg[c] = _mm_mul_ps(a_reg[c], x_reg);
			}
			a_reg[0] = _mm_hadd_ps(a_reg[0], a_reg[1]);
			a_reg[2] = _mm_hadd_ps(a_reg[2], a_reg[3]);
			a_reg[0] = _mm_hadd_ps(a_reg[0], a_reg[2]);
			a_reg[1] = _mm_load_ps(&y[i]);
			a_reg[0] = _mm_add_ps(a_reg[0], a_reg[1]);
			_mm_store_ps(&y[i], a_reg[0]);
		}
	}
#else 
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			y[i] += alpha*A[i*n+j]*x[j];
		}
	}
#endif
	assert(gettimeofday(&t1, NULL) == 0);
	timersub(&t1, &t0, &t);

	if (test) {
		printf("\nAo final o vector Y é ...\n");
		for (i = 0; i < m; i++) {
			printf("%f ", y[i]);
		}
		printf("\n");

		testy = (float *) malloc(m*sizeof(float));
		for (i = 0; i < m; i++) {
			testy[i] = 1-i;
		}

		// calculo sen vectorización
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				testy[i] += alpha*A[i*_n+j]*x[j];
			}
		}

		errors = 0;
		for (i = 0; i < m; i++) {
			if (testy[i] != y[i]) {
				errors++;
				printf("\n Error na posición %d xa que %f != %f", i, y[i], testy[i]);
			}
		}
		printf("\n%d erros no produto matriz vector con dimensións %dx%d\n", errors, m, n);
		free(testy);
	}
	printf("Tempo = %ld:%ld(seg:mseg)\n", t.tv_sec, t.tv_usec/1000);
	_mm_free(x); _mm_free(A); _mm_free(y);

	return(EXIT_SUCCESS);
}
