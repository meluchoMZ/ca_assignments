#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <pmmintrin.h>

//#define HADD

int main(int argc, char *argv[])
{
	int m, _m, n, _n, test, i, j, errors;
	float alpha, *x, *A, *y, *testy;
	struct timeval t0, t1, t;
	__m128 alpha_reg, x_reg, a_reg[4], aux_unpack[5];

	if (argc == 5) {
		m = atoi(argv[1]);
		n = atoi(argv[2]);
		alpha = atof(argv[3]);
		test = atoi(argv[4]);
	} else {
		printf("NUMERO DE PARAMETROS INCORRECTO\n");
		exit(EXIT_FAILURE);
	}

	// se os datos son múltiplos de 4, non se modifican na dimensión que sexan múltiplo
	// noutro caso, amplíase a matriz ao seguinte múltiplo maís próximo
	_m = ((float) m/4 > (int) m/4) ? ((int) (m/4+1)*4) : m;
	_n = ((float) n/4 > (int) n/4) ? ((int) (n/4+1)*4) : n;
	printf("_m = %d; _n = %d\n", _m, _n);

	// resérvase memoria cun aliñamento de 16 bytes
	x = (float *) _mm_malloc(_n*sizeof(float), 16);
	A = (float *) _mm_malloc(_m*_n*sizeof(float), 16);
	y = (float *) _mm_malloc(_m*sizeof(float), 16);

	// inicialízanse os datos
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
			// cárganse 16 bytes de x
			x_reg = _mm_load_ps(&x[j]);
			// multiplícanse polo factor de escalado
			x_reg = _mm_mul_ps(x_reg, alpha_reg);
			for (char c = 0x0; c < 0x4; c++) {
				// carganse 16 bytes da matriz
				a_reg[c] = _mm_load_ps(&A[(i+c)*_n+j]);
				// multiplicación elemento e elemento
				a_reg[c] = _mm_mul_ps(a_reg[c], x_reg);
			}
			// sumas horizontais dos vectores 2 a 2
			a_reg[0] = _mm_hadd_ps(a_reg[0], a_reg[1]);
			a_reg[2] = _mm_hadd_ps(a_reg[2], a_reg[3]);
			a_reg[0] = _mm_hadd_ps(a_reg[0], a_reg[2]);
			// cárgase o valor actual de y (neses 16 bytes)
			a_reg[1] = _mm_load_ps(&y[i]);
			// súmase o valor do produto matriz-vector a y
			a_reg[0] = _mm_add_ps(a_reg[0], a_reg[1]);
			// gárdase o valor no vector resultado
			_mm_store_ps(&y[i], a_reg[0]);
		}
	}
#else 
	for (i = 0; i < m; i+=4) {
		for (j = 0; j < n; j+=4) {
			// cárganse 16 bytes de x
			x_reg = _mm_load_ps(&x[j]);
			// multiplícanse polo factor de escalado
			x_reg = _mm_mul_ps(x_reg, alpha_reg);
			for (char c = 0x0; c < 0x4; c++) {
				// cárganse 16 bytes da matriz
				a_reg[c] = _mm_load_ps(&A[(i+c)*_n+j]);
				// multiplícase a matriz por x
				a_reg[c] = _mm_mul_ps(a_reg[c], x_reg);
			}
			// mestúranse os primeiros 8 bytes dos rexistros 
			aux_unpack[0] = _mm_unpacklo_ps(a_reg[0], a_reg[2]);
			aux_unpack[1] = _mm_unpacklo_ps(a_reg[1], a_reg[3]);
			// mestúranse os primeiros 8 bytes dos rexistros, de xeito que 
			// quedan os primeiros 4 bytes de cada a_reg[i] orixinal (a0,b0,c0,d0)
			aux_unpack[2] = _mm_unpacklo_ps(aux_unpack[0], aux_unpack[1]);
			// mestúranse so últimos 8 bytes dos rexistros, de xeito que quedan
			// os segundos 4 bytes de cada a_reg[i] orixinal (a1,b1,c1,d1)
			aux_unpack[3] = _mm_unpackhi_ps(aux_unpack[0], aux_unpack[1]);
			// mestúranse os 8 bytes máis significativos dos rexistros
			aux_unpack[0] = _mm_unpackhi_ps(a_reg[0], a_reg[2]);
			aux_unpack[1] = _mm_unpackhi_ps(a_reg[1], a_reg[3]);
			// mestúranse os primeiros 8 bytes dos rexistros, quedando os bytes
			// 7 a 11 de cada a_reg[i] orixinal (a2,b2,c2,d2)
			aux_unpack[4] = _mm_unpacklo_ps(aux_unpack[0], aux_unpack[1]);
			// mestúranse os últimos 4 bytes de cada a_reg[i] orixinal (a3,b3,c3,d3)
			aux_unpack[0] = _mm_unpackhi_ps(aux_unpack[0], aux_unpack[1]);
			// súmanse verticalmente os vectores
			aux_unpack[0] = _mm_add_ps(aux_unpack[4], aux_unpack[0]);
			aux_unpack[1] = _mm_add_ps(aux_unpack[2], aux_unpack[3]);
			aux_unpack[0] = _mm_add_ps(aux_unpack[0], aux_unpack[1]);
			// engádese o valor a y
			a_reg[0] = _mm_load_ps(&y[i]);
			a_reg[0] = _mm_add_ps(a_reg[0], aux_unpack[0]);
			// gárdase o valor no vector resultado
			_mm_store_ps(&y[i], a_reg[0]);
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
