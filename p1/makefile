default:
	@echo "Usage:"
	@echo "   p1c:     Practica 1 sin optimizaciones"
	@echo "   autovec    Practica 1 con vectorizacion automatica"
	@echo "   p1SSE:   Practica 1 con vectorizacion manual"

all: pSSEc autovec manvec1 manvec2 manvec3

p1c: matrizVectorP1.c
	gcc -O3 -o matrizVectorP1 matrizVectorP1.c -lm

autovec: matrizVectorP1.c
	gcc -O3 -march=nocona -msse3 -ftree-vectorize -ftree-vectorizer-verbose=2 -o matrizVectorP1Vec matrizVectorP1.c -lm

p1SSE: matrizVectorP1SSE.c
	gcc  -O3 -march=nocona -msse3 -o matrizVectorP1SSE matrizVectorP1SSE.c -lm

clean:
	-rm p3c matrizVectorP1 matrizVectorP1Vec matrizVectorP1SSE

