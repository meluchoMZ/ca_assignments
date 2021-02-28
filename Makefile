all:
	mpicc -o p2 p2.c -g -Wall -Wextra -lm

clean:
	rm ./p2
