all:
	mpicc -o p2 p2.c -g -Wall -Wextra

clean:
	rm ./p2
