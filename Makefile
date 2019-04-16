default: all

all:
	mpicxx -o gauss_mpi main.cpp `pkg-config --libs opencv`

clean:
	rm gauss_mpi
