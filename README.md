# AlgParaleli
The program uses MPI and OpenCV to blur an image. It works as a serial algorithm( by providing 1 as the number of processes) or a parallel algorithm(nr of processes > 1).

compile: make

run example: mpirun -n 4 ./gauss_mpi input.jpg output.jpg
