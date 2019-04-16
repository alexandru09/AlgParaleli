//standard includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
//openCV include
//#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
//MPI includes
#include "mpi.h"

using namespace std;
using namespace cv;

const int MAXBYTES = 8 * 1024 * 1024*100;
uchar buffer[MAXBYTES];

bool validateArgs(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <input_file> <output_file> \n", argv[0]);
        return false;
    }
    //verificam daca putem citi imaginea de input
    struct stat st;
    int result = stat(argv[1], &st);
    if (result != 0)
    {
        fprintf(stderr, "Program can not open input file \n");
        return false;
    }
    return true;
}

int main(int argc, char *argv[])
{
    int nRank, nWorldSize, nRoot = 0;
    int nSource, dest;
    int slice_size = 0;
    Mat src;
    MPI_Status status;
    double startTime, totalTime;
    int ksize = 15;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nWorldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &nRank);

    //citim imaginea
    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    div_t colsPerProcess = div(src.cols, nWorldSize); //cate coloane prelucram la fiecare proces
    slice_size = colsPerProcess.quot;                 //valoarea pt toate procesele
    
    startTime = MPI_Wtime();
    //verificam daca suntem in primul proces
    if (nRank == nRoot)
    {
        //verificam input-ul
        if(!validateArgs(argc, argv)) {
            exit(EXIT_FAILURE);
        }

        int sliceCols = colsPerProcess.quot + colsPerProcess.rem; //ajustam in caz ca nr de coloane nu e div cu nr de procese
        int startX = 0, startY = 0, width = sliceCols, height = src.rows;
        //selectam regiunea care va fi prelucrata de acest proces
        Rect region = Rect(startX, startY, width, height);
        Mat slice = src(region);
        //aplicam filtrul de blur
        Mat p_slice;
        GaussianBlur(slice, p_slice, Size(ksize, ksize), 10);
        slice = p_slice;

        dest = nRank + 1; //procesul la care vom trimite
        //cream un buffer care sa contina informatiile matricei prelucrate
        int rows = slice.rows;
        int cols = slice.cols;
        int type = slice.type();
        int channels = slice.channels();

        memcpy(&buffer[0 * sizeof(int)], (uchar *)&rows, sizeof(int));
        memcpy(&buffer[1 * sizeof(int)], (uchar *)&cols, sizeof(int));
        memcpy(&buffer[2 * sizeof(int)], (uchar *)&type, sizeof(int));

        int bytespersample = 1; // change if using shorts or floats
        int bytes = slice.rows * slice.cols * channels * bytespersample;

        if (!slice.isContinuous())
        {
            slice = slice.clone();
        }
        memcpy(&buffer[3 * sizeof(int)], slice.data, bytes);
        //verificam daca suntem in modul serial
        if (nWorldSize > 1)
            MPI_Send(&buffer, bytes + 3 * sizeof(int), MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
    }
    else
    {
        int startX = (nRank * slice_size) - 1, startY = 0, width = slice_size, height = src.rows;
        Rect region = Rect(startX, startY, width, height);
        Mat slice = src(region);
        Mat p_slice;
        GaussianBlur(slice, p_slice, Size(ksize, ksize), 10);
        slice = p_slice;

        int rows, cols, type, channels;
        nSource = nRank - 1;
        //primim matricea prelucrata de la procesul anterior
        MPI_Recv(buffer, sizeof(buffer), MPI_UNSIGNED_CHAR, nSource, 0, MPI_COMM_WORLD, &status);
        memcpy((uchar *)&rows, &buffer[0 * sizeof(int)], sizeof(int));
        memcpy((uchar *)&cols, &buffer[1 * sizeof(int)], sizeof(int));
        memcpy((uchar *)&type, &buffer[2 * sizeof(int)], sizeof(int));

        // Make the mat
        Mat received = Mat(rows, cols, type, (uchar *)&buffer[3 * sizeof(int)]);
        Mat processed;
        hconcat(received, slice, processed);
        dest = nRank + 1; //procesul la care vom trimite
        rows = processed.rows;
        cols = processed.cols;
        type = processed.type();
        channels = processed.channels();

        memcpy(&buffer[0 * sizeof(int)], (uchar *)&rows, sizeof(int));
        memcpy(&buffer[1 * sizeof(int)], (uchar *)&cols, sizeof(int));
        memcpy(&buffer[2 * sizeof(int)], (uchar *)&type, sizeof(int));

        int bytespersample = 1; // change if using shorts or floats
        int bytes = processed.rows * processed.cols * channels * bytespersample;

        if (!processed.isContinuous())
        {
            processed = processed.clone();
        }
        memcpy(&buffer[3 * sizeof(int)], processed.data, bytes);
        if (nRank < nWorldSize - 1 )
        {
            MPI_Send(&buffer, bytes + 3 * sizeof(int), MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
        }
        else if (nWorldSize > 1)
        {
            MPI_Send(&buffer, bytes + 3 * sizeof(int), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (nRank == nRoot)
    {
        int rows, cols, type, channels;
        totalTime = MPI_Wtime() - startTime;
    	nSource = nWorldSize - 1;
        if (nWorldSize > 1)
        {
    	    MPI_Recv(buffer, sizeof(buffer), MPI_UNSIGNED_CHAR, nSource, 0, MPI_COMM_WORLD, &status);
        }
        memcpy((uchar *)&rows, &buffer[0 * sizeof(int)], sizeof(int));
        memcpy((uchar *)&cols, &buffer[1 * sizeof(int)], sizeof(int));
        memcpy((uchar *)&type, &buffer[2 * sizeof(int)], sizeof(int));

        // Make the mat
        Mat received = Mat(rows, cols, type, (uchar *)&buffer[3 * sizeof(int)]);
    	imwrite(argv[2], received);
        printf("%f\n", totalTime);
    }

    MPI_Finalize();
}