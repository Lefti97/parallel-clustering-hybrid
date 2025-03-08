# Hybrid Parallel Clustering Algorithms

Title: 
Development and Evaluation of Parallel Clustering Algorithms in Hybrid Enviroment using OpenMP and MPI

Abstract:
The object of this thesis will be the design, development and evaluation, in a parallel environment of
shared memory, distributed memory, and hybrid form (massive parallel programming in a combined
environment of distributed-shared memory), of efficient algorithms for the problem of data clustering.
The development of the algorithms that will be selected will be done in C/C++ language and their
evaluation will be done in a suitable real environment. Individual implementations in OpenMP and/or
MPI as well as combined implementations will be developed indicatively, such as e.g. using
MPI+OpenMP and/or using MPI+MPI Shared Memory, and corresponding comparative
measurements and conclusions will be drawn.

- Link to final document of the thesis in greek will be added.

For this thesis parallel implementations were made for the clustering algorithms Kmeans and CURE.
Further implementations of parallel clustering algorithms may be added in this repo.

# Instructions

- Compile the code in src folder using Makefile
```
make
```

- How to execute:
```
                          ./Kmeans_Serial <filename> <clusters> <distance_threshold>
                          ./Kmeans_OpenMP <filename> <clusters> <distance_threshold> <OpenMP threads>
mpirun -n <mpi processes> ./Kmeans_MPI    <filename> <clusters> <distance_threshold>
mpirun -n <mpi processes> ./Kmeans_Hybrid <filename> <clusters> <distance_threshold> <OpenMP threads>

                          ./Cure_Serial <filename> <clusters> <representatives> <shrink fraction>
                          ./Cure_OpenMP <filename> <clusters> <representatives> <shrink fraction> <OpenMP threads>
mpirun -n <mpi processes> ./Cure_MPI    <filename> <clusters> <representatives> <shrink fraction>
mpirun -n <mpi processes> ./Cure_Hybrid <filename> <clusters> <representatives> <shrink fraction> <OpenMP threads>
```

- Example Runs:
```
            ./Kmeans_Serial.out ../inputs/test1K.txt 3 1
            ./Kmeans_OpenMP.out ../inputs/test1K.txt 3 1 4
mpirun -n 4 ./Kmeans_MPI.out    ../inputs/test1K.txt 3 1
mpirun -n 4 ./Kmeans_Hybrid.out ../inputs/test1K.txt 3 1 4

            ./Cure_Serial.out ../inputs/test1K.txt 3 5 0.4
            ./Cure_OpenMP.out ../inputs/test1K.txt 3 5 0.4 4
mpirun -n 4 ./Cure_MPI.out    ../inputs/test1K.txt 3 5 0.4
mpirun -n 4 ./Cure_Hybrid.out ../inputs/test1K.txt 3 5 0.4 4
```

- If gnuplot is available, a scatter plot will be saved in src/output folder.
- A txt file with the terminal output will be saved in src/output folder.

# Resources
- Hadjidoukas, Panagiotis & Amsaleg, Laurent. (2006). Parallelization of a Hierarchical Data Clustering Algorithm Using OpenMP. 4315. 289-299. 10.1007/978-3-540-68555-5_24. [Link](https://www.researchgate.net/publication/220875737_Parallelization_of_a_Hierarchical_Data_Clustering_Algorithm_Using_OpenMP)
- Zhang, Jing & Wu, Gongqing & Xuegang, Hu & Li, Shiying & Hao, Shuilong. (2011). A Parallel K-Means Clustering Algorithm with MPI. 10.1109/PAAP.2011.17. [Link](https://www.researchgate.net/publication/239764038_A_Parallel_K-Means_Clustering_Algorithm_with_MPI)
