all:
	gcc   -o Kmeans_Serial.out Kmeans_Serial.c -lm
	gcc   -o Kmeans_OpenMP.out Kmeans_OpenMP.c -lm -fopenmp
	mpicc -o Kmeans_MPI.out    Kmeans_MPI.c    -lm
	mpicc -o Kmeans_Hybrid.out Kmeans_Hybrid.c -lm -fopenmp

	gcc   -o Cure_Serial.out   Cure_Serial.c -lm
	gcc   -o Cure_OpenMP.out   Cure_OpenMP.c   -lm -fopenmp
	mpicc -o Cure_MPI.out      Cure_MPI.c      -lm
	mpicc -o Cure_Hybrid.out   Cure_Hybrid.c   -lm -fopenmp

clean:
	rm Kmeans_Serial.out
	rm Kmeans_OpenMP.out
	rm Kmeans_MPI.out
	rm Kmeans_Hybrid.out
	rm Cure_Serial.out
	rm Cure_OpenMP.out
	rm Cure_MPI.out
	rm Cure_Hybrid.out