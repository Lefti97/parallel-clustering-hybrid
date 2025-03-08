#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "extras.h"

void   mem_alloc_local_arrays();
void   mem_alloc_master_arrays();
void   mpi_before_calc();
void   mpi_after_calc();
void   get_random_initial_clusters(double *clustersX, double *clustersY, double *pointsX, double *pointsY, int elements);
long double get_distance_from_clusters(double *p_clustersX, double *p_clustersY, double *p_recv_pointsX, double *p_recv_pointsY, int p_recv_size);
void   assign_to_cluster(double *p_clustersX, double *p_clustersY, double *p_recv_pointsX, double *p_recv_pointsY, int *p_recv_pointsClusterAssign, int p_recv_size);
void   calc_cluster_centroid(double *p_clustersX, double *p_clustersY, double *p_recv_pointsX, double *p_recv_pointsY, int *p_recv_pointsClusterAssign, int p_recv_size);
void   merge_clusters(double *p_clustersX, double *p_clustersY, double *p_finalClustersX, double *p_finalClustersY);
void   print_progress();
void   plot_kmeans(double *pX, double *pY, int *pClAssign, double *clX, double *clY, int numClusters, int numPoints);


// Allocate memory for all processors
void mem_alloc_local_arrays(){
    clustersX = (double *)malloc(sizeof(double) * num_of_clusters);
    clustersY = (double *)malloc(sizeof(double) * num_of_clusters);
    if(clustersX == NULL || clustersY == NULL)
    { perror("clusters malloc"); exit(-1); }

    if (run_enviroment == ENV_HYBRID || run_enviroment == ENV_MPI)
    {
        //GET RECEIVING SIZE
        sendcounts = (int *)malloc(sizeof(int) * num_of_processes);
        displs     = (int *)malloc(sizeof(int) * num_of_processes);
        get_scatter_variables(sendcounts, displs, num_of_elements, 0);
        recv_size = sendcounts[process_rank];

        //ALLOCATE MEMORY FOR RECEIVING ELEMENTS
        recv_pointsX             = (double *)malloc(sizeof(double) * recv_size);
        recv_pointsY             = (double *)malloc(sizeof(double) * recv_size);
        recv_pointsClusterAssign = (int *)   malloc(sizeof(int)    * recv_size);
        if(recv_pointsX == NULL || recv_pointsY == NULL || recv_pointsClusterAssign == NULL)
        { perror("recv malloc"); exit(-1); }
    
        globalClustersX        = (double *)malloc(sizeof(double) * total_proc_clusters);
        globalClustersY        = (double *)malloc(sizeof(double) * total_proc_clusters);
        procClusterFinalAssign = (int *)malloc(sizeof(int) * total_proc_clusters);
        if(globalClustersX == NULL || globalClustersY == NULL || procClusterFinalAssign == NULL)
        { perror("globalClusters malloc"); exit(-1); }

        //ALLOCATE MEMORY FOR FINAL CLUSTERS
        finalClustersX = (double *)malloc(sizeof(double) * num_of_clusters);
        finalClustersY = (double *)malloc(sizeof(double) * num_of_clusters);
        if(finalClustersX == NULL || finalClustersY == NULL)
        { perror("clusters malloc"); exit(-1); }
    }
}

// Allocate memory for master processor
void mem_alloc_master_arrays(){
    //ALLOCATE MEMORY FOR ALL PROCESSES CLUSTERS
    total_proc_clusters = num_of_clusters * num_of_processes;
    globalClustersX = (double *)malloc(sizeof(double) * total_proc_clusters);
    globalClustersY = (double *)malloc(sizeof(double) * total_proc_clusters);
    procClusterFinalAssign = (int *)malloc(sizeof(int) * total_proc_clusters);
    if(globalClustersX == NULL || globalClustersY == NULL || procClusterFinalAssign == NULL )
    { perror("clusters malloc"); exit(-1); }

    //ALLOCATE MEMORY FOR FINAL CLUSTERS
    finalClustersX = (double *)malloc(sizeof(double) * num_of_clusters);
    finalClustersY = (double *)malloc(sizeof(double) * num_of_clusters);
    if(finalClustersX == NULL || finalClustersY == NULL)
    { perror("clusters malloc"); exit(-1); }

    //ALLOCATE MEMORY FOR TOTAL TIMES
    proc_total_times = (double *)malloc(sizeof(double) * num_of_processes);
    proc_overhead_times = (double *)malloc(sizeof(double) * num_of_processes);
    proc_omp_over_times = (double *)malloc(sizeof(double) * num_of_processes);
    if(proc_total_times == NULL)
    { perror("total times malloc"); exit(-1); }
}

#ifdef MPI_INCLUDED
// MPI calls made before algorithm
void mpi_before_calc(){

    tmp_time = get_time();
    //BROADCAST INFO TO ALL PROCESSES
    MPI_Bcast(&num_of_clusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_of_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&distance_threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_proc_clusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef _OMP_H
    MPI_Bcast(&num_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    
    mem_alloc_local_arrays();

    //Distribute the points equally to all processors. The points do not
    //change and remain constant throughout the run.
    MPI_Scatterv(pointsX, sendcounts, displs, MPI_DOUBLE,
		recv_pointsX, recv_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Scatterv(pointsY, sendcounts, displs, MPI_DOUBLE,
		recv_pointsY, recv_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    mpi_overhead_time += get_time() - tmp_time;
}

// MPI calls made after algorithm
void mpi_after_calc(){
    tmp_time = get_time();
    //GATHER ALL FINAL CLUSTERS TO ALL PROCESSES
    MPI_Allgather(clustersX, num_of_clusters, MPI_DOUBLE,
        globalClustersX, num_of_clusters, MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgather(clustersY, num_of_clusters, MPI_DOUBLE,
        globalClustersY, num_of_clusters, MPI_DOUBLE, MPI_COMM_WORLD);
    mpi_overhead_time += get_time() - tmp_time;
}
#endif

// Select randomly k points to be the initial clusters
void get_random_initial_clusters(double *clustersX, double *clustersY, double *pointsX, double *pointsY, int elements){
// Completely random
    time_t t;
    srand((unsigned) time(&t) + process_rank);
    // srand(5); //Use for no randomness

    int i, random;
    int *tmpIndexes = (int *)malloc(sizeof(int) * num_of_clusters);

    for(i = 0; i < num_of_clusters; i++) {
        //Get random number
        random = rand() % elements;
        for (int j = 0; j < i; j++){
            if (random == tmpIndexes[j] && j!=i){
                random = rand() % elements;
                j = 0;
            }
        }
            
        clustersX[i] = pointsX[random];
        clustersY[i] = pointsY[random];
        tmpIndexes[i] = random;
    }

    free(tmpIndexes);

// // Each initial cluster is calculated using the mean of the point ranges i*k*N 
// // Calculate cluster centroid using the mean of all assigned points
// 	for(int i = 0; i < num_of_clusters; i++)
// 	{
// 		long double total_x = 0, total_y = 0;
// 		int numOfpoints = 0;
//         long int start = i*(elements/num_of_clusters), end = ((i+1)*(elements/num_of_clusters))-1;
//         if (end > num_of_elements)
//             end = num_of_elements;
            
// #ifdef _OMP_H
// #pragma omp parallel for schedule(static) reduction(+:total_x,total_y,numOfpoints)
// #endif
//         for(int j = start; j < end; j++)
//         {
//             // Calc sum of assigned points
//             numOfpoints++;
//             total_x += pointsX[j];
//             total_y += pointsY[j];
//         }
        
//         //Finally calc mean
// 		if(numOfpoints != 0)
// 		{
// 			clustersX[i] = total_x / numOfpoints;
// 			clustersY[i] = total_y / numOfpoints;
// 		}

// 	}
}

// Get sum total distance of all points from all cluster centroids
long double get_distance_from_clusters(double *p_clustersX, double *p_clustersY, double *p_recv_pointsX, double *p_recv_pointsY, int p_recv_size){
    long double total_distance = 0;

#ifdef _OMP_H
#pragma omp parallel for schedule(static) reduction(+:total_distance) collapse (2)
#endif
    for(int i=0; i<p_recv_size; i++){
        for(int j=0; j<num_of_clusters; j++){
            total_distance += get_dist(p_recv_pointsX[i], p_recv_pointsY[i], p_clustersX[j], p_clustersY[j]);
        }
    }

    return total_distance;
}

// Assign all points to their closest cluster centroid
void assign_to_cluster(double *p_clustersX, double *p_clustersY, double *p_recv_pointsX, double *p_recv_pointsY, int *p_recv_pointsClusterAssign, int p_recv_size){

#ifdef _OMP_H
#pragma omp parallel for schedule(guided)
#endif
    for(int i = 0; i < p_recv_size; i++)
    {
        double min_dist = __INT_MAX__, temp_dist=0;

        // Find closest cluster
        for(int j = 0; j < num_of_clusters; j++)
        {
            temp_dist = get_dist(p_recv_pointsX[i], p_recv_pointsY[i], p_clustersX[j], p_clustersY[j]);

            if(temp_dist <= min_dist)
            {
                min_dist = temp_dist;
                p_recv_pointsClusterAssign[i] = j;
            }
        }
    }
}

// Calculate cluster centroid using the mean of all assigned points
void calc_cluster_centroid(double *p_clustersX, double *p_clustersY, double *p_recv_pointsX, double *p_recv_pointsY, int *p_recv_pointsClusterAssign, int p_recv_size){
	for(int i = 0; i < num_of_clusters; i++)
	{
		long double total_x = 0, total_y = 0;
		int numOfpoints = 0;
        
#ifdef _OMP_H
#pragma omp parallel for schedule(guided) reduction(+:total_x,total_y,numOfpoints)
#endif
        for(int j = 0; j < p_recv_size; j++)
        {
            // Calc sum of assigned points
            if(p_recv_pointsClusterAssign[j] == i)
            {
                numOfpoints++;
                total_x += p_recv_pointsX[j];
                total_y += p_recv_pointsY[j];
            }
        }
        
        //Finally calc mean
		if(numOfpoints != 0)
		{
			p_clustersX[i] = total_x / numOfpoints;
			p_clustersY[i] = total_y / numOfpoints;
		}
	}
}

#ifdef MPI_INCLUDED
// Merge all cluster found from MPI processes.
void merge_clusters(double *p_clustersX, double *p_clustersY, double *p_finalClustersX, double *p_finalClustersY){

    int i,j;
    double tmpClusterX, tmpClusterY;
    double x, y;
    long double temp_dist;
    double min_dist, max_total_dist;
    int    min_dist_ind, max_total_dist_ind;
    int    counter;
    struct double_int local_pair, global_pair;

    //If only one process, assign directly
    if (num_of_processes == 1)
    {
        for (i = 0; i < num_of_clusters; i++)
        {
            p_finalClustersX[i] = p_clustersX[i];
            p_finalClustersY[i] = p_clustersY[i];
            procClusterFinalAssign[i] = i;
        }

    }else{

        indStart = 0; indEnd = total_proc_clusters;
        calc_start_end();
        
        for(i=0; i<total_proc_clusters; i++)
            procClusterFinalAssign[i] = -1;

        // Merge Clusters
        for(i=0; i<num_of_clusters; i++){

            // max_total_dist = 0;
            local_pair.index = 0;
            local_pair.value = 0;

            //Get farthest available cluster
            for(j=indStart; j<indEnd; j++){
                if(procClusterFinalAssign[j] != -1)
                    continue;
                
                temp_dist = 0;

#ifdef _OMP_H
#pragma omp parallel for schedule(guided) reduction(+:temp_dist)
#endif
                for (int j2 = 0; j2 < total_proc_clusters; j2++){
                    if(procClusterFinalAssign[j2] != -1 || j==j2)
                        continue;

                    temp_dist += get_dist(p_clustersX[j], p_clustersY[j], p_clustersX[j2], p_clustersY[j2]);
                }
                temp_dist = temp_dist/total_proc_clusters;
                
                if (temp_dist > local_pair.value){
                    local_pair.value = temp_dist;
                    local_pair.index = j;
                }
            }
            tmp_time = get_time();
            MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
            mpi_overhead_time += get_time() - tmp_time;

            max_total_dist     = global_pair.value; 
            max_total_dist_ind = global_pair.index;

            procClusterFinalAssign[max_total_dist_ind] = i;
            tmpClusterX = p_clustersX[max_total_dist_ind];
            tmpClusterY = p_clustersY[max_total_dist_ind];

            counter = 1;
            p_finalClustersX[i] = p_clustersX[max_total_dist_ind];
            p_finalClustersY[i] = p_clustersY[max_total_dist_ind];

            // Find other closest clusters and merge in one
            for(int count=0; count<num_of_processes-1; count++){

                local_pair.value = __INT_MAX__;
                local_pair.index = -1;

#ifdef _OMP_H
#pragma omp parallel for schedule(guided) reduction(minimum:local_pair)
#endif
                for(j=indStart; j<indEnd; j++){
                    if(procClusterFinalAssign[j] != -1)
                        continue;

                    temp_dist = get_dist(p_clustersX[j], p_clustersY[j], tmpClusterX, tmpClusterY);

                    if(temp_dist < local_pair.value || local_pair.value == __INT_MAX__){
                        local_pair.value = temp_dist;
                        local_pair.index = j;
                    }
                }
                tmp_time = get_time();
                MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
                mpi_overhead_time += get_time() - tmp_time;
                min_dist     = global_pair.value; 
                min_dist_ind = global_pair.index;

                if(min_dist != -1){
                    procClusterFinalAssign[min_dist_ind] = i;
                    p_finalClustersX[i] += p_clustersX[min_dist_ind];
                    p_finalClustersY[i] += p_clustersY[min_dist_ind];
                    counter++;
                }

            }

            if(counter > 0){
                p_finalClustersX[i] = p_finalClustersX[i] / counter;
                p_finalClustersY[i] = p_finalClustersY[i] / counter;
            }
            
        }

#ifdef PLOT
        if (process_rank == 0){
            assign_to_cluster(p_finalClustersX, p_finalClustersY, pointsX, pointsY, pointsClusterAssign, num_of_elements);
        }
#endif
    }
}
#endif

//Print progress
void    print_progress(){
    int tmp2 = (int)trunc((((double)count/((double)MAX_ITERATIONS)))*100);

    if(tmp2 % 5 == 0 && tmp_progress != tmp2)
        printf("%d%% ", tmp2);
    tmp_progress = tmp2;
}

void plot_kmeans(double *pX, double *pY, int *pClAssign, double *clX, double *clY, int numClusters, int numPoints){
#ifdef PLOT
    //Use gnuplot to plot all points and clusters
    char filename_tmp[100];
    sprintf(filename_tmp, "output/%s.txt", name);

    FILE * gnuplotPipe = popen("gnuplot -persist", "w");
    int i;

    if(gnuplotPipe != NULL){
        fprintf(gnuplotPipe, "set key off\n"); //Remove upper right text
        fprintf(gnuplotPipe, "set terminal png size 1000,900 background rgb \"gray75\"\n");
        
        fprintf(gnuplotPipe, "set output '%s.png';\n", filename_tmp);
        
        if (run_enviroment == ENV_MPI || run_enviroment == ENV_HYBRID)
            fprintf(gnuplotPipe, "plot '-' with points palette, '-' using 1:2 with points pointtype 6 pointsize 1.5, '-' using 1:2 with points pointtype 7 pointsize 1.5\n");
        else
            fprintf(gnuplotPipe, "plot '-' with points palette, '-' using 1:2 with points pointtype 7 pointsize 1.5\n");

        // All points
        for ( i = 0; i < numPoints; i++)
        {
            fprintf(gnuplotPipe, "%lf %lf %d\n", pX[i], pY[i], pClAssign[i]);
        }

        fprintf(gnuplotPipe, "e\n");

        //MPI processes clusters
        if (run_enviroment == ENV_MPI || run_enviroment == ENV_HYBRID){
            for ( i = 0; i < total_proc_clusters; i++)
            {
                fprintf(gnuplotPipe, "%lf %lf\n", globalClustersX[i], globalClustersY[i]);
            }

            fprintf(gnuplotPipe, "e\n");
        }

        // Cluster points
        for ( i = 0; i < numClusters; i++)
        {
            fprintf(gnuplotPipe, "%lf %lf\n", clX[i], clY[i]);
        }

        fprintf(gnuplotPipe, "e\n");

        pclose(gnuplotPipe);

        printf("plot '%s.png' created\n",filename_tmp);
    }
#endif
}