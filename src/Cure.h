#include<stdio.h>
#include<stdlib.h>
#include"extras.h"

void    mem_alloc_main();
double  get_clusters_dist(int cl1, int cl2);
void    print_progress();
void    init_clusters_data();
void    init_nnbs();
void    update_nnbs(int pair_low, int pair_high);
void    find_nnb(int i, int *index, double *distance);
void    get_closest_clusters(int *pair_low, int *pair_high);
void    merge_clusters(int pair_low, int pair_high);
void    shift_array(int ind);
void    compute_centroid(int cl1, int cl2, double* newX, double* newY);
void    generate_rep_points(int cl1, int cl2, double newX, double newY);
double  shrink_formula(double p, double c, double a);
void    gather_point_assigns();
void    move_final_clusters(double *gClustersX,    double *gClustersY, int *gClustersValid, 
                            double *gFinalClustX,  double *gFinalClustY,
                            double *gRepX,         double *gRepY,
                            double *gFinalRepX,    double *gFinalRepY);
void    plot_cure(  double *pX,   double *pY,   int *pClAssign, int numPoints,
                    double *clX,  double *clY,  int numClusters, 
                    double *repX, double *repY, int numReps );
void    mem_alloc_worker();
void    mpi_before_calc(int *pair_low, int *pair_high);
void    mpi_after_calc();
void    calc_tmp_valid();

// Memory allocations for master process
void mem_alloc_main(){
    clustersX        = (double *)malloc(sizeof(double) * num_of_elements);
    clustersY        = (double *)malloc(sizeof(double) * num_of_elements);
    clustersSize     = (int *)   malloc(sizeof(int)    * num_of_elements);
    clustersValid    = (int *)   malloc(sizeof(int)    * num_of_elements);
    clustersNNBdist  = (double *)malloc(sizeof(double) * num_of_elements);
    clustersNNBindex = (int *)   malloc(sizeof(int)    * num_of_elements);
    clustersRepX     = (double *)malloc(sizeof(double) * num_of_elements * num_of_rep_points);
    clustersRepY     = (double *)malloc(sizeof(double) * num_of_elements * num_of_rep_points);
    if( clustersValid    == NULL || clustersSize == NULL || clustersNNBdist == NULL || 
        clustersNNBindex == NULL || clustersRepX == NULL || clustersRepY    == NULL )
    { perror("calc clusters malloc"); exit(-1); }

    finalClustersX = (double *)malloc(sizeof(double) * num_of_clusters);
    finalClustersY = (double *)malloc(sizeof(double) * num_of_clusters);
    if(finalClustersX == NULL || finalClustersY == NULL)
    { perror("final clusters malloc"); exit(-1); }

    finalRepX = (double *)malloc(sizeof(double) * num_of_clusters * num_of_rep_points);
    finalRepY = (double *)malloc(sizeof(double) * num_of_clusters * num_of_rep_points);
    if(finalRepX == NULL || finalRepY == NULL)
    { perror("final Rep malloc"); exit(-1); }

#ifdef MPI_INCLUDED
    sendcounts          = (int *)   malloc(sizeof(int)    * num_of_processes);
    displs              = (int *)   malloc(sizeof(int)    * num_of_processes);
    proc_total_times    = (double *)malloc(sizeof(double) * num_of_processes);
    proc_overhead_times = (double *)malloc(sizeof(double) * num_of_processes);
    proc_omp_over_times = (double *)malloc(sizeof(double) * num_of_processes);
    if(finalRepX == NULL           || finalRepY == NULL || proc_total_times == NULL ||
       proc_overhead_times == NULL || proc_omp_over_times == NULL )
    { perror("mpi mallocs"); exit(-1); }
#endif
}

// Get distance between two clusters.
// Distance is defined by the distance of the closest representative points of two clusters.
double get_clusters_dist(int cl1, int cl2){
    double dist;
    int ind1 = cl1*num_of_rep_points,
        ind2 = cl2*num_of_rep_points;

    double min_dist = get_dist(clustersRepX[ind1], clustersRepY[ind1], 
                               clustersRepX[ind2], clustersRepY[ind2]);

    for(int i=0; i<num_of_rep_points; i++){
        for(int j=0; j<num_of_rep_points; j++){
            ind1 = (cl1*num_of_rep_points) + i;
            ind2 = (cl2*num_of_rep_points) + j;

            dist = get_dist(clustersRepX[ind1], clustersRepY[ind1], 
                            clustersRepX[ind2], clustersRepY[ind2]);
            if(dist<min_dist)
                min_dist = dist;
        }
    }

    return min_dist;
}

//Print progress
void    print_progress(){
    int tmp2 = (int)trunc((1-((double)(remaining_clusters-num_of_clusters-1)/((double)num_of_elements)))*100);

    if(tmp2 % 5 == 0 && tmp_progress != tmp2)
        printf("%d%% ", tmp2);
    tmp_progress = tmp2;
}

// Initialize cluster data
void init_clusters_data(){
    for(int i=0; i<num_of_elements; i++){
        clustersX[i]        = pointsX[i];
        clustersY[i]        = pointsY[i];
        clustersSize[i]     = 1;
        clustersValid[i]    = 1;
        pointsClusterAssign[i] = i;
        for(int j=0; j<num_of_rep_points; j++){
            int ind = (i*num_of_rep_points) + j;
            clustersRepX[ind] = pointsX[i];
            clustersRepY[ind] = pointsY[i];
        }
    }
}

// Initialise by finding the nearest neighbouring cluster for each cluster
void init_nnbs(){
    remaining_clusters = num_of_elements;

// In MPI runs indStart and indEnd is the range that each process will loop
    indStart = 0; indEnd = num_of_elements;
    
#ifdef MPI_INCLUDED
calc_start_end();
#endif

#ifdef _OMP_H
#pragma omp parallel for schedule(guided)
#endif

    for(int i = indStart; i<indEnd;i++)
        find_nnb(i, &clustersNNBindex[i], &clustersNNBdist[i]);

#ifdef MPI_INCLUDED
mpi_after_calc();
#endif
}

// Find new nearest neighboring cluster for each cluster
void update_nnbs(int pair_low, int pair_high){
     if (remaining_clusters <= num_of_clusters)
        return;

// In MPI runs indStart and indEnd is the range that each process will loop
    indStart = pair_low+1; indEnd = num_of_elements;

#ifdef MPI_INCLUDED
calc_start_end();
#endif

#ifdef _OMP_H
#pragma omp parallel for schedule(dynamic)
#endif

    for (int i = indStart; i<indEnd; i++){ 
        if (!clustersValid[i]) continue;
        if( clustersNNBindex[i] == pair_low || clustersNNBindex[i] == pair_high )
            find_nnb(i, &clustersNNBindex[i], &clustersNNBdist[i]);
        else if(pair_high < i){
            double dist = get_clusters_dist(i, pair_high);
            if(dist < clustersNNBdist[i])
            { clustersNNBindex[i] = pair_high; clustersNNBdist[i] = dist; }
        }
    }

#ifdef MPI_INCLUDED
mpi_after_calc();
#endif
}

// Find nearest neighboring cluster index and distance for cluster i
void find_nnb(int i, int *index, double *distance){
    
    struct double_int pair;
    pair.value = __INT_MAX__;
    pair.index = 0;

// Nested parallelism
#ifdef _OMP_H
#pragma omp parallel for schedule(guided) reduction(minimum:pair)
#endif
    for(int j=0; j<i; j++){ 
        if (clustersValid[j] == 0) continue;
        double dist = get_clusters_dist(i, j);

        if (dist < pair.value){ 
            pair.value = dist; pair.index = j;}

    } 

    *index = pair.index; *distance = pair.value;
}

// Get closest pairs of clusters
void get_closest_clusters(int *pair_low, int *pair_high){
    struct double_int local_pair, global_pair;
    indStart = 0; indEnd = num_of_elements;

    local_pair.index = 0;
    local_pair.value = __INT_MAX__;

#ifdef MPI_INCLUDED
calc_start_end();
#endif

#ifdef _OMP_H
#pragma omp parallel for schedule(guided) reduction(minimum:local_pair)
#endif

    for(int i = indStart; i<indEnd;i++){
        if (clustersValid[i] == 0) continue;

        if (clustersNNBdist[i] < local_pair.value)
        {
            local_pair.value = clustersNNBdist[i];
            local_pair.index = i;            
        }

    }

    *pair_low  = clustersNNBindex[local_pair.index]; 
    *pair_high = local_pair.index;

#ifdef MPI_INCLUDED
// For MPI find the minimum min_dist of all processes with MPI_Allreduce and share the index of it
    MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    *pair_low = clustersNNBindex[global_pair.index]; 
    *pair_high = global_pair.index;
#endif
}

// Merge cluster pairl_low and pair_high
void merge_clusters(int pair_low, int pair_high){
    // Resulting cluster will be saved in pair_high index
    // while pair_low will be invalidated

    double newX, newY;

    // Compute centroid
    compute_centroid(pair_low, pair_high, &newX, &newY);

    // New size
    clustersSize[pair_high] = clustersSize[pair_low] + clustersSize[pair_high];

    // Generate new rep points
    generate_rep_points(pair_low, pair_high, newX, newY);

    // Save merged cluster center
    clustersX[pair_high] = newX;
    clustersY[pair_high] = newY;

    // Invalidate low index cluster
    clustersValid[pair_low] = 0;
    remaining_clusters--;


#ifdef PLOT
#ifdef _OMP_H
#pragma omp parallel for schedule(guided)
#endif
    // Reassign points for scatter plot
    for(int i=indStart;i<indEnd;i++){
        if(pointsClusterAssign[i] == pair_low)
            pointsClusterAssign[i] = pair_high;
    }
#endif
}

// Calculate new merged centroid
void compute_centroid(int cl1, int cl2, double* newX, double* newY){
    int new_size = clustersSize[cl1] + clustersSize[cl2];
    double a1, a2;

    a1 = (double)clustersSize[cl1]/new_size;
    a2 = (double)clustersSize[cl2]/new_size;

    *newX = a1*clustersX[cl1] + a2*clustersX[cl2];
    *newY = a1*clustersY[cl1] + a2*clustersY[cl2];

}

// Generate new representative points using the rep points of clusters cl1 and cl2
void generate_rep_points(int cl1, int cl2, double newX, double newY){

    int i,j,k, maxInd, ind1, ind2, ind_start, ind;
    double minDist, maxDist;
    double *tmpRepX, *tmpRepY;
    double tmpMin, tmpDist;

    if (num_of_rep_points > 1){
        tmpRepX = (double *)malloc(sizeof(double) * num_of_rep_points * 2);
        tmpRepY = (double *)malloc(sizeof(double) * num_of_rep_points * 2);
    
// Keep all representative points in one array
        for(i=0; i<num_of_rep_points; i++){
            ind1 = (cl1*num_of_rep_points) + i;
            ind2 = (cl2*num_of_rep_points) + i;

            tmpRepX[i]                   = clustersRepX[ind1];
            tmpRepY[i]                   = clustersRepY[ind1];

            tmpRepX[i+num_of_rep_points] = clustersRepX[ind2];
            tmpRepY[i+num_of_rep_points] = clustersRepY[ind2];
            
        }

    // Choose the new representative points
        for(i=0; i<num_of_rep_points; i++){
            ind_start = cl2*num_of_rep_points;
            ind       = ind_start + i;
            maxDist = 0;
            
            for(j=0; j<num_of_rep_points*2; j++){
                if(i == 0)
                    minDist = get_dist(tmpRepX[j], tmpRepY[j], newX, newY);
                else{
                    // Get minimum pairwise distance
                    tmpMin = get_dist(tmpRepX[j]     , tmpRepY[j], 
                                    clustersRepX[ind_start], clustersRepY[ind_start]);
                    for(k=ind_start; k<ind; k++){//Loop at already found final rep points
                        tmpDist = get_dist(tmpRepX[j]     , tmpRepY[j], 
                                        clustersRepX[k], clustersRepY[k]);
                        if(tmpDist < tmpMin)
                            tmpMin = tmpDist;
                    }
                    minDist = tmpMin;
                }
                if(minDist >= maxDist){
                    maxDist = minDist;
                    maxInd = j;
                }
            }
            clustersRepX[ind] = tmpRepX[maxInd];
            clustersRepY[ind] = tmpRepY[maxInd];
        }

    // Shrink new representative points
        for(i=0; i<num_of_rep_points; i++){
            ind_start = cl2*num_of_rep_points;
            ind       = ind_start + i;

            clustersRepX[ind] = shrink_formula(clustersRepX[ind], newX, shrink_fraction);
            clustersRepY[ind] = shrink_formula(clustersRepY[ind], newY, shrink_fraction);
        }

        free(tmpRepX);
        free(tmpRepY);

    } else{
        clustersRepX[cl2] = newX;
        clustersRepY[cl2] = newY;
    }
}

// Return shrinked point
double shrink_formula(double p, double c, double a){
    return p + a * (c - p);
}

// Move final clusters to final structures and reassign points to their final clusters
void move_final_clusters(double *gClustersX,    double *gClustersY, int *gClustersValid, 
                         double *gFinalClustX,  double *gFinalClustY,
                         double *gRepX,         double *gRepY,
                         double *gFinalRepX,    double *gFinalRepY){
    int c = 0;
    int rep_ind, rep_ind_final;
    
    for(int i=0; i<num_of_elements;i++){
        if(gClustersValid[i] == 1){
            gFinalClustX[c] = gClustersX[i];
            gFinalClustY[c] = gClustersY[i];
            for(int j=0; j<num_of_rep_points; j++){
                rep_ind       = (i*num_of_rep_points) + j;
                rep_ind_final = (c*num_of_rep_points) + j;
                gFinalRepX[rep_ind_final] = gRepX[rep_ind];
                gFinalRepY[rep_ind_final] = gRepY[rep_ind];
            }
            for(int k=0; k<num_of_elements; k++){
                if (pointsClusterAssign[k] == i)
                    pointsClusterAssign[k] = c;
            }
            c++;
        }
    }
}

void plot_cure(double *pX,   double *pY,   int *pClAssign, int numPoints,
               double *clX,  double *clY,  int numClusters, 
               double *repX, double *repY, int numReps ){
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
        fprintf(gnuplotPipe, "plot '-' with points palette, '-' using 1:2 with points pointtype 7 pointsize 1.5, '-' using 1:2 with points pointtype 6 pointsize 1.5\n");
        
        // All points
        for ( i = 0; i < numPoints; i++)
        {
            fprintf(gnuplotPipe, "%lf %lf %d\n", pX[i], pY[i], pClAssign[i]);
        }

        fprintf(gnuplotPipe, "e\n");

        // Cluster points
        for ( i = 0; i < numClusters; i++)
        {
            fprintf(gnuplotPipe, "%lf %lf\n", clX[i], clY[i]);
        }

        fprintf(gnuplotPipe, "e\n");

        //Representative points
        for ( i = 0; i < numReps*numClusters; i++)
        {
            fprintf(gnuplotPipe, "%lf %lf\n", repX[i], repY[i]);
        }

        fprintf(gnuplotPipe, "e\n");
        

        pclose(gnuplotPipe);

        printf("plot '%s.png' created\n",filename_tmp);
    }
#endif
}

#ifdef MPI_INCLUDED
// Share input parameters and allocate memory for all workers
void mem_alloc_worker(){
    tmp_time = get_time();
    MPI_Bcast(&num_of_elements  , 1 , MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_of_rep_points, 1 , MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_of_processes , 1 , MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_of_clusters  , 1 , MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&shrink_fraction  , 1 , MPI_DOUBLE, 0, MPI_COMM_WORLD);
#ifdef _OMP_H
    MPI_Bcast(&num_threads      , 1 , MPI_INT, 0, MPI_COMM_WORLD);
#endif

    if(process_rank != 0){
        sendcounts       = (int *)   malloc(sizeof(int)    * num_of_processes);
        displs           = (int *)   malloc(sizeof(int)    * num_of_processes);
        clustersX        = (double *)malloc(sizeof(double) * num_of_elements);
        clustersY        = (double *)malloc(sizeof(double) * num_of_elements);
        clustersSize     = (int *)   malloc(sizeof(int)    * num_of_elements);
        clustersValid    = (int *)   malloc(sizeof(int)    * num_of_elements);
        clustersNNBdist  = (double *)malloc(sizeof(double) * num_of_elements);
        clustersNNBindex = (int *)   malloc(sizeof(int)    * num_of_elements);
        clustersRepX     = (double *)malloc(sizeof(double) * num_of_elements * num_of_rep_points);
        clustersRepY     = (double *)malloc(sizeof(double) * num_of_elements * num_of_rep_points);
        pointsClusterAssign = (int *)   malloc(sizeof(int)    * num_of_elements);

        if( clustersValid == NULL || clustersNNBdist == NULL || clustersNNBindex == NULL || 
            clustersRepX  == NULL || clustersRepY    == NULL || sendcounts == NULL       ||
            displs        == NULL || clustersSize    == NULL || clustersX    == NULL  || 
            clustersY     == NULL || pointsClusterAssign    == NULL )
        { perror("calc clusters malloc"); exit(-1); }
    }

    // Share all data of initialized clusters
    MPI_Bcast(pointsClusterAssign , num_of_elements                    , MPI_INT   , 0, MPI_COMM_WORLD);
    MPI_Bcast(clustersX       , num_of_elements                    , MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(clustersY       , num_of_elements                    , MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(clustersSize    , num_of_elements                    , MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(clustersValid   , num_of_elements                    , MPI_INT   , 0, MPI_COMM_WORLD);
    MPI_Bcast(clustersRepX    , num_of_elements * num_of_rep_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(clustersRepY    , num_of_elements * num_of_rep_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    mpi_overhead_time += get_time() - tmp_time;
}

// All processes gather the new calculated nearest neighboring clusters data
void mpi_after_calc(){
    tmp_time = get_time();
    MPI_Allgatherv(MPI_IN_PLACE , sendcounts[process_rank], MPI_DOUBLE,
                clustersNNBdist , sendcounts, displs      , MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE , sendcounts[process_rank], MPI_INT,
                clustersNNBindex, sendcounts, displs      , MPI_INT,    MPI_COMM_WORLD);
    mpi_overhead_time += get_time() - tmp_time;
}

void gather_point_assigns(){
#ifdef PLOT
    tmp_time = get_time();
    indStart = 0; indEnd = num_of_elements;
    calc_start_end();
    if (process_rank == 0){
        MPI_Gatherv(MPI_IN_PLACE, sendcounts[process_rank], MPI_INT,
             pointsClusterAssign, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    } else{
        MPI_Gatherv(pointsClusterAssign+indStart, sendcounts[process_rank], MPI_INT,
             pointsClusterAssign, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    }
    mpi_overhead_time += get_time() - tmp_time;
#endif
}
#endif