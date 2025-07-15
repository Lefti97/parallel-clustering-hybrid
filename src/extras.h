#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

#define PLOT
#define FILE_OUT

#define ENV_SERIAL 0
#define ENV_OPENMP 1
#define ENV_MPI 2
#define ENV_HYBRID 3

#define ALG_KMEANS 1
#define ALG_CURE 2

#define TIME_MULT 1
#define MAX_ITERATIONS 100

//Global variables
int run_enviroment        = 0; //1.OpenMP 2.MPI 3.Hybrid
int run_algorithm         = 0; //1.Kmeans 2.CURE

int    num_of_clusters    = 0; //Number of clusters
long int    num_of_elements    = 0; //Number of elements in input file
double distance_threshold = 0; //Distance threshold
char   *filename;              //Input filename
char   *exename;               //Executable name
char   name[100];               //Used for filename of outputs

int    num_of_processes      = 1; //MPI number of processes
int    process_rank          = 0; //MPI current process number
int    total_proc_clusters   = 0; //MPI total clusters across processes
int    *sendcounts;               //MPI processor number of data
int    *displs;                   //MPI processor offset
int    recv_size             = 0; //MPI process number of elements
double *proc_total_times;         //MPI Counter total time of each process
double *proc_overhead_times;      //MPI Counter overhead time of each process
int    count                 = 0;

int    num_threads        = 0; //OpenMP number of threads
double *proc_omp_over_times;   //OpenMP Counter overhead time of each process

double start_time;             //Counter start time
double end_time;               //Counter end time
double total_time;             //Counter total time
double io_time;                //Counter IO time
double mpi_overhead_time = 0; 
double omp_overhead_time = 0;

double tmp_time;
double tmp_time2 = 0;
time_t start_timestamp;
struct tm *local_time;

//Global clusters data
double *globalClustersX;
double *globalClustersY;
int    *procClusterFinalAssign;
//Final clusters data
double *finalClustersX;
double *finalClustersY;
//Global points data
double *pointsX;
double *pointsY;
int    *pointsClusterAssign; //Index of clustersX, clustersY

//Processing clusters data
double   *clustersX;
double   *clustersY;

//Received points data
double *recv_pointsX;
double *recv_pointsY;
int    *recv_pointsClusterAssign; //Index of clustersX, clustersY

//CURE variables
int      *clustersSize;
int      *clustersValid;     // 1=Valid, 0=Invalid
double   *clustersNNBdist;
int      *clustersNNBindex;
double   *clustersRepX;
double   *clustersRepY;
double   *finalRepX;
double   *finalRepY;
int      *tmpValidInd;
int      num_of_rep_points;
double   shrink_fraction;
int      remaining_clusters;
int      tmpInd;

int tmp_progress;
int indStart, indEnd;

struct double_int {
    double value;
    int index;
};
#ifdef _OMP_H
#pragma omp declare reduction(minimum : struct double_int : omp_out = omp_in.value < omp_out.value ? omp_in : omp_out) initializer (omp_priv={__INT_MAX__, 0})
#endif

double get_time();
double get_dist(double x1, double y1, double x2, double y2);
void   get_inputs(int p_argc, char *p_argv[]);
void   create_namefile();
int    get_number_of_elements(FILE* fp);
void   get_file_content();
void   get_elements(FILE* fp, double *pointsX, double *pointsY, int *pointsClusterAssign);
void   init_prints(FILE *stream);
void   final_prints(double *p_clusterX, double *p_clusterY, FILE *stream);
void   printf_to_file(double *p_clusterX, double *p_clusterY);
void   get_scatter_variables(int *sendcounts, int *displs, int elements, int start);
void   get_scatter_variables_update_nnbs(int *sendcounts, int *displs);
void   calc_start_end();
void   gather_times();
void   free_memory();

// Returns clock time depending on the type of enviroment that is running
double get_time(){

    // MPI or Hybrid
    #ifdef MPI_INCLUDED
    return MPI_Wtime();
    #endif

    // Openpm only
    #ifdef _OMP_H
    return omp_get_wtime();
    #endif

    // Serial
    return (double)clock()/CLOCKS_PER_SEC;
        
}

// Get Euclidian distance of two 2D points
double get_dist(double x1, double y1, double x2, double y2){
    double x, y, res;
    
    x = fabs(x1 - x2);
    y = fabs(y1 - y2);
    res = sqrt((x*x) + (y*y));

    return res;
}

// Get input parameters depending on type of algorithm and enviroment
void get_inputs(int p_argc, char *p_argv[]){
    int argc_needed = 0;

    start_time = get_time();

    time(&start_timestamp);
    local_time = localtime(&start_timestamp);
    
    switch (run_algorithm)
    {
    case ALG_KMEANS:
        
        if      (run_enviroment == ENV_MPI || run_enviroment == ENV_SERIAL)
            argc_needed = 4;
        else if (run_enviroment == ENV_OPENMP  || run_enviroment == ENV_HYBRID)
            argc_needed = 5;

        if(p_argc == argc_needed){
            exename = p_argv[0];
            filename = p_argv[1];
            num_of_clusters = atoi(p_argv[2]);
            distance_threshold = atof(p_argv[3]);
            if (run_enviroment == ENV_OPENMP || run_enviroment == ENV_HYBRID)
                num_threads = atoi(p_argv[4]);
            
        }
        else{
            printf("Arguments needed: 1)Filename \n2)Number of Clusters \n3)Distance Threshold \n");
            if (run_enviroment == ENV_OPENMP || run_enviroment == ENV_HYBRID)
                printf("4)Number of OpenMP threads\n");
            exit(0);
        }
        
        if(num_of_clusters <= 0){
            perror("no clusters given"); 
            exit(1); 
        }
        if(distance_threshold < 0){
            perror("distance threshold error"); 
            exit(2); 
        }
        if (run_enviroment == ENV_OPENMP || run_enviroment == ENV_HYBRID){
            if(num_threads <= 0){
                perror("no threads given"); exit(3); 
            }
        }
   
        break;

    case ALG_CURE:
        if      (run_enviroment == ENV_MPI     || run_enviroment == ENV_SERIAL)
            argc_needed = 5;
        else if (run_enviroment == ENV_OPENMP  || run_enviroment == ENV_HYBRID)
            argc_needed = 6;

        if(p_argc == argc_needed){
            if (run_enviroment == ENV_OPENMP || run_enviroment == ENV_HYBRID)
            {    num_threads = atoi(p_argv[5]); }
            shrink_fraction   = atof(p_argv[4]);
            num_of_rep_points = atoi(p_argv[3]);
            num_of_clusters = atoi(p_argv[2]);
            filename = p_argv[1];
            exename = p_argv[0];
        }
        else{
            printf("Arguments needed: \n1)Filename\n2)Number of Clusters\n3)Representative Points\n4)Shrink Fraction\n");
            if (run_enviroment == ENV_OPENMP || run_enviroment == ENV_HYBRID)
                printf("5)Number of OpenMP threads\n");
            exit(0);
        }

        if(num_of_clusters <= 0)
        {   perror("no clusters given"); exit(-1);          }
        if(num_of_rep_points <= 0)
        {   perror("number of representative points error"); exit(-1);   }
        if(shrink_fraction < 0 || shrink_fraction > 1.0)
        {   perror("shrink fraction error (0.0 to 1.0)"); exit(-1);   }
        if (run_enviroment == ENV_OPENMP || run_enviroment == ENV_HYBRID)
            if(num_threads <= 0)
            {   perror("no threads given"); exit(-1);           }
    
    default:
        break;
    }
}

// Create the name of the files to be created
void   create_namefile(){

    if      (run_algorithm == ALG_KMEANS)
        sprintf(name, "Kmeans_");
    else if (run_algorithm == ALG_CURE)
        sprintf(name, "CURE_%dR_", num_of_rep_points);

    sprintf(name, "%s%dC_", name, num_of_clusters);

    if      (num_of_elements < 1000)
        sprintf(name, "%s%dN_", name, num_of_elements);
    else if (num_of_elements < 1000000)
        sprintf(name, "%s%dKN_", name, num_of_elements/1000);
    else if (num_of_elements < 1000000000)
        sprintf(name, "%s%dMN_", name, num_of_elements/1000000);
    else if (num_of_elements < 1000000000000)
        sprintf(name, "%s%dBN_", name, num_of_elements/1000000000);

    if      (run_enviroment == ENV_OPENMP)
        sprintf(name, "%sOpenMP_%dT_", name, num_threads);
    else if (run_enviroment == ENV_MPI)
        sprintf(name, "%sMPI_%dP_", name, num_of_processes);
    else if (run_enviroment == ENV_HYBRID)
        sprintf(name, "%sHybrid_%dP_%dT_", name, num_of_processes, num_threads );
    else if (run_enviroment == ENV_SERIAL)
        sprintf(name, "%sSerial_", name);


    sprintf(name, "%s%dY%dM%dD%dH%dM%dS", name,
            local_time->tm_year+1900, local_time->tm_mon+1, local_time->tm_mday,
            local_time->tm_hour, local_time->tm_min,   local_time->tm_sec);
}

// Counts number of elements in input file
int get_number_of_elements(FILE* fp){
    int c = 0;
    int num_of_elements = 0;

    while(!feof(fp))
    {
        c = fgetc(fp);
        if(c == '\n')
        {
            num_of_elements++;
        }
    }

    return num_of_elements;
}

// Open input file to read all points inside it
void get_file_content(){
    //OPEN FILE
    FILE* fp = fopen(filename, "r");
    if(!fp)
    {   perror("file open"); exit(-1); }

    //GET NUMBER OF ELEMENTS IN FILE
    num_of_elements = get_number_of_elements(fp);

    //ALLOCATE MEMORY FOR  POINTS
    pointsX =             (double *)malloc(sizeof(double) * num_of_elements);
    pointsY =             (double *)malloc(sizeof(double) * num_of_elements);
    pointsClusterAssign = (int *)   malloc(sizeof(int)    * num_of_elements);
    if(pointsX == NULL || pointsY == NULL || pointsClusterAssign == NULL)
    { perror("points malloc"); exit(-1); }

    //GET FILE ELEMENTS LOCALLY
    fseek(fp, 0, SEEK_SET);
    get_elements(fp, pointsX, pointsY, pointsClusterAssign);
    fclose(fp);

    end_time = get_time();
    io_time = end_time - start_time;

    create_namefile();
}

// Initial prints before algorithm is run
void init_prints(FILE *stream){
    //PRINTS
    fprintf(stream,"----------START------------\n");
    fprintf(stream,"%s", asctime(local_time));
    if      (run_algorithm == ALG_KMEANS)
        fprintf(stream,"Kmeans ");
    else if (run_algorithm == ALG_CURE)
        fprintf(stream,"Cure ");
    
    if      (run_enviroment == ENV_HYBRID)
        fprintf(stream,"Hybrid (MPI+OpenMP)\n");
    else if (run_enviroment == ENV_MPI)
        fprintf(stream,"MPI\n");
    else if (run_enviroment == ENV_OPENMP)
        fprintf(stream,"OpenMP\n");
    else if (run_enviroment == ENV_SERIAL)
        fprintf(stream,"Serial\n");
    fprintf(stream,"----------INPUTS-----------\n");
    fprintf(stream,"Number of points in file %s: %d\n", filename, num_of_elements);
    fprintf(stream,"Number of clusters: %d\n", num_of_clusters); fflush(stream);
    if      (run_algorithm == ALG_KMEANS){
        fprintf(stream,"Distance threshold: %f\n", distance_threshold);
        fprintf(stream,"Main Loop Iterations: %ld\n", MAX_ITERATIONS);
    }
    else if (run_algorithm == ALG_CURE){
        fprintf(stream,"Number of representative points: %d\n", num_of_rep_points);
        fprintf(stream,"Shrink fraction: %f\n", shrink_fraction);
        fprintf(stream,"Main Loop Iterations: %ld\n", num_of_elements-num_of_clusters); }

#ifdef MPI_INCLUDED
    fprintf(stream,"MPI processes: %d\n", num_of_processes);
#endif

#ifdef _OMP_H
    fprintf(stream,"Number of OpenMP threads per process: %d\n", num_threads);
#endif
}

// Final prints after algorithm is run
void final_prints(double *p_clusterX, double *p_clusterY, FILE *stream){
    int    i;
    double max_time  = 0;
    int    max_time_ind = 0;
    double sum_clust_t = 0, sum_mpi_t = 0, sum_omp_t = 0, sum_total_t = 0;

    fprintf(stream,"\n----------RESULTS----------\n");
    for(i=0;i<num_of_clusters;i++){
        fprintf(stream,"Cluster Centroid %d: ( %f , %f )\n", i, p_clusterX[i], p_clusterY[i]);
        if (run_algorithm == ALG_CURE){
            for (int j = 0; j < num_of_rep_points; j++){
                int rep_ind = (i*num_of_rep_points) + j;
                fprintf(stream,"-Cluster %d Representative Point %d ( %f , %f )\n", i , j, finalRepX[rep_ind], finalRepY[rep_ind]);
            } 
        }
        
    }
    fprintf(stream,"----------TIMES------------\n");
    fprintf(stream,"IO\t  %f (not included in total time)\n", io_time);

    if (run_enviroment == ENV_HYBRID || run_enviroment == ENV_MPI){
        fprintf(stream,"Proc\tClustering\tMPI Overhead\tTotal\n");
        for(i=0;i<num_of_processes;i++){
            fprintf(stream,"%d\t%f\t%f\t%f\t\n", 
                    i, proc_total_times[i]-proc_overhead_times[i], 
                    proc_overhead_times[i], 
                    proc_total_times[i]);
            if(max_time < proc_total_times[i]){
                max_time     = proc_total_times[i];
                max_time_ind = i;
            }
            sum_clust_t += proc_total_times[i]-proc_overhead_times[i]; 
            sum_mpi_t   += proc_overhead_times[i]; 
            sum_total_t += proc_total_times[i];
        }
        fprintf(stream,"\n");
        fprintf(stream,"-\t%f\t%f\t%f\tTotals\n", 
                    sum_clust_t, 
                    sum_mpi_t,
                    sum_total_t);
        fprintf(stream,"-\t%f\t%f\t%f\tMean\n", 
                    sum_clust_t/num_of_processes, 
                    sum_mpi_t/num_of_processes,
                    sum_total_t/num_of_processes);
        fprintf(stream,"%d\t%f\t%f\t%f\tLast Processor\n", 
                    max_time_ind, max_time-proc_overhead_times[max_time_ind], 
                    proc_overhead_times[max_time_ind],
                    max_time);
        
    }
    else{
        fprintf(stream,"Total %f\n", total_time);
    }

    fprintf(stream,"----------END-------------\n");
}

// Create file in output folder from init_prints and final_prints
void printf_to_file(double *p_clusterX, double *p_clusterY){
#ifdef FILE_OUT
    char filename_tmp[100];
    sprintf(filename_tmp, "output/%s.txt", name);

    FILE *file = fopen(filename_tmp, "w"); 
    if (file != NULL) {
        init_prints(file);
        final_prints(p_clusterX, p_clusterY, file);
        fclose(file); 
        printf("log '%s' created\n",filename_tmp);
    }  
#endif
}

// Calculate number of points and offset for each MPI process
void get_scatter_variables(int *sendcounts, int *displs, int elements, int start){
    int i;
    int excess_elements = elements % num_of_processes;
    int minSize         = elements / num_of_processes;
    int tmp1            = excess_elements;
    
    for(i=0; i<num_of_processes; i++){
        sendcounts[i] = 0;
        sendcounts[i] = minSize;
        if(tmp1 > 0){
            sendcounts[i] += 1;
            --tmp1;
        }   
    }

    displs[0] = start;
    for(i=1; i<num_of_processes; i++){
        displs[i] = displs[i-1] + sendcounts[i-1];
    }
        
}

void    calc_start_end(){
    get_scatter_variables(sendcounts, displs, indEnd-indStart, indStart);
    
    indStart = displs[process_rank];
    if (process_rank < num_of_processes-1)
        indEnd   = displs[process_rank+1];
}

// Read all points from input file
void get_elements(FILE* fp, double *pointsX, double *pointsY, int *pointsClusterAssign){
    int    i = 0;
    double point_x=0, point_y=0;

    while(fscanf(fp, "%lf %lf", &point_x, &point_y) != EOF)
    {
        pointsX[i]             = point_x;
        pointsY[i]             = point_y;
        pointsClusterAssign[i] = 0; //Initialize to cluster 0
        i++;
    }
}

// Gather total and overhead times from all processes
void gather_times(){
#ifdef MPI_INCLUDED
    MPI_Gather(&total_time, 1, MPI_DOUBLE,
          proc_total_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&mpi_overhead_time, 1, MPI_DOUBLE,
              proc_overhead_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
}

// Free all allocated memory
void free_memory(){
    free(sendcounts);
    free(displs);
    free(proc_total_times);
    free(proc_overhead_times);
    free(proc_omp_over_times);
    free(globalClustersX);
    free(globalClustersY);
    free(procClusterFinalAssign);
    free(finalClustersX);
    free(finalClustersY);
    free(pointsX);
    free(pointsY);
    free(pointsClusterAssign);
    free(clustersX);
    free(clustersY);
    free(recv_pointsX);
    free(recv_pointsY);
    free(recv_pointsClusterAssign);
    free(clustersSize);
    free(clustersValid);
    free(clustersNNBdist);
    free(clustersNNBindex);
    free(clustersRepX);
    free(clustersRepY);
    free(finalRepX);
    free(finalRepY);
    free(tmpValidInd);
}