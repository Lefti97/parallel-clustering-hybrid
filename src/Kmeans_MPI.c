#include <stdio.h>
#include <mpi.h>
#include "Kmeans.h" 

int main(int argc, char *argv[]){

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &num_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    run_enviroment = ENV_MPI; //MPI
    run_algorithm  = ALG_KMEANS; //Kmeans

    long double init_distance, new_distance;

    // Master process reads the input file and prints info
    if(process_rank == 0){
        get_inputs(argc, argv);
        get_file_content();
        mem_alloc_master_arrays();
        init_prints(stdout);
    }

    // All processes wait until master finishes reading file
    MPI_Barrier(MPI_COMM_WORLD);

//TIMER START
    start_time = get_time();

    // Share input parameters and distribute points equally
    mpi_before_calc();

//Kmeans START
    get_random_initial_clusters(clustersX, clustersY, recv_pointsX, recv_pointsY, recv_size);

	while(count < MAX_ITERATIONS){

        init_distance = get_distance_from_clusters(clustersX, clustersY, recv_pointsX, recv_pointsY, recv_size);
        assign_to_cluster(clustersX, clustersY, recv_pointsX, recv_pointsY, recv_pointsClusterAssign, recv_size);		
        calc_cluster_centroid(clustersX, clustersY, recv_pointsX, recv_pointsY, recv_pointsClusterAssign, recv_size);
        if(count != 0)
            new_distance = get_distance_from_clusters(clustersX, clustersY, recv_pointsX, recv_pointsY, recv_size);

        // If difference of new distance reaches threshold then stop 
        if(fabs(init_distance - new_distance) <= distance_threshold)
            break;
        count++;   

        if (process_rank == 0)
            print_progress();
    }
//Kmeans END

    // Master process gathers all clusters found by workers
    mpi_after_calc();

    // Merges all clusters found into k final clusters
    merge_clusters(globalClustersX, globalClustersY, finalClustersX, finalClustersY);
    
    end_time = get_time();
    total_time = end_time - start_time;
//TIMER END

    //Gather all times from processes
    gather_times();
        
    //Final outputs
    if(process_rank == 0){
        final_prints(finalClustersX, finalClustersY, stdout);
        printf_to_file(finalClustersX, finalClustersY);
        plot_kmeans(pointsX, pointsY, pointsClusterAssign, finalClustersX, finalClustersY, num_of_clusters, num_of_elements);
    }
    free_memory();
    MPI_Finalize();
}