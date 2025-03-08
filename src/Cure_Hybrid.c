#include<stdio.h>
#include<mpi.h>
#include<omp.h>
#include"Cure.h" 

int main(int argc, char *argv[]){
    int provided;

    // Initialize the MPI environment
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    int pair_low, pair_high;

    run_enviroment = ENV_HYBRID; //Hybrid
    run_algorithm  = ALG_CURE; //CURE

    // Master process reads the input file and prints info
    if(process_rank == 0){
        get_inputs(argc, argv);
        get_file_content();
        mem_alloc_main();
        init_prints(stdout);
        init_clusters_data();
    }

    // All processes wait until master finishes reading file
    MPI_Barrier(MPI_COMM_WORLD);

//TIMER START
    start_time = get_time();
    
    mem_alloc_worker();
    
    omp_set_num_threads(num_threads);
    omp_set_max_active_levels(2);//For nested parallelism in update_nnbs

// CURE START
    init_nnbs();
    
    while (remaining_clusters > num_of_clusters)
    {
        if (process_rank == 0)
            print_progress();
            
        get_closest_clusters(&pair_low, &pair_high);
        
        merge_clusters(pair_low, pair_high);

        update_nnbs(pair_low, pair_high);
    }
// CURE END

    gather_point_assigns();

    end_time   = get_time();
    total_time = end_time - start_time;
//TIMER END

    //Gather all times from processes
    gather_times();

    //Final outputs
    if(process_rank == 0){
        move_final_clusters(clustersX,      clustersY,      clustersValid,
                            finalClustersX, finalClustersY,
                            clustersRepX,   clustersRepY,
                            finalRepX,      finalRepY);
        final_prints(finalClustersX, finalClustersY, stdout);
        printf_to_file(finalClustersX, finalClustersY);

        plot_cure(  pointsX,        pointsY,        pointsClusterAssign, num_of_elements,
                    finalClustersX, finalClustersY, num_of_clusters, 
                    finalRepX,      finalRepY,      num_of_rep_points);
    }
    free_memory();
    MPI_Finalize();
}
