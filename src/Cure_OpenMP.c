#include<stdio.h>
#include<omp.h>
#include"Cure.h" 

int main(int argc, char *argv[]){

    int pair_low, pair_high;

    run_enviroment = ENV_OPENMP; //OpenMP
    run_algorithm  = ALG_CURE; //CURE

    // Read the input file and prints info
    get_inputs(argc, argv);
    omp_set_num_threads(num_threads);
    omp_set_max_active_levels(2);//For nested parallelism in update_nnbs
    get_file_content();
    mem_alloc_main();
    init_prints(stdout);
    init_clusters_data();

//TIMER START
    start_time = get_time();

// CURE START
    init_nnbs();
    
    while (remaining_clusters > num_of_clusters)
    {
        print_progress();
        
        get_closest_clusters(&pair_low, &pair_high);

        merge_clusters(pair_low, pair_high);
        
        update_nnbs(pair_low, pair_high);

    }
// CURE END

    end_time   = get_time();
    total_time = end_time - start_time;
//TIMER END

    //Final outputs
    move_final_clusters(clustersX,      clustersY,      clustersValid,
                        finalClustersX, finalClustersY,
                        clustersRepX,   clustersRepY,
                        finalRepX,      finalRepY);
    final_prints(finalClustersX, finalClustersY, stdout);
    printf_to_file(finalClustersX, finalClustersY);
    plot_cure(  pointsX,        pointsY,        pointsClusterAssign, num_of_elements,
                finalClustersX, finalClustersY, num_of_clusters, 
                finalRepX,      finalRepY,      num_of_rep_points);
    free_memory();
}
