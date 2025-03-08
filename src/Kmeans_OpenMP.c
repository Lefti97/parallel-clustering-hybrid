#include <stdio.h>
#include <omp.h>
#include "Kmeans.h" 

int main(int argc, char *argv[]){
    
    run_enviroment = ENV_OPENMP; //OpenMP
    run_algorithm  = ALG_KMEANS; //Kmeans

    long double init_distance, new_distance;

    // Read the input file and prints info
    get_inputs(argc, argv);
    get_file_content();
    omp_set_num_threads(num_threads);
    init_prints(stdout);

//TIMER START
    start_time = get_time();
    mem_alloc_local_arrays();

//Kmeans START
    get_random_initial_clusters(clustersX, clustersY, pointsX, pointsY, num_of_elements);

	while(count < MAX_ITERATIONS){

        init_distance = get_distance_from_clusters(clustersX, clustersY, pointsX, pointsY, num_of_elements);
        assign_to_cluster(clustersX, clustersY, pointsX, pointsY, pointsClusterAssign, num_of_elements);		
        calc_cluster_centroid(clustersX, clustersY, pointsX, pointsY, pointsClusterAssign, num_of_elements);
        if(count != 0)
            new_distance = get_distance_from_clusters(clustersX, clustersY, pointsX, pointsY, num_of_elements);
        
        // If difference of new distance reaches threshold then stop 
        if(fabs(init_distance - new_distance) <= distance_threshold)
            break;
		count++;

        print_progress();
    }
//Kmeans END

    end_time   = get_time();
    total_time = end_time - start_time;
//TIMER END

    //Final outputs
    final_prints(clustersX, clustersY, stdout);
    printf_to_file(clustersX, clustersY);
    plot_kmeans(pointsX, pointsY, pointsClusterAssign, clustersX, clustersY, num_of_clusters, num_of_elements);

    free_memory();
}

