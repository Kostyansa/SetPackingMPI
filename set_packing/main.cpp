#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string>
#include <vector>
#include <bitset>
#include <mpi.h>

// Function to read input from file
void read_input(const char *filename, int *n, int *m, bool ***matrix) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d %d", n, m);
    *matrix = (bool **)calloc(*n, sizeof(bool *));
    for (int i = 0; i < *n; ++i) {
        (*matrix)[i] = (bool *)calloc(*m, sizeof(bool));
        for (int j = 0; j < *m; ++j) {
            int value;
            fscanf(file, "%d", &value);
            (*matrix)[i][j] = value;
        }
    }

    fclose(file);
}

bool do_intersect(const bool *a, const bool *b, int m) {
    for (int i = 0; i < m; ++i) {
        if (a[i] and b[i]) {
            return true;
        }
    }
    return false;
}

int set_packing_exact(bool **matrix, int n, int m, long long int start, long long int end) {
    int max = 0;
    for (long long i = start; i < end; i++) {
        long long variation = i;
        bool flag = true;
        int count = 0;
        for (int i = 0; (i < n) && flag; i++) {
            if (( variation >> i) & 1) {
                for (int j = i + 1; (j < n) && flag; j++) {
                    if (( variation >> j) & 1) {
                        if (do_intersect(matrix[i], matrix[j], m)) {
                            flag &= false;
                        }
                    }
                }
                count++;
            }
        }
        if (flag) {
            max = max < count ? count : max;
        }
    }

    return max;
}

int main(int argc, char **argv) {
    std::string path ="/Users/kostyansa/input2";
    const char* file_name = path.c_str();

    int n, m, rank, world_size, local_result, global_result;
    bool **matrix;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0) {
        read_input(file_name, &n, &m, &matrix);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank != 0) {
        matrix = (bool **)calloc(n, sizeof(bool *));
        for (int i = 0; i < n; ++i) {
            matrix[i] = (bool *)calloc(m, sizeof(bool));
        }
    }

    for (int i = 0; i < n; ++i) {
        MPI_Bcast(matrix[i], m, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    //Divide elements
    long long int size = 1LL << n;
    
    long long int elements_per_process = size / world_size;
    long long int start = rank * elements_per_process;
    long long int end = (rank == world_size - 1) ? size : (rank + 1) * elements_per_process;

    local_result = set_packing_exact(matrix, n, m, start, end);

    MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Set packing size: %d\n", global_result);

        for (int i = 0; i < n; ++i) {
            free(matrix[i]);
        }
        free(matrix);
    }

    MPI_Finalize();
    return 0;
}

