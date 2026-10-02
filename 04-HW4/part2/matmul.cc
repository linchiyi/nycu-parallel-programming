#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    /* TODO: The data is stored in a_mat and b_mat.
     * You need to allocate memory for a_mat_ptr and b_mat_ptr,
     * and copy the data from a_mat and b_mat to a_mat_ptr and b_mat_ptr, respectively.
     * You can use any size and layout you want if they provide better performance.
     * Unambitiously copying the data is also acceptable.
     *
     * The matrix multiplication will be performed on a_mat_ptr and b_mat_ptr.
     */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Broadcast dimensions to all processes
    int dims[3] = {n, m, l};
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    n = dims[0];
    m = dims[1];
    l = dims[2];

    // Allocate memory for matrices (aligned for better cache performance)
    *a_mat_ptr = new int[n * m];
    *b_mat_ptr = new int[m * l];

    // Broadcast matrices from rank 0 to all processes
    if (rank == 0)
    {
        std::memcpy(*a_mat_ptr, a_mat, n * m * sizeof(int));
        std::memcpy(*b_mat_ptr, b_mat, m * l * sizeof(int));
    }

    MPI_Bcast(*a_mat_ptr, n * m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    /* TODO: Perform matrix multiplication on a_mat and b_mat. Which are the matrices you've
     * constructed. The result should be stored in out_mat, which is a continuous memory placing n *
     * l elements of int. You need to make sure rank 0 receives the result.
     */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate rows per process
    int rows_per_proc = n / size;
    int extra_rows = n % size;
    int start_row = rank * rows_per_proc + (rank < extra_rows ? rank : extra_rows);
    int end_row = start_row + rows_per_proc + (rank < extra_rows ? 1 : 0);
    int local_rows = end_row - start_row;

    // Allocate local result buffer
    int *local_result = new int[local_rows * l];

    // Adaptive block size based on matrix dimensions
    const int BLOCK_I = std::min(32, local_rows);
    const int BLOCK_J = std::min(32, l);
    const int BLOCK_K = std::min(64, m);
    
    // Initialize result to zero
    std::memset(local_result, 0, local_rows * l * sizeof(int));
    
    // Multi-level blocking (ikj order with blocking)
    for (int ii = 0; ii < local_rows; ii += BLOCK_I)
    {
        int i_end = std::min(ii + BLOCK_I, local_rows);
        
        for (int kk = 0; kk < m; kk += BLOCK_K)
        {
            int k_end = std::min(kk + BLOCK_K, m);
            
            for (int jj = 0; jj < l; jj += BLOCK_J)
            {
                int j_end = std::min(jj + BLOCK_J, l);
                
                // Inner loops - compute block
                for (int i = ii; i < i_end; ++i)
                {
                    int global_y = start_row + i;
                    const int *a_row = &a_mat[global_y * m];
                    int *c_row = &local_result[i * l];
                    
                    for (int k = kk; k < k_end; ++k)
                    {
                        int a_val = a_row[k];
                        const int *b_col = &b_mat[k];
                        
                        // Unroll inner loop for better performance
                        int j = jj;
                        for (; j + 3 < j_end; j += 4)
                        {
                            c_row[j] += a_val * b_col[(j) * m];
                            c_row[j + 1] += a_val * b_col[(j + 1) * m];
                            c_row[j + 2] += a_val * b_col[(j + 2) * m];
                            c_row[j + 3] += a_val * b_col[(j + 3) * m];
                        }
                        // Handle remaining elements
                        for (; j < j_end; ++j)
                        {
                            c_row[j] += a_val * b_col[j * m];
                        }
                    }
                }
            }
        }
    }

    // Gather results at rank 0 using MPI_Gatherv for better performance
    int *recvcounts = nullptr;
    int *displs = nullptr;
    
    if (rank == 0)
    {
        recvcounts = new int[size];
        displs = new int[size];
        
        int offset = 0;
        for (int i = 0; i < size; ++i)
        {
            int i_rows_per_proc = n / size;
            int i_extra = n % size;
            int i_start = i * i_rows_per_proc + (i < i_extra ? i : i_extra);
            int i_end = i_start + i_rows_per_proc + (i < i_extra ? 1 : 0);
            int i_local_rows = i_end - i_start;
            
            recvcounts[i] = i_local_rows * l;
            displs[i] = offset;
            offset += i_local_rows * l;
        }
    }
    
    MPI_Gatherv(local_result, local_rows * l, MPI_INT,
                out_mat, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        delete[] recvcounts;
        delete[] displs;
    }

    delete[] local_result;
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    /* TODO */
    delete[] a_mat;
    delete[] b_mat;
}
