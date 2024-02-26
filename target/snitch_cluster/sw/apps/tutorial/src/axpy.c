#include "snrt.h"
#include "printf.h"
#include "data.h"

// Define your kernel
void axpy(uint32_t l, double a, double *x, double *y, double *z) {
    for (uint32_t i = 0; i < l ; i++) {
        z[i] = a * x[i] + y[i];
    }
    snrt_fpu_fence();
}

int main() {
    // Read the mcycle CSR (this is our way to mark/delimit a specific code region for benchmarking)
    uint32_t start_cycle = snrt_mcycle();

    // DM core does not participate in the computation
    if(snrt_is_compute_core())
        axpy(L, a, x, y, z);

    // Read the mcycle CSR
    uint32_t end_cycle = snrt_mcycle();
}

// int main() {
//     uint32_t core_idx = snrt_global_core_idx();
//     uint32_t core_num = snrt_global_core_num();

//     printf("# hart %d global core %d(%d) ", snrt_hartid(),
//            snrt_global_core_idx(), snrt_global_core_num());
//     printf("in cluster %d(%d) ", snrt_cluster_idx(), snrt_cluster_num());
//     printf("cluster core %d(%d) ", snrt_cluster_core_idx(),
//            snrt_cluster_core_num());
//     printf("compute core %d(%d) ", snrt_cluster_core_idx(),
//            snrt_cluster_compute_core_num());
//     printf("compute: %d dm: %d ", snrt_is_compute_core(), snrt_is_dm_core());
//     printf("\n");

//     printf("# cluster mem [%#llx:%#llx]", snrt_l1_start_addr(), snrt_l1_end_addr());
//     printf("\n");

//     return 0;
// }