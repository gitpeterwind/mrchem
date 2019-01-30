#pragma once

#include "MRCPP/Parallel"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "mrchem.h"
#include "qmfunctions/qmfunction_fwd.h"

namespace mrchem {

namespace omp {
extern int n_threads;
} //namespace omp

namespace mpi {
extern bool numerically_exact;
extern bool share_nuc_pot;
extern bool share_coul_dens;
extern bool share_coul_pot;
extern bool local_coul_pot;
extern bool share_xc_dens;
extern bool share_xc_pot;
extern int shared_memory_size;

extern int orb_rank;
extern int orb_size;
extern int share_rank;
extern int share_size;
extern int sh_group_rank;

extern MPI_Comm comm_orb;
extern MPI_Comm comm_share;
extern MPI_Comm comm_sh_group;

void initialize(int argc, char **argv);
void finalize();
void barrier(MPI_Comm comm);

bool grand_master();
bool share_master();
bool my_orb(const Orbital &orb);
bool my_unique_orb(const Orbital &orb);
void distribute(OrbitalVector &Phi);
void free_foreign(OrbitalVector &Phi);
OrbitalChunk get_my_chunk(OrbitalVector &Phi);

void send_orbital(Orbital &orb, int dst, int tag);
void recv_orbital(Orbital &orb, int src, int tag);

void send_function(QMFunction &func, int dst, int tag, MPI_Comm comm);
void recv_function(QMFunction &func, int src, int tag, MPI_Comm comm);
void share_function(QMFunction &func, int src, int tag, MPI_Comm comm);

void reduce_function(double prec, QMFunction &func, MPI_Comm comm);
void broadcast_function(QMFunction &func, MPI_Comm comm);

void allreduce_vector(IntVector &vec, MPI_Comm comm);
void allreduce_vector(DoubleVector &vec, MPI_Comm comm);
void allreduce_vector(ComplexVector &vec, MPI_Comm comm);
void allreduce_matrix(IntMatrix &vec, MPI_Comm comm);
void allreduce_matrix(DoubleMatrix &mat, MPI_Comm comm);
void allreduce_matrix(ComplexMatrix &mat, MPI_Comm comm);

} //namespace mpi

} //namespace mrchem
