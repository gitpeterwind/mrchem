/*
 * MRChem, a numerical real-space code for molecular electronic structure
 * calculations within the self-consistent field (SCF) approximations of quantum
 * chemistry (Hartree-Fock and Density Functional Theory).
 * Copyright (C) 2020 Stig Rune Jensen, Luca Frediani, Peter Wind and contributors.
 *
 * This file is part of MRChem.
 *
 * MRChem is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MRChem is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with MRChem.  If not, see <https://www.gnu.org/licenses/>.
 *
 * For information on the complete list of contributors to MRChem, see:
 * <https://mrchem.readthedocs.io/>
 */

#include <MRCPP/Printer>
#include <MRCPP/Timer>

#include "parallel.h"
#include "qmfunctions/ComplexFunction.h"
#include "qmfunctions/Density.h"
#include "qmfunctions/Orbital.h"

#ifdef MRCHEM_HAS_OMP
#ifndef MRCPP_HAS_OMP
#include <omp.h>
#endif
#define mrchem_get_max_threads() omp_get_max_threads()
#define mrchem_get_num_threads() omp_get_num_threads()
#define mrchem_get_thread_num() omp_get_thread_num()
#define mrchem_set_dynamic(n) omp_set_dynamic(n)
#else
#define mrchem_get_max_threads() 1
#define mrchem_get_num_threads() 1
#define mrchem_get_thread_num() 0
#define mrchem_set_dynamic(n)
#endif

using mrcpp::Printer;

namespace mrchem {

namespace omp {

int n_threads = mrchem_get_max_threads();

} // namespace omp

using namespace Eigen;

namespace mpi {

bool numerically_exact = false;
int shared_memory_size = 1000;

int world_size = 1;
int world_rank = 0;
int orb_size = 1;
int orb_rank = 0;
int share_size = 1;
int share_rank = 0;
int sh_group_rank = 0;
int is_bank = 0;
int is_bankclient = 1;
int is_bankmaster = 0; // only one bankmaster is_bankmaster
int bank_size = -1;
std::vector<int> bankmaster;

MPI_Comm comm_orb;
MPI_Comm comm_share;
MPI_Comm comm_sh_group;
MPI_Comm comm_bank;

Bank orb_bank;

} // namespace mpi

int id_shift; // to ensure that nodes, orbitals and functions do not collide

int metadata_block[3]; // can add more metadata in future
int const size_metadata = 3;

void mpi::initialize() {
    Eigen::setNbThreads(1);
    mrchem_set_dynamic(0);
    mrcpp::set_max_threads(omp::n_threads);

#ifdef MRCHEM_HAS_MPI
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi::world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi::world_rank);

    // divide the world into groups
    // each group has its own group communicator definition

    // define independent group of MPI processes, that are not part of comm_orb
    // for now the new group does not include comm_share
    mpi::comm_bank = MPI_COMM_WORLD; // clients and master
    MPI_Comm comm_remainder;         // clients only

    // set bank_size automatically if not defined by user
    if (mpi::world_size < 2) {
        mpi::bank_size = 0;
    } else if (mpi::bank_size < 0) {
        mpi::bank_size = std::max(mpi::world_size / 4, 1);
    }
    if (mpi::world_size - mpi::bank_size < 1) MSG_ABORT("No MPI ranks left for working!");
    if (mpi::bank_size < 1 and mpi::world_size > 1) MSG_ABORT("Bank size must be at least one when using MPI!");

    mpi::bankmaster.resize(mpi::bank_size);
    for (int i = 0; i < mpi::bank_size; i++) {
        mpi::bankmaster[i] = mpi::world_size - i - 1; // rank of the bankmasters
    }
    if (mpi::world_rank < mpi::world_size - mpi::bank_size) {
        // everything which is left
        mpi::is_bank = 0;
        mpi::is_bankclient = 1;
    } else {
        // special group of bankmasters
        mpi::is_bank = 1;
        mpi::is_bankclient = 0;
        if (mpi::world_rank == mpi::world_size - mpi::bank_size) mpi::is_bankmaster = 1;
    }
    MPI_Comm_split(MPI_COMM_WORLD, mpi::is_bankclient, mpi::world_rank, &comm_remainder);

    // split world into groups that can share memory
    MPI_Comm_split_type(comm_remainder, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &mpi::comm_share);

    MPI_Comm_rank(mpi::comm_share, &mpi::share_rank);
    MPI_Comm_size(mpi::comm_share, &mpi::share_size);

    // define a rank of the group
    MPI_Comm_split(comm_remainder, mpi::share_rank, mpi::world_rank, &mpi::comm_sh_group);
    // mpiShRank is color (same color->in same group)
    // MPI_worldrank is key (orders rank within the groups)

    // we define a new orbital rank, so that the orbitals within
    // a shared memory group, have consecutive ranks
    MPI_Comm_rank(mpi::comm_sh_group, &mpi::sh_group_rank);

    mpi::orb_rank = mpi::share_rank + mpi::sh_group_rank * mpi::world_size;
    MPI_Comm_split(comm_remainder, 0, mpi::orb_rank, &mpi::comm_orb);
    // 0 is color (same color->in same group)
    // mpiOrbRank is key (orders rank in the group)

    MPI_Comm_rank(mpi::comm_orb, &mpi::orb_rank);
    MPI_Comm_size(mpi::comm_orb, &mpi::orb_size);

    // determine the maximum value alowed for mpi tags
    void *val;
    int flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &val, &flag); // max value allowed by MPI for tags
    id_shift = *(int *)val / 2;                                 // half is reserved for non orbital.

    if (mpi::is_bank) {
        // bank is open until end of program
        mpi::orb_bank.open();
        mpi::finalize();
        exit(EXIT_SUCCESS);
    }
#else
    mpi::bank_size = 0;
#endif
}

void mpi::finalize() {
#ifdef MRCHEM_HAS_MPI
    if (mpi::bank_size > 0 and mpi::grand_master()) {
        println(3, " max data in bank " << mpi::orb_bank.get_maxtotalsize() << " MB ");
        mpi::orb_bank.close();
    }
    MPI_Barrier(MPI_COMM_WORLD); // to ensure everybody got here
    MPI_Finalize();
#endif
}

void mpi::barrier(MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
    MPI_Barrier(comm);
#endif
}

/*********************************
 * Orbital related MPI functions *
 *********************************/

bool mpi::grand_master() {
    return (mpi::world_rank == 0 and is_bankclient) ? true : false;
}

bool mpi::share_master() {
    return (mpi::share_rank == 0) ? true : false;
}

/** @brief Test if orbital belongs to this MPI rank (or is common)*/
bool mpi::my_orb(const Orbital &orb) {
    return (orb.rankID() < 0 or orb.rankID() == mpi::orb_rank) ? true : false;
}

/** @brief Test if orbital belongs to this MPI rank */
bool mpi::my_unique_orb(const Orbital &orb) {
    return (orb.rankID() == mpi::orb_rank) ? true : false;
}

/** @brief Distribute orbitals in vector round robin. Orbitals should be empty.*/
void mpi::distribute(OrbitalVector &Phi) {
    for (int i = 0; i < Phi.size(); i++) Phi[i].setRankID(i % mpi::orb_size);
}

/** @brief Free all function pointers not belonging to this MPI rank */
void mpi::free_foreign(OrbitalVector &Phi) {
    for (auto &i : Phi) {
        if (not mpi::my_orb(i)) i.free(NUMBER::Total);
    }
}

/** @brief Return the subset of an OrbitalVector that belongs to this MPI rank */
OrbitalChunk mpi::get_my_chunk(OrbitalVector &Phi) {
    OrbitalChunk chunk;
    for (int i = 0; i < Phi.size(); i++) {
        if (mpi::my_orb(Phi[i])) chunk.push_back(std::make_tuple(i, Phi[i]));
    }
    return chunk;
}

/** @brief Add up each entry of the vector with contributions from all MPI ranks */
void mpi::allreduce_vector(IntVector &vec, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
    int N = vec.size();
    MPI_Allreduce(MPI_IN_PLACE, vec.data(), N, MPI_INT, MPI_SUM, comm);
#endif
}

/** @brief Add up each entry of the vector with contributions from all MPI ranks */
void mpi::allreduce_vector(DoubleVector &vec, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
    int N = vec.size();
    MPI_Allreduce(MPI_IN_PLACE, vec.data(), N, MPI_DOUBLE, MPI_SUM, comm);
#endif
}

/** @brief Add up each entry of the vector with contributions from all MPI ranks */
void mpi::allreduce_vector(ComplexVector &vec, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
    int N = vec.size();
    MPI_Allreduce(MPI_IN_PLACE, vec.data(), N, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm);
#endif
}

/** @brief Add up each entry of the matrix with contributions from all MPI ranks */
void mpi::allreduce_matrix(IntMatrix &mat, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
    int N = mat.size();
    MPI_Allreduce(MPI_IN_PLACE, mat.data(), N, MPI_INT, MPI_SUM, comm);
#endif
}

/** @brief Add up each entry of the matrix with contributions from all MPI ranks */
void mpi::allreduce_matrix(DoubleMatrix &mat, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
    int N = mat.size();
    MPI_Allreduce(MPI_IN_PLACE, mat.data(), N, MPI_DOUBLE, MPI_SUM, comm);
#endif
}

/** @brief Add up each entry of the matrix with contributions from all MPI ranks */
void mpi::allreduce_matrix(ComplexMatrix &mat, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
    int N = mat.size();
    MPI_Allreduce(MPI_IN_PLACE, mat.data(), N, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm);
#endif
}

// send an orbital with MPI, includes orbital meta data
void mpi::send_orbital(Orbital &orb, int dst, int tag, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
    mpi::send_function(orb, dst, tag, comm);
    OrbitalData &orbinfo = orb.getOrbitalData();
    MPI_Send(&orbinfo, sizeof(OrbitalData), MPI_BYTE, dst, 0, comm);
#endif
}

// receive an orbital with MPI, includes orbital meta data
void mpi::recv_orbital(Orbital &orb, int src, int tag, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
    mpi::recv_function(orb, src, tag, comm);

    MPI_Status status;
    OrbitalData &orbinfo = orb.getOrbitalData();
    MPI_Recv(&orbinfo, sizeof(OrbitalData), MPI_BYTE, src, 0, comm, &status);
#endif
}

// send a function with MPI
void mpi::send_function(QMFunction &func, int dst, int tag, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
#ifdef MRCPP_HAS_MPI
    if (func.isShared()) MSG_WARN("Sending a shared function is not recommended");
    FunctionData &funcinfo = func.getFunctionData();
    MPI_Send(&funcinfo, sizeof(FunctionData), MPI_BYTE, dst, 0, comm);
    if (func.hasReal()) mrcpp::send_tree(func.real(), dst, tag, comm, funcinfo.real_size);
    if (func.hasImag()) mrcpp::send_tree(func.imag(), dst, tag + 10000, comm, funcinfo.imag_size);
#else
    MSG_ABORT("MRCPP compiled without MPI support");
#endif
#endif
}

// receive a function with MPI
void mpi::recv_function(QMFunction &func, int src, int tag, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
#ifdef MRCPP_HAS_MPI
    if (func.isShared()) MSG_WARN("Receiving a shared function is not recommended");
    MPI_Status status;

    FunctionData &funcinfo = func.getFunctionData();
    MPI_Recv(&funcinfo, sizeof(FunctionData), MPI_BYTE, src, 0, comm, &status);
    if (funcinfo.real_size > 0) {
        // We must have a tree defined for receiving nodes. Define one:
        if (not func.hasReal()) func.alloc(NUMBER::Real);
        mrcpp::recv_tree(func.real(), src, tag, comm, funcinfo.real_size);
    }

    if (funcinfo.imag_size > 0) {
        // We must have a tree defined for receiving nodes. Define one:
        if (not func.hasImag()) func.alloc(NUMBER::Imag);
        mrcpp::recv_tree(func.imag(), src, tag + 10000, comm, funcinfo.imag_size);
    }
#else
    MSG_ABORT("MRCPP compiled without MPI support");
#endif
#endif
}

/** Update a shared function after it has been changed by one of the MPI ranks. */
void mpi::share_function(QMFunction &func, int src, int tag, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
#ifdef MRCPP_HAS_MPI
    if (func.isShared()) {
        if (func.hasReal()) mrcpp::share_tree(func.real(), src, tag, comm);
        if (func.hasImag()) mrcpp::share_tree(func.imag(), src, 2 * tag, comm);
    }
#else
    MSG_ABORT("MRCPP compiled without MPI support");
#endif
#endif
}

/** @brief Add all mpi function into rank zero */
void mpi::reduce_function(double prec, QMFunction &func, MPI_Comm comm) {
/* 1) Each odd rank send to the left rank
   2) All odd ranks are "deleted" (can exit routine)
   3) new "effective" ranks are defined within the non-deleted ranks
      effective rank = rank/fac , where fac are powers of 2
   4) repeat
 */
#ifdef MRCHEM_HAS_MPI
    int comm_size, comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    if (comm_size == 1) return;

    int fac = 1; // powers of 2
    while (fac < comm_size) {
        if ((comm_rank / fac) % 2 == 0) {
            // receive
            int src = comm_rank + fac;
            if (src < comm_size) {
                QMFunction func_i(false);
                int tag = 3333 + src;
                mpi::recv_function(func_i, src, tag, comm);
                func.add(1.0, func_i); // add in place using union grid
                func.crop(prec);
            }
        }
        if ((comm_rank / fac) % 2 == 1) {
            // send
            int dest = comm_rank - fac;
            if (dest >= 0) {
                int tag = 3333 + comm_rank;
                mpi::send_function(func, dest, tag, comm);
                break; // once data is sent we are done
            }
        }
        fac *= 2;
    }
    MPI_Barrier(comm);
#endif
}

/** @brief make union tree and send into rank zero */
void mpi::reduce_Tree_noCoeff(mrcpp::FunctionTree<3> &tree, MPI_Comm comm) {
/* 1) Each odd rank send to the left rank
   2) All odd ranks are "deleted" (can exit routine)
   3) new "effective" ranks are defined within the non-deleted ranks
      effective rank = rank/fac , where fac are powers of 2
   4) repeat
 */
#ifdef MRCHEM_HAS_MPI
    int comm_size, comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    if (comm_size == 1) return;

    int fac = 1; // powers of 2
    while (fac < comm_size) {
        if ((comm_rank / fac) % 2 == 0) {
            // receive
            int src = comm_rank + fac;
            if (src < comm_size) {
                int tag = 3333 + src;
                mrcpp::FunctionTree<3> tree_i(*MRA);
                mrcpp::recv_tree(tree_i, src, tag, comm, -1, false);
                tree.appendTreeNoCoeff(tree_i); // make union grid
            }
        }
        if ((comm_rank / fac) % 2 == 1) {
            // send
            int dest = comm_rank - fac;
            if (dest >= 0) {
                int tag = 3333 + comm_rank;
                mrcpp::send_tree(tree, dest, tag, comm, -1, false);
                break; // once data is sent we are done
            }
        }
        fac *= 2;
    }
    MPI_Barrier(comm);
#endif
}

/** @brief make union tree without coeff and send to all
 *  Include both real and imaginary parts
 */
void mpi::allreduce_Tree_noCoeff(mrcpp::FunctionTree<3> &tree, OrbitalVector &Phi, MPI_Comm comm) {
    /* 1) make union grid of own orbitals
       2) make union grid with others orbitals (sent to rank zero)
       3) rank zero broadcast func to everybody
     */
    int N = Phi.size();
    for (int j = 0; j < N; j++) {
        if (not mpi::my_orb(Phi[j])) continue;
        if (Phi[j].hasReal()) tree.appendTreeNoCoeff(Phi[j].real());
        if (Phi[j].hasImag()) tree.appendTreeNoCoeff(Phi[j].imag());
    }
#ifdef MRCHEM_HAS_MPI
    mpi::reduce_Tree_noCoeff(tree, mpi::comm_orb);
    mpi::broadcast_Tree_noCoeff(tree, mpi::comm_orb);
#endif
}

/** @brief Distribute rank zero function to all ranks */
void mpi::broadcast_function(QMFunction &func, MPI_Comm comm) {
/* use same strategy as a reduce, but in reverse order */
#ifdef MRCHEM_HAS_MPI
    int comm_size, comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    if (comm_size == 1) return;

    int fac = 1; // powers of 2
    while (fac < comm_size) fac *= 2;
    fac /= 2;

    while (fac > 0) {
        if (comm_rank % fac == 0 and (comm_rank / fac) % 2 == 1) {
            // receive
            int src = comm_rank - fac;
            int tag = 4334 + comm_rank;
            mpi::recv_function(func, src, tag, comm);
        }
        if (comm_rank % fac == 0 and (comm_rank / fac) % 2 == 0) {
            // send
            int dst = comm_rank + fac;
            int tag = 4334 + dst;
            if (dst < comm_size) mpi::send_function(func, dst, tag, comm);
        }
        fac /= 2;
    }
    MPI_Barrier(comm);
#endif
}

/** @brief Distribute rank zero function to all ranks */
void mpi::broadcast_Tree_noCoeff(mrcpp::FunctionTree<3> &tree, MPI_Comm comm) {
/* use same strategy as a reduce, but in reverse order */
#ifdef MRCHEM_HAS_MPI
    int comm_size, comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    if (comm_size == 1) return;

    int fac = 1; // powers of 2
    while (fac < comm_size) fac *= 2;
    fac /= 2;

    while (fac > 0) {
        if (comm_rank % fac == 0 and (comm_rank / fac) % 2 == 1) {
            // receive
            int src = comm_rank - fac;
            int tag = 4334 + comm_rank;
            mrcpp::recv_tree(tree, src, tag, comm, -1, false);
        }
        if (comm_rank % fac == 0 and (comm_rank / fac) % 2 == 0) {
            // send
            int dst = comm_rank + fac;
            int tag = 4334 + dst;
            if (dst < comm_size) mrcpp::send_tree(tree, dst, tag, comm, -1, false);
        }
        fac /= 2;
    }
    MPI_Barrier(comm);
#endif
}

/**************************
 * Bank related functions *
 **************************/

Bank::~Bank() {
    for (int ix = 1; ix < this->deposits.size(); ix++) this->clear(ix);
}

void Bank::open() {
#ifdef MRCHEM_HAS_MPI
    MPI_Status status;
    char safe_data1;
    int deposit_size = sizeof(bank::deposit);
    int n_chunks, ix;
    int message;
    int datasize = -1;
    struct Blockdata_struct {
        std::vector<double *> data; // to store the incoming data
        std::vector<bool> deleted;  // to indicate if it has been deleted already
        MatrixXd BlockData;         // to put the final block
        // eigen matrix are per default stored column-major (one can add columns at the end)
        std::vector<int> N_rows;
        std::map<int, int> id2data; // internal index of the data in the block
        std::vector<int> id;        // the id of each column. Either nodeid, or orbid
    };
    std::map<int, Blockdata_struct *> nodeid2block; // to get block from its nodeid (all coeff for one node)
    std::map<int, Blockdata_struct *> orbid2block;  // to get block from its orbid

    deposits.resize(1); // we reserve 0, since it is the value returned for undefined key
    queue.resize(1);    // we reserve 0, since it is the value returned for undefined key

    bool printinfo = false;

    // The bank never goes out of this loop until it receives a close message!
    while (true) {
        MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, mpi::comm_bank, &status);
        if (printinfo)
            std::cout << mpi::world_rank << " got message " << message << " from " << status.MPI_SOURCE << std::endl;
        if (message == CLOSE_BANK) {
            if (mpi::is_bankmaster and printinfo) std::cout << "Bank is closing" << std::endl;
            this->clear_bank();
            break; // close bank, i.e stop listening for incoming messages
        }
        if (message == CLEAR_BANK) {
            this->clear_bank();
            for (auto const &block : nodeid2block) {
                if (block.second == nullptr) continue;
                for (int i = 0; i < block.second->data.size(); i++) {
                    if (not block.second->deleted[i]) {
                        this->currentsize -= block.second->N_rows[i] / 128; // converted into kB
                        delete[] block.second->data[i];
                    }
                }
                delete block.second;
            }
            nodeid2block.clear();
            orbid2block.clear();
            // send message that it is ready (value of message is not used)
            MPI_Ssend(&message, 1, MPI_INT, status.MPI_SOURCE, 77, mpi::comm_bank);
        }
        if (message == CLEAR_BLOCKS) {
            // clear only blocks whith id less than status.MPI_TAG.
            std::vector<int> toeraseVec; // it is dangerous to erase an iterator within its own loop
            for (auto const &block : nodeid2block) {
                if (block.second == nullptr) toeraseVec.push_back(block.first);
                if (block.second == nullptr) continue;
                if (block.first >= status.MPI_TAG and status.MPI_TAG != 0) continue;
                for (int i = 0; i < block.second->data.size(); i++) {
                    if (not block.second->deleted[i]) {
                        this->currentsize -= block.second->N_rows[i] / 128; // converted into kB
                        delete[] block.second->data[i];
                    }
                }
                this->currentsize -= block.second->BlockData.size() / 128; // converted into kB
                block.second->BlockData.resize(0, 0); // NB: the matrix does not clear itself otherwise
                assert(this->currentsize >= 0);
                this->currentsize = std::max(0ll, this->currentsize);
                toeraseVec.push_back(block.first);
            }
            for (int ierase : toeraseVec) { nodeid2block.erase(ierase); }
            toeraseVec.clear();
            std::vector<int> datatoeraseVec;
            for (auto const &block : orbid2block) {
                if (block.second == nullptr) toeraseVec.push_back(block.first);
                if (block.second == nullptr) continue;
                datatoeraseVec.clear();
                for (int i = 0; i < block.second->data.size(); i++) {
                    if (block.second->id[i] < status.MPI_TAG or status.MPI_TAG == 0) datatoeraseVec.push_back(i);
                    if (block.second->id[i] < status.MPI_TAG or status.MPI_TAG == 0) block.second->data[i] = nullptr;
                }
                std::sort(datatoeraseVec.begin(), datatoeraseVec.end());
                std::reverse(datatoeraseVec.begin(), datatoeraseVec.end());
                for (int ierase : datatoeraseVec) {
                    block.second->id.erase(block.second->id.begin() + ierase);
                    block.second->data.erase(block.second->data.begin() + ierase);
                    block.second->N_rows.erase(block.second->N_rows.begin() + ierase);
                }
                if (block.second->data.size() == 0) toeraseVec.push_back(block.first);
            }
            for (int ierase : toeraseVec) { orbid2block.erase(ierase); }

            if (status.MPI_TAG == 0) orbid2block.clear();
            // could have own clear for data?
            for (int ix = 1; ix < deposits.size(); ix++) {
                if (deposits[ix].id >= id_shift) {
                    if (deposits[ix].hasdata) delete deposits[ix].data;
                    if (deposits[ix].hasdata) id2ix[deposits[ix].id] = 0; // indicate that it does not exist
                    deposits[ix].hasdata = false;
                }
            }
            // send message that it is ready (value of message is not used)
            MPI_Ssend(&message, 1, MPI_INT, status.MPI_SOURCE, 78, mpi::comm_bank);
        }
        if (message == GETMAXTOTDATA) {
            int maxsize_int = maxsize / 1024; // convert into MB
            MPI_Send(&maxsize_int, 1, MPI_INT, status.MPI_SOURCE, 1171, mpi::comm_bank);
        }
        if (message == GETTOTDATA) {
            int maxsize_int = currentsize / 1024; // convert into MB
            MPI_Send(&maxsize_int, 1, MPI_INT, status.MPI_SOURCE, 1172, mpi::comm_bank);
        }

        if (message == GET_NODEDATA or message == GET_NODEBLOCK) {
            // NB: has no queue system yet
            int nodeid = status.MPI_TAG; // which block to fetch from
            if (nodeid2block.count(nodeid) and nodeid2block[nodeid] != nullptr) {
                Blockdata_struct *block = nodeid2block[nodeid];
                int dataindex = 0; // internal index of the data in the block
                int size = 0;
                if (message == GET_NODEDATA) {
                    // get id of data within block
                    MPI_Recv(
                        metadata_block, size_metadata, MPI_INT, status.MPI_SOURCE, nodeid + 1, mpi::comm_bank, &status);
                    int orbid = metadata_block[1];     // which part of the block to fetch
                    dataindex = block->id2data[orbid]; // column of the data in the block
                    size = block->N_rows[dataindex];   // number of doubles to fetch
                    if (metadata_block[2] == 0) {
                        metadata_block[2] = size;
                        MPI_Send(metadata_block, size_metadata, MPI_INT, status.MPI_SOURCE, nodeid, mpi::comm_bank);
                    }
                } else {
                    // send entire block. First make one contiguous superblock
                    // Prepare the data as one contiguous block
                    if (block->data.size() == 0)
                        std::cout << "Zero size blockdata! " << nodeid << " " << block->N_rows.size() << std::endl;
                    block->BlockData.resize(block->N_rows[0], block->data.size());
                    size = block->N_rows[0] * block->data.size();
                    if (printinfo)
                        std::cout << " rewrite into superblock " << block->data.size() << " " << block->N_rows[0]
                                  << " tag " << status.MPI_TAG << std::endl;
                    for (int j = 0; j < block->data.size(); j++) {
                        for (int i = 0; i < block->N_rows[j]; i++) { block->BlockData(i, j) = block->data[j][i]; }
                    }
                    // repoint to the data in BlockData
                    for (int j = 0; j < block->data.size(); j++) {
                        if (block->deleted[j] == true) std::cout << "ERROR data already deleted " << std::endl;
                        assert(block->deleted[j] == false);
                        delete[] block->data[j];
                        block->deleted[j] = true;
                        block->data[j] = block->BlockData.col(j).data();
                    }
                    dataindex = 0; // start from first column
                    // send info about the size of the superblock
                    metadata_block[0] = status.MPI_TAG;     // nodeid
                    metadata_block[1] = block->data.size(); // number of columns
                    metadata_block[2] = size;               // total size = rows*columns
                    MPI_Send(metadata_block, size_metadata, MPI_INT, status.MPI_SOURCE, nodeid, mpi::comm_bank);
                    // send info about the id of each column
                    MPI_Send(
                        block->id.data(), metadata_block[1], MPI_INT, status.MPI_SOURCE, nodeid + 1, mpi::comm_bank);
                }
                double *data_p = block->data[dataindex];
                if (size > 0) MPI_Send(data_p, size, MPI_DOUBLE, status.MPI_SOURCE, nodeid + 2, mpi::comm_bank);
            } else {
                // Block with this id does not exist.
                if (message == GET_NODEDATA) {
                    MPI_Recv(
                        metadata_block, size_metadata, MPI_INT, status.MPI_SOURCE, nodeid + 1, mpi::comm_bank, &status);
                    int size = metadata_block[2]; // number of doubles to send
                    if (size == 0) {
                        metadata_block[2] = size;
                        MPI_Send(metadata_block, size_metadata, MPI_INT, status.MPI_SOURCE, nodeid, mpi::comm_bank);
                    } else {
                        std::vector<double> zero(size, 0.0); // send zeroes
                        MPI_Ssend(zero.data(), size, MPI_DOUBLE, status.MPI_SOURCE, nodeid + 2, mpi::comm_bank);
                    }
                } else {
                    metadata_block[0] = status.MPI_TAG; // nodeid
                    metadata_block[1] = 0;              // number of columns
                    metadata_block[2] = 0;              // total size = rows*columns
                    MPI_Send(
                        metadata_block, size_metadata, MPI_INT, status.MPI_SOURCE, metadata_block[0], mpi::comm_bank);
                }
            }
        }
        if (message == GET_ORBBLOCK) {
            // NB: BLOCKDATA has no queue system yet
            int orbid = status.MPI_TAG; // which block to fetch from

            if (orbid2block.count(orbid) and orbid2block[orbid] != nullptr) {
                Blockdata_struct *block = orbid2block[orbid];
                int dataindex = 0; // internal index of the data in the block
                int size = 0;
                // send entire block. First make one contiguous superblock
                // Prepare the data as one contiguous block
                if (block->data.size() == 0)
                    std::cout << "Zero size blockdata! C " << orbid << " " << block->N_rows.size() << std::endl;
                size = 0;
                for (int j = 0; j < block->data.size(); j++) size += block->N_rows[j];

                std::vector<double> coeff(size);
                int ij = 0;
                for (int j = 0; j < block->data.size(); j++) {
                    for (int i = 0; i < block->N_rows[j]; i++) { coeff[ij++] = block->data[j][i]; }
                }
                // send info about the size of the superblock
                metadata_block[0] = orbid;
                metadata_block[1] = block->data.size(); // number of columns
                metadata_block[2] = size;               // total size = rows*columns
                MPI_Send(metadata_block, size_metadata, MPI_INT, status.MPI_SOURCE, orbid, mpi::comm_bank);
                MPI_Send(block->id.data(), metadata_block[1], MPI_INT, status.MPI_SOURCE, orbid + 1, mpi::comm_bank);
                MPI_Send(coeff.data(), size, MPI_DOUBLE, status.MPI_SOURCE, orbid + 2, mpi::comm_bank);
            } else {
                // it is possible and allowed that the block has not been written
                if (printinfo)
                    std::cout << " block does not exist " << orbid << " " << orbid2block.count(orbid) << std::endl;
                // Block with this id does not exist.
                metadata_block[0] = orbid;
                metadata_block[1] = 0; // number of columns
                metadata_block[2] = 0; // total size = rows*columns
                MPI_Send(metadata_block, size_metadata, MPI_INT, status.MPI_SOURCE, orbid, mpi::comm_bank);
            }
        }

        if (message == GET_ORBITAL or message == GET_ORBITAL_AND_WAIT or message == GET_ORBITAL_AND_DELETE or
            message == GET_FUNCTION or message == GET_DATA) {
            // withdrawal
            int ix = id2ix[status.MPI_TAG];
            if (ix == 0) {
                if (printinfo) std::cout << mpi::world_rank << " not found " << status.MPI_TAG << std::endl;
                if (message == GET_ORBITAL or message == GET_ORBITAL_AND_DELETE) {
                    // do not wait for the orbital to arrive
                    int found = 0;
                    if (printinfo)
                        std::cout << mpi::world_rank << " sending found 0 to " << status.MPI_SOURCE << std::endl;
                    MPI_Send(&found, 1, MPI_INT, status.MPI_SOURCE, 117, mpi::comm_bank);
                } else {
                    // the id does not exist. Put in queue and Wait until it is defined
                    if (printinfo) std::cout << mpi::world_rank << " queuing " << status.MPI_TAG << std::endl;
                    if (id2qu[status.MPI_TAG] == 0) {
                        queue.push_back({status.MPI_TAG, {status.MPI_SOURCE}});
                        id2qu[status.MPI_TAG] = queue.size() - 1;
                    } else {
                        // somebody is already waiting for this id. queue in queue
                        queue[id2qu[status.MPI_TAG]].clients.push_back(status.MPI_SOURCE);
                    }
                }
            } else {
                if (deposits[ix].id != status.MPI_TAG) std::cout << ix << " Bank accounting error " << std::endl;
                if (message == GET_ORBITAL or message == GET_ORBITAL_AND_WAIT or message == GET_ORBITAL_AND_DELETE) {
                    if (message == GET_ORBITAL or message == GET_ORBITAL_AND_DELETE) {
                        int found = 1;
                        MPI_Send(&found, 1, MPI_INT, status.MPI_SOURCE, 117, mpi::comm_bank);
                    }
                    mpi::send_orbital(*deposits[ix].orb, status.MPI_SOURCE, deposits[ix].id, mpi::comm_bank);
                    if (message == GET_ORBITAL_AND_DELETE) {
                        this->currentsize -= deposits[ix].orb->getSizeNodes(NUMBER::Total);
                        deposits[ix].orb->free(NUMBER::Total);
                        id2ix[status.MPI_TAG] = 0;
                    }
                }
                if (message == GET_FUNCTION) {
                    mpi::send_function(*deposits[ix].orb, status.MPI_SOURCE, deposits[ix].id, mpi::comm_bank);
                }
                if (message == GET_DATA) {
                    MPI_Send(deposits[ix].data,
                             deposits[ix].datasize,
                             MPI_DOUBLE,
                             status.MPI_SOURCE,
                             deposits[ix].id,
                             mpi::comm_bank);
                }
            }
        }
        if (message == SAVE_NODEDATA) {
            // get the extra metadata
            MPI_Recv(
                metadata_block, size_metadata, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, mpi::comm_bank, &status);
            int nodeid = metadata_block[0]; // which block to write (should = status.MPI_TAG)
            int orbid = metadata_block[1];  // which part of the block
            int size = metadata_block[2];   // number of doubles

            // test if the block exists already
            if (printinfo)
                std::cout << mpi::world_rank << " save data nodeid " << nodeid << " size " << size << std::endl;
            if (nodeid2block.count(nodeid) == 0 or nodeid2block[nodeid] == nullptr) {
                if (printinfo) std::cout << mpi::world_rank << " block does not exist yet  " << std::endl;
                // the block does not exist yet, create it
                Blockdata_struct *block = new Blockdata_struct;
                nodeid2block[nodeid] = block;
            }
            if (orbid2block.count(orbid) == 0 or orbid2block[orbid] == nullptr) {
                // the block does not exist yet, create it
                Blockdata_struct *orbblock = new Blockdata_struct;
                orbid2block[orbid] = orbblock;
            }
            // append the incoming data
            Blockdata_struct *block = nodeid2block[nodeid];
            block->id2data[orbid] = nodeid2block[nodeid]->data.size(); // internal index of the data in the block
            double *data_p = new double[size];
            this->currentsize += size / 128; // converted into kB
            this->maxsize = std::max(this->currentsize, this->maxsize);
            block->data.push_back(data_p);
            block->deleted.push_back(false);
            block->id.push_back(orbid);
            block->N_rows.push_back(size);

            Blockdata_struct *orbblock = orbid2block[orbid];
            orbblock->id2data[nodeid] = orbblock->data.size(); // internal index of the data in the block
            orbblock->data.push_back(data_p);
            orbblock->deleted.push_back(false);
            orbblock->id.push_back(nodeid);
            orbblock->N_rows.push_back(size);

            MPI_Recv(data_p, size, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, mpi::comm_bank, &status);
            if (printinfo)
                std::cout << " written block " << nodeid << " id " << orbid << " subblocks "
                          << nodeid2block[nodeid]->data.size() << std::endl;
        }
        if (message == SAVE_ORBITAL or message == SAVE_FUNCTION or message == SAVE_DATA) {
            // make a new deposit
            int exist_flag = 0;
            if (id2ix[status.MPI_TAG]) {
                std::cout << "WARNING: id " << status.MPI_TAG << " exists already"
                          << " " << status.MPI_SOURCE << " " << message << " " << std::endl;
                ix = id2ix[status.MPI_TAG]; // the deposit exist from before. Will be overwritten
                exist_flag = 1;
                if (message == SAVE_DATA and !deposits[ix].hasdata) {
                    exist_flag = 0;
                    deposits[ix].data = new double[datasize];
                    deposits[ix].hasdata = true;
                }
            } else {
                ix = deposits.size(); // NB: ix is now index of last element + 1
                deposits.resize(ix + 1);
                if (message == SAVE_ORBITAL or message == SAVE_FUNCTION) deposits[ix].orb = new Orbital(0);
                if (message == SAVE_DATA) {
                    deposits[ix].data = new double[datasize];
                    deposits[ix].hasdata = true;
                }
            }
            deposits[ix].id = status.MPI_TAG;
            id2ix[deposits[ix].id] = ix;
            deposits[ix].source = status.MPI_SOURCE;
            if (message == SAVE_ORBITAL) {
                mpi::recv_orbital(*deposits[ix].orb, deposits[ix].source, deposits[ix].id, mpi::comm_bank);
                if (exist_flag == 0) {
                    this->currentsize += deposits[ix].orb->getSizeNodes(NUMBER::Total);
                    this->maxsize = std::max(this->currentsize, this->maxsize);
                }
            }
            if (message == SAVE_FUNCTION) {
                mpi::recv_function(*deposits[ix].orb, deposits[ix].source, deposits[ix].id, mpi::comm_bank);
            }
            if (message == SAVE_DATA) {
                deposits[ix].datasize = datasize;
                MPI_Recv(deposits[ix].data,
                         datasize,
                         MPI_DOUBLE,
                         deposits[ix].source,
                         deposits[ix].id,
                         mpi::comm_bank,
                         &status);
                this->currentsize += datasize / 128; // converted into kB
                this->maxsize = std::max(this->currentsize, this->maxsize);
            }
            if (id2qu[deposits[ix].id] != 0) {
                // someone is waiting for those data. Send to them
                int iq = id2qu[deposits[ix].id];
                if (deposits[ix].id != queue[iq].id) std::cout << ix << " Bank queue accounting error " << std::endl;
                for (int iqq : queue[iq].clients) {
                    if (message == SAVE_ORBITAL) {
                        mpi::send_orbital(*deposits[ix].orb, iqq, queue[iq].id, mpi::comm_bank);
                    }
                    if (message == SAVE_FUNCTION) {
                        mpi::send_function(*deposits[ix].orb, iqq, queue[iq].id, mpi::comm_bank);
                    }
                    if (message == SAVE_DATA) {
                        MPI_Send(deposits[ix].data, datasize, MPI_DOUBLE, iqq, queue[iq].id, mpi::comm_bank);
                    }
                }
                queue[iq].clients.clear(); // cannot erase entire queue[iq], because that would require to shift all the
                                           // id2qu value larger than iq
                queue[iq].id = -1;
                id2qu.erase(deposits[ix].id);
            }
        }
        if (message == SET_DATASIZE) {
            int datasize_new;
            MPI_Recv(&datasize_new, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, mpi::comm_bank, &status);
            if (datasize_new != datasize) {
                // make sure that all old data arrays are deleted
                for (int ix = 1; ix < deposits.size(); ix++) {
                    if (deposits[ix].hasdata) {
                        delete deposits[ix].data;
                        deposits[ix].hasdata = false;
                    }
                }
            }
            datasize = datasize_new;
        }
    }
#endif
}

// save orbital in Bank with identity id
int Bank::put_orb(int id, Orbital &orb) {
#ifdef MRCHEM_HAS_MPI
    // for now we distribute according to id
    if (id > id_shift) MSG_ABORT("Bank id must be less than max allowed tag / 2 ");
    MPI_Send(&SAVE_ORBITAL, 1, MPI_INT, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
    mpi::send_orbital(orb, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
#endif
    return 1;
}

// get orbital with identity id.
// If wait=0, return immediately with value zero if not available (default)
// else, wait until available
int Bank::get_orb(int id, Orbital &orb, int wait) {
#ifdef MRCHEM_HAS_MPI
    MPI_Status status;
    if (wait == 0) {
        MPI_Send(&GET_ORBITAL, 1, MPI_INT, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
        int found;
        MPI_Recv(&found, 1, MPI_INT, mpi::bankmaster[id % mpi::bank_size], 117, mpi::comm_bank, &status);
        if (found != 0) {
            mpi::recv_orbital(orb, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
            return 1;
        } else {
            return 0;
        }
    } else {
        MPI_Send(&GET_ORBITAL_AND_WAIT, 1, MPI_INT, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
        mpi::recv_orbital(orb, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
    }
#endif
    return 1;
}

// get orbital with identity id, and delete from bank.
// return immediately with value zero if not available
int Bank::get_orb_del(int id, Orbital &orb) {
#ifdef MRCHEM_HAS_MPI
    MPI_Status status;
    MPI_Send(&GET_ORBITAL_AND_DELETE, 1, MPI_INT, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
    int found;
    MPI_Recv(&found, 1, MPI_INT, mpi::bankmaster[id % mpi::bank_size], 117, mpi::comm_bank, &status);
    if (found != 0) {
        mpi::recv_orbital(orb, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
        return 1;
    } else {
        return 0;
    }
#endif
    return 1;
}

// save function in Bank with identity id
int Bank::put_func(int id, QMFunction &func) {
#ifdef MRCHEM_HAS_MPI
    // for now we distribute according to id
    id += id_shift;
    MPI_Send(&SAVE_FUNCTION, 1, MPI_INT, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
    mpi::send_function(func, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
#endif
    return 1;
}

// get function with identity id
int Bank::get_func(int id, QMFunction &func) {
#ifdef MRCHEM_HAS_MPI
    MPI_Status status;
    id += id_shift;
    MPI_Send(&GET_FUNCTION, 1, MPI_INT, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
    mpi::recv_function(func, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
#endif
    return 1;
}

// set the size of the data arrays (in size of doubles) to be sent/received later
int Bank::set_datasize(int datasize) {
#ifdef MRCHEM_HAS_MPI
    for (int i = 0; i < mpi::bank_size; i++) {
        MPI_Send(&SET_DATASIZE, 1, MPI_INT, mpi::bankmaster[i], 0, mpi::comm_bank);
        MPI_Send(&datasize, 1, MPI_INT, mpi::bankmaster[i], 0, mpi::comm_bank);
    }
#endif
    return 1;
}

// save data in Bank with identity id . datasize MUST have been set already. NB:not tested
int Bank::put_data(int id, int size, double *data) {
#ifdef MRCHEM_HAS_MPI
    // for now we distribute according to id
    id += id_shift;
    MPI_Send(&SAVE_DATA, 1, MPI_INT, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
    MPI_Send(data, size, MPI_DOUBLE, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
#endif
    return 1;
}

// get data with identity id
int Bank::get_data(int id, int size, double *data) {
#ifdef MRCHEM_HAS_MPI
    MPI_Status status;
    id += id_shift;
    MPI_Send(&GET_DATA, 1, MPI_INT, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
    MPI_Recv(data, size, MPI_DOUBLE, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank, &status);
#endif
    return 1;
}

// save data in Bank with identity id as part of block with identity nodeid.
int Bank::put_nodedata(int id, int nodeid, int size, double *data) {
#ifdef MRCHEM_HAS_MPI
    // for now we distribute according to nodeid
    metadata_block[0] = nodeid; // which block
    metadata_block[1] = id;     // id within block
    metadata_block[2] = size;   // size of this data
    MPI_Send(&SAVE_NODEDATA, 1, MPI_INT, mpi::bankmaster[nodeid % mpi::bank_size], nodeid, mpi::comm_bank);
    MPI_Send(metadata_block, size_metadata, MPI_INT, mpi::bankmaster[nodeid % mpi::bank_size], nodeid, mpi::comm_bank);
    MPI_Send(data, size, MPI_DOUBLE, mpi::bankmaster[nodeid % mpi::bank_size], nodeid, mpi::comm_bank);
#endif
    return 1;
}

// get data with identity id
int Bank::get_nodedata(int id, int nodeid, int size, double *data, std::vector<int> &idVec) {
#ifdef MRCHEM_HAS_MPI
    MPI_Status status;
    // get the column with identity id
    metadata_block[0] = nodeid; // which block
    metadata_block[1] = id;     // id within block.
    metadata_block[2] = size;   // expected size of data
    MPI_Send(&GET_NODEDATA, 1, MPI_INT, mpi::bankmaster[nodeid % mpi::bank_size], nodeid, mpi::comm_bank);
    MPI_Send(
        metadata_block, size_metadata, MPI_INT, mpi::bankmaster[nodeid % mpi::bank_size], nodeid + 1, mpi::comm_bank);

    MPI_Recv(data, size, MPI_DOUBLE, mpi::bankmaster[nodeid % mpi::bank_size], nodeid + 2, mpi::comm_bank, &status);
#endif
    return 1;
}

// get all data for nodeid
int Bank::get_nodeblock(int nodeid, double *data, std::vector<int> &idVec) {
#ifdef MRCHEM_HAS_MPI
    MPI_Status status;
    // get the entire superblock and also the id of each column
    MPI_Send(&GET_NODEBLOCK, 1, MPI_INT, mpi::bankmaster[nodeid % mpi::bank_size], nodeid, mpi::comm_bank);
    MPI_Recv(metadata_block,
             size_metadata,
             MPI_INT,
             mpi::bankmaster[nodeid % mpi::bank_size],
             nodeid,
             mpi::comm_bank,
             &status);
    idVec.resize(metadata_block[1]);
    int size = metadata_block[2];
    if (size > 0)
        MPI_Recv(idVec.data(),
                 metadata_block[1],
                 MPI_INT,
                 mpi::bankmaster[nodeid % mpi::bank_size],
                 nodeid + 1,
                 mpi::comm_bank,
                 &status);
    if (size > 0)
        MPI_Recv(data, size, MPI_DOUBLE, mpi::bankmaster[nodeid % mpi::bank_size], nodeid + 2, mpi::comm_bank, &status);
#endif
    return 1;
}

// get all data with identity orbid
int Bank::get_orbblock(int orbid, double *&data, std::vector<int> &nodeidVec, int bankstart) {
#ifdef MRCHEM_HAS_MPI
    MPI_Status status;
    int nodeid = mpi::orb_rank + bankstart;
    // get the entire superblock and also the nodeid of each column
    MPI_Send(&GET_ORBBLOCK, 1, MPI_INT, mpi::bankmaster[nodeid % mpi::bank_size], orbid, mpi::comm_bank);
    MPI_Recv(metadata_block,
             size_metadata,
             MPI_INT,
             mpi::bankmaster[nodeid % mpi::bank_size],
             orbid,
             mpi::comm_bank,
             &status);
    nodeidVec.resize(metadata_block[1]);
    int totsize = metadata_block[2];
    if (totsize > 0)
        MPI_Recv(nodeidVec.data(),
                 metadata_block[1],
                 MPI_INT,
                 mpi::bankmaster[nodeid % mpi::bank_size],
                 orbid + 1,
                 mpi::comm_bank,
                 &status);
    data = new double[totsize];
    if (totsize > 0)
        MPI_Recv(
            data, totsize, MPI_DOUBLE, mpi::bankmaster[nodeid % mpi::bank_size], orbid + 2, mpi::comm_bank, &status);
#endif
    return 1;
}

// Ask to close the Bank
void Bank::close() {
#ifdef MRCHEM_HAS_MPI
    for (int i = 0; i < mpi::bank_size; i++) {
        MPI_Send(&CLOSE_BANK, 1, MPI_INT, mpi::bankmaster[i], 0, mpi::comm_bank);
    }
#endif
}

int Bank::get_maxtotalsize() {
    int maxtot = 0;
#ifdef MRCHEM_HAS_MPI
    MPI_Status status;
    int datasize;
    for (int i = 0; i < mpi::bank_size; i++) {
        MPI_Send(&GETMAXTOTDATA, 1, MPI_INT, mpi::bankmaster[i], 0, mpi::comm_bank);
        MPI_Recv(&datasize, 1, MPI_INT, mpi::bankmaster[i], 1171, mpi::comm_bank, &status);
        maxtot = std::max(maxtot, datasize);
    }
#endif
    return maxtot;
}

std::vector<int> Bank::get_totalsize() {
    std::vector<int> tot;
#ifdef HAVE_MPI
    MPI_Status status;
    int datasize;
    for (int i = 0; i < mpi::bank_size; i++) {
        MPI_Send(&GETTOTDATA, 1, MPI_INT, mpi::bankmaster[i], 0, mpi::comm_bank);
        MPI_Recv(&datasize, 1, MPI_INT, mpi::bankmaster[i], 1172, mpi::comm_bank, &status);
        tot.push_back(datasize);
    }
#endif
    return tot;
}

// remove all deposits
// NB:: collective call. All clients must call this
void Bank::clear_all(int iclient, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
    // 1) wait until all clients are ready
    mpi::barrier(comm);
    // master send signal to bank
    if (iclient == 0) {
        for (int i = 0; i < mpi::bank_size; i++) {
            // should be made explicitely non-blocking
            MPI_Send(&CLEAR_BANK, 1, MPI_INT, mpi::bankmaster[i], 0, mpi::comm_bank);
        }
        for (int i = 0; i < mpi::bank_size; i++) {
            // wait until Bank is finished and has sent signal
            MPI_Status status;
            int message;
            MPI_Recv(&message, 1, MPI_INT, mpi::bankmaster[i], 77, mpi::comm_bank, &status);
        }
    }
    mpi::barrier(comm);
#endif
}

// remove all blockdata with nodeid < nodeidmax
// NB:: collective call. All clients must call this
void Bank::clear_blockdata(int iclient, int nodeidmax, MPI_Comm comm) {
#ifdef MRCHEM_HAS_MPI
    // 1) wait until all clients are ready
    mpi::barrier(comm);
    // master send signal to bank
    if (iclient == 0) {
        for (int i = 0; i < mpi::bank_size; i++) {
            MPI_Send(&CLEAR_BLOCKS, 1, MPI_INT, mpi::bankmaster[i], nodeidmax, mpi::comm_bank);
        }
        for (int i = 0; i < mpi::bank_size; i++) {
            // wait until Bank is finished and has sent signal
            MPI_Status status;
            int message;
            MPI_Recv(&message, 1, MPI_INT, mpi::bankmaster[i], 78, mpi::comm_bank, &status);
        }
    }
    mpi::barrier(comm);
#endif
}

void Bank::clear_bank() {
#ifdef MRCHEM_HAS_MPI
    for (int ix = 1; ix < this->deposits.size(); ix++) this->clear(ix);
    this->deposits.resize(1);
    this->queue.resize(1);
    this->id2ix.clear();
    this->id2qu.clear();
    this->currentsize = 0;
#endif
}

void Bank::clear(int ix) {
#ifdef MRCHEM_HAS_MPI
    if (deposits[ix].orb != nullptr) deposits[ix].orb->free(NUMBER::Total);
    if (deposits[ix].hasdata) delete deposits[ix].data;
    deposits[ix].hasdata = false;
#endif
}

} // namespace mrchem
