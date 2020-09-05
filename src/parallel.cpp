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

#include <bits/stdc++.h>
#include <MRCPP/Printer>
#include <MRCPP/Timer>

#include "parallel.h"
#include "qmfunctions/ComplexFunction.h"
#include "qmfunctions/Density.h"
#include "qmfunctions/Orbital.h"

using mrcpp::Printer;
using mrcpp::Timer;

namespace mrchem {

namespace omp {

int n_threads = omp_get_max_threads();

} // namespace omp

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
int bank_rank = 0;
std::vector<int> bankmaster;

MPI_Comm comm_orb;
MPI_Comm comm_share;
MPI_Comm comm_sh_group;
MPI_Comm comm_bank; // communicator that allow using the bank, i.e. all = world

Bank orb_bank;
int orb_bank_size;
int task_bank;

} // namespace mpi

int id_shift; // to ensure that nodes, orbitals and functions do not collide

void mpi::initialize() {
    omp_set_dynamic(0);

#ifdef HAVE_MPI
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
    if (mpi::world_size > 1 and mpi::bank_size < 0) mpi::bank_size = mpi::world_size / 6 + 1;
    mpi::bank_size = std::max(0, mpi::bank_size);

    if (mpi::world_size - mpi::bank_size < 1) MSG_ABORT("No MPI ranks left for working!");
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

    //if bank_size is large enough, we reserve one as "task manager"
    mpi::orb_bank_size = mpi::bank_size;
    mpi::task_bank=-1;
    if(mpi::bank_size == 1){
        //use the bank as task manager
        mpi::task_bank=mpi::bankmaster[0];
    } else if(mpi::bank_size > 1){
        // reserve one bank for task management only
        mpi::orb_bank_size = mpi::bank_size-1;
        mpi::task_bank=mpi::bankmaster[mpi::orb_bank_size];
    }

    void *val;
    int flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &val, &flag); // max value allowed by MPI for tags
    id_shift = *(int*)val / 2;
    if(mpi::world_rank==0) std::cout<<" max tag value "<<*(int*)val<<std::endl;
    if (mpi::is_bank) {
        // define rank among bankmasters
        mpi::bank_rank = mpi::world_rank % mpi::bank_size;

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
#ifdef HAVE_MPI
    if (mpi::bank_size > 0 and mpi::grand_master()){
        auto plevel = Printer::getPrintLevel();
        if (plevel > 1) std::cout<<" max data in bank "<<mpi::orb_bank.get_maxtotalsize()<<" MB "<<std::endl;
        mpi::orb_bank.close();
    }
    MPI_Barrier(MPI_COMM_WORLD); // to ensure everybody got here
    MPI_Finalize();
#endif
}

void mpi::barrier(MPI_Comm comm) {
#ifdef HAVE_MPI
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
#ifdef HAVE_MPI
    int N = vec.size();
    MPI_Allreduce(MPI_IN_PLACE, vec.data(), N, MPI_INT, MPI_SUM, comm);
#endif
}

/** @brief Add up each entry of the vector with contributions from all MPI ranks */
void mpi::allreduce_vector(DoubleVector &vec, MPI_Comm comm) {
#ifdef HAVE_MPI
    int N = vec.size();
    MPI_Allreduce(MPI_IN_PLACE, vec.data(), N, MPI_DOUBLE, MPI_SUM, comm);
#endif
}

/** @brief Add up each entry of the vector with contributions from all MPI ranks */
void mpi::allreduce_vector(ComplexVector &vec, MPI_Comm comm) {
#ifdef HAVE_MPI
    int N = vec.size();
    MPI_Allreduce(MPI_IN_PLACE, vec.data(), N, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm);
#endif
}

/** @brief Add up each entry of the matrix with contributions from all MPI ranks */
void mpi::allreduce_matrix(IntMatrix &mat, MPI_Comm comm) {
#ifdef HAVE_MPI
    int N = mat.size();
    MPI_Allreduce(MPI_IN_PLACE, mat.data(), N, MPI_INT, MPI_SUM, comm);
#endif
}

/** @brief Add up each entry of the matrix with contributions from all MPI ranks */
void mpi::allreduce_matrix(DoubleMatrix &mat, MPI_Comm comm) {
#ifdef HAVE_MPI
    int N = mat.size();
    MPI_Allreduce(MPI_IN_PLACE, mat.data(), N, MPI_DOUBLE, MPI_SUM, comm);
#endif
}

/** @brief Add up each entry of the matrix with contributions from all MPI ranks */
void mpi::allreduce_matrix(ComplexMatrix &mat, MPI_Comm comm) {
#ifdef HAVE_MPI
    int N = mat.size();
    MPI_Allreduce(MPI_IN_PLACE, mat.data(), N, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm);
#endif
}

// send an orbital with MPI, includes orbital meta data
void mpi::send_orbital(Orbital &orb, int dst, int tag, MPI_Comm comm) {
#ifdef HAVE_MPI
    mpi::send_function(orb, dst, tag, comm);
    OrbitalData &orbinfo = orb.getOrbitalData();
    MPI_Send(&orbinfo, sizeof(OrbitalData), MPI_BYTE, dst, 0, comm);
#endif
}

// receive an orbital with MPI, includes orbital meta data
void mpi::recv_orbital(Orbital &orb, int src, int tag, MPI_Comm comm) {
#ifdef HAVE_MPI
    mpi::recv_function(orb, src, tag, comm);

    MPI_Status status;
    OrbitalData &orbinfo = orb.getOrbitalData();
    MPI_Recv(&orbinfo, sizeof(OrbitalData), MPI_BYTE, src, 0, comm, &status);
#endif
}

// send a function with MPI
void mpi::send_function(QMFunction &func, int dst, int tag, MPI_Comm comm) {
#ifdef HAVE_MPI
    if (func.isShared()) MSG_WARN("Sending a shared function is not recommended");
    FunctionData &funcinfo = func.getFunctionData();
    MPI_Send(&funcinfo, sizeof(FunctionData), MPI_BYTE, dst, 0, comm);
    if (func.hasReal()) mrcpp::send_tree(func.real(), dst, tag, comm, funcinfo.real_size);
    if (func.hasImag()) mrcpp::send_tree(func.imag(), dst, tag + 10000, comm, funcinfo.imag_size);
#endif
}

// receive a function with MPI
void mpi::recv_function(QMFunction &func, int src, int tag, MPI_Comm comm) {
#ifdef HAVE_MPI
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
#endif
}

/** Update a shared function after it has been changed by one of the MPI ranks. */
void mpi::share_function(QMFunction &func, int src, int tag, MPI_Comm comm) {
#ifdef HAVE_MPI
    if (func.isShared()) {
        if (func.hasReal()) mrcpp::share_tree(func.real(), src, tag, comm);
        if (func.hasImag()) mrcpp::share_tree(func.imag(), src, 2 * tag, comm);
    }
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
#ifdef HAVE_MPI
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

/** @brief Distribute rank zero function to all ranks */
void mpi::broadcast_function(QMFunction &func, MPI_Comm comm) {
/* use same strategy as a reduce, but in reverse order */
#ifdef HAVE_MPI
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

/**************************
 * Bank related functions *
 **************************/

Bank::~Bank() {
    for (int ix = 1; ix < this->deposits.size(); ix++) this->clear(ix);
}

// structure that allow to sort ix according to value
struct s_ix {
    long long ix, value;
};
bool operator<(s_ix  a, s_ix  b) {
    return a.value < b.value;
}
std::set<s_ix> priority;
std::map<int, s_ix> rank2q;


// Sort banks in a linked list.
// Can take the first, the last, or extract a specific bank and put it in first or last position
// Very fast, O(1)
// In a linked list the memory address of the banks is fixed(and stored in br2pos), but they
// relative position is defined by the "linked list"
struct SimpleQueue {
    int nbanks;
    std::list<int> order; // linked list
    std::vector<std::list<int>::iterator> br2pos;
    std::vector<long long> timestamp;
    long long first_timestamp, last_timestamp;
    SimpleQueue(int nbanks) : nbanks(nbanks) {
        assert(nbanks > 0);
        for (int i = 0; i < nbanks; i++) {
            order.push_back(i);
            timestamp.push_back(i);
            br2pos.push_back( prev(order.end()) );
        }
        first_timestamp = 0;
        last_timestamp = nbanks-1;
    }
    void moveToBack(int bank_rank) { // back is newest timestamp. Show banks which recently got a task
        auto pos = br2pos[bank_rank];
        order.splice(order.end(), order, pos); // move element from position pos to end
        last_timestamp++;
        timestamp[bank_rank] = last_timestamp;
    }
    void moveToFront(int bank_rank) { // front is oldest timestamp. Show banks which did not do anything for a while
        auto pos = br2pos[bank_rank];
        order.splice(order.begin(), order, pos); // move element from position pos to front
        first_timestamp--;
        timestamp[bank_rank] = first_timestamp;
    }
    int getFront() {
        return order.front();
    }
    int getBack() {
        return order.back();
    }
    long long getTimestamp(int bank_rank) {
        return timestamp[bank_rank];
    }
};




void Bank::open() {
#ifdef HAVE_MPI
    MPI_Status status;
    char safe_data1;
    int deposit_size = sizeof(bank::deposit);
    int n_chunks, ix;
    int message;
    int datasize = -1;

    deposits.resize(1); // we reserve 0, since it is the value returned for undefined key
    queue.resize(1);    // we reserve 0, since it is the value returned for undefined key

    int tot_ntasks = 0;
    int tot_ntasksij[2];
    std::vector<int> tot_ntasks_2D;
    int next_task = 0;
    std::map<int, std::vector<int>> readytasks;
    std::map<int, std::vector<int>> banks_fromid;
    int request_counter = 0;
    std::vector<long long> queue_n(mpi::orb_bank_size, 0ll);

    bool printinfo = false;
    SimpleQueue bank_queue(mpi::orb_bank_size);
    // The bank never goes out of this loop until it receives a close message!
    while (true) {
        Timer t_timer;
        MPI_Recv(&message, 1, MPI_INTEGER, MPI_ANY_SOURCE, MPI_ANY_TAG, mpi::comm_bank, &status);
        //        if (printinfo or (int)((float)t_timer.elapsed()* 1000)>150)
        //  std::cout << mpi::world_rank << " got message " << message <<" from " << status.MPI_SOURCE <<" "<<(int)((float)t_timer.elapsed()* 1000)<<std::endl;
        if (message == CLOSE_BANK) {
            if (mpi::is_bankmaster and printinfo) std::cout << "Bank is closing" << std::endl;
            this->clear_bank();
	    readytasks.clear();
            break; // close bank, i.e stop listening for incoming messages
        }
        if (message == CLEAR_BANK) {
            this->clear_bank();
	    readytasks.clear();
            banks_fromid.clear();
            // send message that it is ready (value of message is not used)
            MPI_Ssend(&message, 1, MPI_INTEGER, status.MPI_SOURCE, 77, mpi::comm_bank);
        }
        if (message == GETMAXTOTDATA) {
            int maxsize_int = maxsize/1024; // convert into MB
            MPI_Send(&maxsize_int, 1, MPI_INTEGER, status.MPI_SOURCE, 1171, mpi::comm_bank);
	}
        if (message == GETTOTDATA) {
            int maxsize_int = currentsize/1024; // convert into MB
            MPI_Send(&maxsize_int, 1, MPI_INTEGER, status.MPI_SOURCE, 1172, mpi::comm_bank);
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
                    MPI_Send(&found, 1, MPI_INTEGER, status.MPI_SOURCE, 117, mpi::comm_bank);
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
                        MPI_Send(&found, 1, MPI_INTEGER, status.MPI_SOURCE, 117, mpi::comm_bank);
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
        if (message == SAVE_ORBITAL or message == SAVE_FUNCTION or message == SAVE_DATA) {
            // make a new deposit
            int exist_flag = 0;
            if (id2ix[status.MPI_TAG]) {
	      std::cout<<"WARNING: id "<<status.MPI_TAG<<" exists already"<<" "<<status.MPI_SOURCE<<std::endl;
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
                if (message == SAVE_ORBITAL or message == SAVE_FUNCTION) {
                    deposits[ix].orb = new Orbital(0);
                }
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
                this->currentsize += datasize/128; // converted into kB
                this->maxsize = std::max(this->currentsize, this->maxsize);
            }
            if (id2qu[deposits[ix].id] != 0) {
                // someone is waiting for those data. Send to them
                int iq = id2qu[deposits[ix].id];
                if (deposits[ix].id != queue[iq].id)
                    std::cout << ix << " Bank queue accounting ERROR " << deposits[ix].id<<" "<<queue[iq].id<<std::endl;
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
                queue[iq].clients.clear(); // cannot erase entire queue[iq], because that would require to shift all the id2qu value larger than iq
                queue[iq].id = -1;
                id2qu.erase(deposits[ix].id);
            }
        }
        if (message == SET_DATASIZE) {
            int datasize_new;
            MPI_Recv(&datasize_new, 1, MPI_INTEGER, status.MPI_SOURCE, status.MPI_TAG, mpi::comm_bank, &status);
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

        if (message == INIT_TASKS) {
            MPI_Recv(&tot_ntasksij, 2, MPI_INTEGER, status.MPI_SOURCE, status.MPI_TAG, mpi::comm_bank, &status);
	    tot_ntasks = tot_ntasksij[0]*tot_ntasksij[1];
	    tot_ntasks_2D.resize(tot_ntasksij[1]);
	    for (int i = 0; i < tot_ntasksij[1]; i++) tot_ntasks_2D[i] = tot_ntasksij[0]; // each j has  tot_ntasksij[0] i blocks
            next_task = 0;
        }
        if (message == GET_NEXTTASK) {
            int task = next_task;
            if(next_task >= tot_ntasks) task = -1; // flag to show all tasks are assigned
            MPI_Send(&task, 1, MPI_INTEGER, status.MPI_SOURCE, 843, mpi::comm_bank);
            next_task++;
        }
        if (message == GET_NEXTTASK_2D) {
            int task = next_task;
            int task_2D[2];
            if(next_task >= tot_ntasks) {
                task = -1; // flag to show all tasks are assigned
                task_2D[0] = -1;
                task_2D[1] = -1;
  	    } else {
	      //if possible, give a task with same j (=status.MPI_TAG)
	      if(tot_ntasks_2D[status.MPI_TAG]>0){
                  tot_ntasks_2D[status.MPI_TAG]--; // next i for that j
                  task = status.MPI_TAG * tot_ntasksij[0] + tot_ntasks_2D[status.MPI_TAG];
                  task_2D[0] = tot_ntasks_2D[status.MPI_TAG];
                  task_2D[1] = status.MPI_TAG; // same j as asked
	      } else {
                  //find any available task
                  int err=1;
                  for (int j = 0; j<tot_ntasks_2D.size(); j++) {
                      int jj = (j+status.MPI_TAG)%tot_ntasks_2D.size();// we want to spread among i
                      if(tot_ntasks_2D[jj]>0){
                          tot_ntasks_2D[jj]--;
                          task_2D[0] = tot_ntasks_2D[jj];
                          task_2D[1] = jj;
                          err = 0;
                          break;
                      }
                  }
                  if (err) std::cout<<"ERROR find 2Dtask"<<std::endl;
	      }
	    }
            MPI_Send(task_2D, 2, MPI_INTEGER, status.MPI_SOURCE, 853, mpi::comm_bank);
            next_task++;
        }
        if (message == PUT_READYTASK) {
	    int iready;
            MPI_Recv(&iready, 1, MPI_INTEGER, status.MPI_SOURCE, status.MPI_TAG, mpi::comm_bank, &status);
	    readytasks[iready].push_back(status.MPI_TAG); // status.MPI_TAG gives the id
            if(readytasks[iready][0]==0)std::cout<<"ERROR putreadytask"<<std::endl;
	}
        if (message == DEL_READYTASK) {
	    int iready; // corresponding orbital index given by client
            MPI_Recv(&iready, 1, MPI_INTEGER, status.MPI_SOURCE, status.MPI_TAG, mpi::comm_bank, &status);
            for (int i=0; i< readytasks[iready].size(); i++) {
                if(readytasks[iready][i] == status.MPI_TAG){
                    readytasks[iready].erase(readytasks[iready].begin() + i); // status.MPI_TAG gives the id
                    break;
                }
            }
	}
        if (message == GET_READYTASK) {
	    int iready = status.MPI_TAG; // the value of the index is sent through the tag
	    int nready = readytasks[iready].size();
            if(nready>0 and readytasks[iready][0]==0)std::cout<<"ERROR getreadytask"<<std::endl;
            MPI_Send(&nready, 1, MPI_INTEGER, status.MPI_SOURCE, 844, mpi::comm_bank);
            MPI_Send(readytasks[iready].data(), nready, MPI_INTEGER, status.MPI_SOURCE, 845, mpi::comm_bank);
	}
        if (message == GET_READYTASK_DEL) {
	    int iready = status.MPI_TAG; // the value of the index is sent through the tag
	    int nready = readytasks[iready].size();
            MPI_Send(&nready, 1, MPI_INTEGER, status.MPI_SOURCE, 844, mpi::comm_bank);
            if(nready>0 and readytasks[iready][0]==0)std::cout<<"ERROR del getreadytask"<<std::endl;
            MPI_Send(readytasks[iready].data(), nready, MPI_INTEGER, status.MPI_SOURCE, 845, mpi::comm_bank);
            readytasks[iready].clear();
	}
        if (message == PUT_ORB_N) {
            int nbanks;
            int id=status.MPI_TAG;
            MPI_Recv(&nbanks, 1, MPI_INTEGER, status.MPI_SOURCE, id, mpi::comm_bank, &status);
            // make a list of n banks that will store the orbital id.
            std::vector<int> banks(nbanks); // the ranks of the banks to whom to send the orbital
            if(banks_fromid[id].size()>0){
	      std::cout<<mpi::orb_rank<<"WARNING: bank id "<<id<<" is already defined "<<banks_fromid[id].size()<<" "<<status.MPI_SOURCE<<std::endl;
            }
            banks_fromid[id].clear();
            for (int i = 0; i < nbanks; i++) {
                //                banks[i] = (id+i) % mpi::orb_bank_size;
                banks[i] = bank_queue.getFront();
                // also banks which are on the same node are marked as "busy"
                //                std::cout<<" put "<<(banks[i])<<std::endl;
                //                for (int ii = 0; ii < 8; ii++) if(banks[i]-banks[i]%8 + ii==mpi::orb_bank_size)std::cout<<" error "<<banks[i]-banks[i]%8 + ii<<" "<<mpi::orb_bank_size<<" tbank"<<mpi::task_bank <<std::endl;
                //      for (int ii = 0; ii < 8; ii++) if(banks[i]-banks[i]%8 + ii!=0) bank_queue.moveToBack(banks[i]-banks[i]%8 + ii-1); // asummes 8 mpi per node, ranks starting from a multiple of 8
               bank_queue.moveToBack(banks[i]);// mark last-> busiest
               //banks[i] = rand()% mpi::orb_bank_size;
               //if(i>0)banks[i] = (banks[0]+ 1 + rand()% (mpi::orb_bank_size-1))% mpi::orb_bank_size;
                banks_fromid[id].push_back(banks[i]); // list of banks that hold the orbital id
          }
            MPI_Send(banks.data(), nbanks, MPI_INTEGER, status.MPI_SOURCE, 846, mpi::comm_bank);
        }
        if (message == GET_ORB_N) {
            // find an available bank
            int id=status.MPI_TAG;
            int b_rank = (id) % mpi::orb_bank_size;
            long long mn=1e18;
            for (int rank : banks_fromid[id]) {
                if(bank_queue.getTimestamp(rank)<mn) {
                    //                if(queue_n[rank]<mn) {
                    //                    mn = queue_n[rank];
                    mn = bank_queue.getTimestamp(rank);
                    b_rank = rank;
                }
            }
            //            std::cout<<" giving bank "<<b_rank<<" score "<<queue_n[b_rank]<<" Timestamp "<<bank_queue.getTimestamp(b_rank) <<" "<<queue_n[b_rank]<<std::endl;
            queue_n[b_rank] = request_counter++;
            // also banks which are on the same node are marked as "busy"
            //            for (int ii = 0; ii < 8; ii++)  if(b_rank-b_rank%8+ii!=0) bank_queue.moveToBack(b_rank-b_rank%8+ii-1); // asummes 8 mpi per node, ranks starting from a multiple of 8
            bank_queue.moveToBack(b_rank); // mark last-> busiest
            MPI_Send(&b_rank, 1, MPI_INTEGER, status.MPI_SOURCE, 847, mpi::comm_bank);
       }
    }
#endif
}

// save orbital in Bank with identity id
int Bank::put_orb(int id, Orbital &orb) {
#ifdef HAVE_MPI
    // for now we distribute according to id
  if (id > id_shift) std::cout<<"WARNING: Bank id should be less than id_shift ("<<id_shift<<"), found id="<<id<<std::endl;
    if (id > id_shift) MSG_WARN("Bank id should be less than id_shift");
    MPI_Send(&SAVE_ORBITAL, 1, MPI_INTEGER, mpi::bankmaster[id % mpi::orb_bank_size], id, mpi::comm_bank);
    mpi::send_orbital(orb, mpi::bankmaster[id % mpi::orb_bank_size], id, mpi::comm_bank);
#endif
    return 1;
}

// get orbital with identity id.
// If wait=0, return immediately with value zero if not available (default)
// else, wait until available
int Bank::get_orb(int id, Orbital &orb, int wait) {
#ifdef HAVE_MPI
    MPI_Status status;
    if (wait == 0) {
        MPI_Send(&GET_ORBITAL, 1, MPI_INTEGER, mpi::bankmaster[id % mpi::orb_bank_size], id, mpi::comm_bank);
        int found;
        MPI_Recv(&found, 1, MPI_INTEGER, mpi::bankmaster[id % mpi::orb_bank_size], 117, mpi::comm_bank, &status);
        if (found != 0) {
            mpi::recv_orbital(orb, mpi::bankmaster[id % mpi::orb_bank_size], id, mpi::comm_bank);
            return 1;
        } else {
            return 0;
        }
    } else {
        MPI_Send(&GET_ORBITAL_AND_WAIT, 1, MPI_INTEGER, mpi::bankmaster[id % mpi::orb_bank_size], id, mpi::comm_bank);
        mpi::recv_orbital(orb, mpi::bankmaster[id % mpi::orb_bank_size], id, mpi::comm_bank);
    }
#endif
    return 1;
}

// get orbital with identity id, and delete from bank.
// return immediately with value zero if not available
int Bank::get_orb_del(int id, Orbital &orb) {
#ifdef HAVE_MPI
    MPI_Status status;
    MPI_Send(&GET_ORBITAL_AND_DELETE, 1, MPI_INTEGER, mpi::bankmaster[id % mpi::orb_bank_size], id, mpi::comm_bank);
    int found;
    MPI_Recv(&found, 1, MPI_INTEGER, mpi::bankmaster[id % mpi::orb_bank_size], 117, mpi::comm_bank, &status);
    if (found != 0) {
        mpi::recv_orbital(orb, mpi::bankmaster[id % mpi::orb_bank_size], id, mpi::comm_bank);
        return 1;
    } else {
        return 0;
    }
#endif
    return 1;
}

// save function in Bank with identity id
int Bank::put_func(int id, QMFunction &func) {
#ifdef HAVE_MPI
    // for now we distribute according to id
    id += id_shift;
    MPI_Send(&SAVE_FUNCTION, 1, MPI_INTEGER, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
    mpi::send_function(func, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
#endif
    return 1;
}

// get function with identity id
int Bank::get_func(int id, QMFunction &func) {
#ifdef HAVE_MPI
    MPI_Status status;
    id += id_shift;
    MPI_Send(&GET_FUNCTION, 1, MPI_INTEGER, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
    mpi::recv_function(func, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
#endif
    return 1;
}

// set the size of the data arrays (in size of doubles) to be sent/received later
int Bank::set_datasize(int datasize) {
#ifdef HAVE_MPI
    for (int i = 0; i < mpi::bank_size; i++) {
        MPI_Send(&SET_DATASIZE, 1, MPI_INTEGER, mpi::bankmaster[i], 0, mpi::comm_bank);
        MPI_Send(&datasize, 1, MPI_INTEGER, mpi::bankmaster[i], 0, mpi::comm_bank);
    }
#endif
    return 1;
}

// save data in Bank with identity id . datasize MUST have been set already. NB:not tested
int Bank::put_data(int id, int size, double *data) {
#ifdef HAVE_MPI
    // for now we distribute according to id
    id += 2 * id_shift;
    MPI_Send(&SAVE_DATA, 1, MPI_INTEGER, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
    MPI_Send(data, size, MPI_DOUBLE, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
#endif
    return 1;
}

// get data with identity id
int Bank::get_data(int id, int size, double *data) {
#ifdef HAVE_MPI
    MPI_Status status;
    id += 2 * id_shift;
    MPI_Send(&GET_DATA, 1, MPI_INTEGER, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank);
    MPI_Recv(data, size, MPI_DOUBLE, mpi::bankmaster[id % mpi::bank_size], id, mpi::comm_bank, &status);
#endif
    return 1;
}

// Ask to close the Bank
void Bank::close() {
#ifdef HAVE_MPI
    for (int i = 0; i < mpi::bank_size; i++) {
        MPI_Send(&CLOSE_BANK, 1, MPI_INTEGER, mpi::bankmaster[i], 0, mpi::comm_bank);
    }
#endif
}

int Bank::get_maxtotalsize() {
    int maxtot = 0;
#ifdef HAVE_MPI
    MPI_Status status;
    int datasize;
    for (int i = 0; i < mpi::bank_size; i++) {
        MPI_Send(&GETMAXTOTDATA, 1, MPI_INTEGER, mpi::bankmaster[i], 0, mpi::comm_bank);
        MPI_Recv(&datasize, 1, MPI_INTEGER, mpi::bankmaster[i], 1171, mpi::comm_bank, &status);
        maxtot = std::max(maxtot, datasize);
    }
#endif
    return maxtot;
}

std::vector<int>  Bank::get_totalsize() {
    std::vector<int> tot;
#ifdef HAVE_MPI
    MPI_Status status;
    int datasize;
    for (int i = 0; i < mpi::bank_size; i++) {
        MPI_Send(&GETTOTDATA, 1, MPI_INTEGER, mpi::bankmaster[i], 0, mpi::comm_bank);
        MPI_Recv(&datasize, 1, MPI_INTEGER, mpi::bankmaster[i], 1172, mpi::comm_bank, &status);
        tot.push_back(datasize);
    }
#endif
    return tot;
}



// remove all deposits
// NB:: collective call. All clients must call this
void Bank::clear_all(int iclient, MPI_Comm comm) {
#ifdef HAVE_MPI
    comm = mpi::comm_orb;
    // 1) wait until all clients are ready
    mpi::barrier(comm);
   // master send signal to bank
    if (iclient == 0) {
        for (int i = 0; i < mpi::bank_size; i++) {
            MPI_Send(&CLEAR_BANK, 1, MPI_INTEGER, mpi::bankmaster[i], 0, mpi::comm_bank);
        }
        for (int i = 0; i < mpi::bank_size; i++) {
            // wait until Bank is finished and has sent signal
            MPI_Status status;
            int message;
            MPI_Recv(&message, 1, MPI_INTEGER, mpi::bankmaster[i], 77, mpi::comm_bank, &status);
        }
    }
    mpi::barrier(comm); //NB: MUST wait until cleared finished before starting other things
#endif
}

void Bank::clear_bank() {
#ifdef HAVE_MPI
    for (int ix = 1; ix < this->deposits.size(); ix++) this->clear(ix);
    this->deposits.resize(1);
    this->queue.resize(1);
    this->id2ix.clear();
    this->id2qu.clear();
    this->currentsize = 0;
#endif
}

void Bank::clear(int ix) {
#ifdef HAVE_MPI
    deposits[ix].orb->free(NUMBER::Total);
    delete deposits[ix].data;
    deposits[ix].hasdata = false;
#endif
}

void Bank::init_tasks(int ntasksi, int ntasksj, int rank, MPI_Comm comm) {
#ifdef HAVE_MPI
    mpi::barrier(comm); // NB: everybody must be done before init
    int dest = mpi::task_bank;
    int ntasks[2];
    ntasks[0]=ntasksi;
    ntasks[1]=ntasksj;
    // NB: Must be Synchronous send (SSend), because others must not
    // be able to start
    if(rank == 0) {
        MPI_Ssend(&INIT_TASKS, 1, MPI_INTEGER, dest, 0, mpi::comm_bank);
        MPI_Ssend(ntasks, 2, MPI_INTEGER, dest, 0, mpi::comm_bank);
    }
    mpi::barrier(comm); // NB: nobody must start before init is ready
 #endif
}

void Bank::init_tasks(int ntaskstot, int rank, MPI_Comm comm) {
#ifdef HAVE_MPI
    mpi::barrier(comm); // NB: everybody must be done before init
    int dest = mpi::task_bank;
    int ntasks[2];
    ntasks[0] = ntaskstot;
    ntasks[1] = 1;
    // NB: Must be Synchronous send (SSend), because others must not
    // be able to start
    if(rank == 0) {
        MPI_Ssend(&INIT_TASKS, 1, MPI_INTEGER, dest, 0, mpi::comm_bank);
        MPI_Ssend(ntasks, 2, MPI_INTEGER, dest, 0, mpi::comm_bank);
    }
    mpi::barrier(comm); // NB: nobody must start before init is ready
 #endif
}

void Bank::get_task(int *task_2D, int i) {
#ifdef HAVE_MPI
    MPI_Status status;
    int dest = mpi::task_bank;
    MPI_Send(&GET_NEXTTASK_2D, 1, MPI_INTEGER, dest, i, mpi::comm_bank);
    MPI_Recv(task_2D, 2, MPI_INTEGER, dest, 853, mpi::comm_bank, &status);
#endif
}


void Bank::get_task(int *task) {
#ifdef HAVE_MPI
    MPI_Status status;
    int dest = mpi::task_bank;
    MPI_Send(&GET_NEXTTASK, 1, MPI_INTEGER, dest, 0, mpi::comm_bank);
    MPI_Recv(task, 1, MPI_INTEGER, dest, 843, mpi::comm_bank, &status);
#endif
}

void Bank::put_readytask(int id, int i) { // register a task as ready, id tells where the result can be found
#ifdef HAVE_MPI
    MPI_Status status;
    int dest = mpi::task_bank;
    MPI_Send(&PUT_READYTASK, 1, MPI_INTEGER, dest, id, mpi::comm_bank);
    MPI_Send(&i, 1, MPI_INTEGER, dest, id, mpi::comm_bank);
#endif
}

void Bank::del_readytask(int id, int i) { // remove the task
#ifdef HAVE_MPI
    MPI_Status status;
    int dest = mpi::task_bank;

    MPI_Send(&DEL_READYTASK, 1, MPI_INTEGER, dest, id, mpi::comm_bank);
    MPI_Send(&i, 1, MPI_INTEGER, dest, id, mpi::comm_bank);
#endif
}

std::vector<int> Bank::get_readytasks(int i, int del) {
    std::vector<int> readytasks;
#ifdef HAVE_MPI
    MPI_Status status;
    int task = GET_READYTASK;
    if(del == 1) task = GET_READYTASK_DEL;

    MPI_Send(&task, 1, MPI_INTEGER, mpi::task_bank, i, mpi::comm_bank);
    int nready;
    MPI_Recv(&nready, 1, MPI_INTEGER, mpi::task_bank, 844, mpi::comm_bank, &status);
    readytasks.resize(nready);
    MPI_Recv(readytasks.data(), nready, MPI_INTEGER, mpi::task_bank, 845, mpi::comm_bank, &status);
    if(nready>0 and readytasks[0]==0)std::cout<<"ERROR get_readytask "<<nready<<std::endl;

#endif
    return readytasks;
}

// save n copies of orbital in Bank with identity id
int Bank::put_orb_n(int id, Orbital &orb, int n) {
#ifdef HAVE_MPI
    MPI_Status status;
    //the task manager assigns n bank to whom the orbital is sent
    int dest = mpi::task_bank;
    MPI_Send(&PUT_ORB_N, 1, MPI_INTEGER, dest, id, mpi::comm_bank);
    MPI_Send(&n, 1, MPI_INTEGER, dest, id, mpi::comm_bank);
    std::vector<int> banks(n); // the rank of the banks to whom to send the orbital
    MPI_Recv(banks.data(), n, MPI_INTEGER, dest, 846, mpi::comm_bank, &status);
    for (int i = 0; i < n; i++) {
        MPI_Send(&SAVE_ORBITAL, 1, MPI_INTEGER, mpi::bankmaster[banks[i]], id, mpi::comm_bank);
        mpi::send_orbital(orb, mpi::bankmaster[banks[i]], id, mpi::comm_bank);
    }
#endif
    return 1;
}

int Bank::get_orb_n(int id, Orbital &orb, int del) {
#ifdef HAVE_MPI
    MPI_Status status;
    //the task manager tells from whom the orbital is to be fetched
    int dest = mpi::task_bank;
    MPI_Send(&GET_ORB_N, 1, MPI_INTEGER, dest, id, mpi::comm_bank);
    int b_rank;
    MPI_Recv(&b_rank, 1, MPI_INTEGER, dest, 847, mpi::comm_bank, &status);
    /*MPI_Send(&GET_ORBITAL, 1, MPI_INTEGER, bank_rank, id, mpi::comm_bank);
    mpi::recv_orbital(orb, bank_rank, id, mpi::comm_bank);*/
     if(del == 1){
        MPI_Status status;
        MPI_Send(&GET_ORBITAL_AND_DELETE, 1, MPI_INTEGER, mpi::bankmaster[b_rank], id, mpi::comm_bank);
        int found;
        MPI_Recv(&found, 1, MPI_INTEGER, mpi::bankmaster[b_rank], 117, mpi::comm_bank, &status);
        if (found != 0) {
            mpi::recv_orbital(orb, mpi::bankmaster[b_rank], id, mpi::comm_bank);
            return 1;
        } else {
            return 0;
        }
     } else {
         Timer tt;
         MPI_Send(&GET_ORBITAL_AND_WAIT, 1, MPI_INTEGER, mpi::bankmaster[b_rank], id, mpi::comm_bank);
         mpi::recv_orbital(orb, mpi::bankmaster[b_rank], id, mpi::comm_bank);
         //         if(mpi::world_rank==0)std::cout << mpi::world_rank << "waited " << (int)((float)tt.elapsed()* 1000)<<" for "<<orb.getSizeNodes(NUMBER::Total)/1000<<" MB from bank:"<<b_rank<<std::endl;
    }
#endif
    return 1;
}

} // namespace mrchem
