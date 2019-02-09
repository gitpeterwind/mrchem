/*
 * MRChem, a numerical real-space code for molecular electronic structure
 * calculations within the self-consistent field (SCF) approximations of quantum
 * chemistry (Hartree-Fock and Density Functional Theory).
 * Copyright (C) 2019 Stig Rune Jensen, Luca Frediani, Peter Wind and contributors.
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

#include "MRCPP/Printer"

#include "Orbital.h"
#include "OrbitalIterator.h"
#include "qmfunction_utils.h"
#include "parallel.h"

namespace mrchem {

OrbitalIterator::OrbitalIterator(OrbitalVector &Phi, bool sym)
        : symmetric(sym)
        , iter(0)
        , received_counter(0)
        , sent_counter(0)
        , orbitals(&Phi) {
    //Find and save the orbitals each rank owns
    int my_mpi_rank = mpi::orb_rank;
    for (int impi = 0; impi < mpi::orb_size; impi++) {
        std::vector<int> orb_ix;
        for (int i = 0; i < this->orbitals->size(); i++) {
            Orbital &phi_i = (*this->orbitals)[i];
            int phi_rank = phi_i.rankID();
            if (phi_rank == impi) orb_ix.push_back(i);
            if (impi == my_mpi_rank and phi_rank < 0) orb_ix.push_back(i);
        }
        this->orb_ix_map.push_back(orb_ix);
    }
}

// Receive all the orbitals, for non symmetric treatment
// Receive half of the orbitals, for symmetric treatment
// Send own orbitals to half of the MPI, for symmetric treatment
// The algorithm ensures that at each iteration, each MPI is communicating
// with only one other MPI. This ensures good communication parallelism.
// NB: iter counts the number of MPI looped over:
//     there may be several calls to "next" for the same iter in case max_recv<0,
//     or double iterations for symmetric treatment.
bool OrbitalIterator::next(int max_recv) {
    mpi::free_foreign(*this->orbitals); //delete foreign orbitals
    //NB: for now we completely delete orbitals at each iteration. Could reuse the memory in next iteration.
    this->received_orbital_index.clear();
    this->received_orbitals.clear();
    this->sent_orbital_index.clear();
    this->sent_orbital_mpirank.clear();
    this->rcv_step.clear();

    if (this->iter >= mpi::orb_size) {
        // We have talked to all MPIs -> return
        this->iter = 0;
        return false;
    }

    //  We communicate with one MPI per iteration:
    //   - Symmetric: do two iterations, send in one step receive in the other
    //   - send ALL my orbitals to other_rank (or max_recv if set)
    //   - receive ALL orbitals owned by other_rank (or max_recv if set)
    //   - other_rank increases by one for each iteration, modulo size
    // Each MPI pair will be treated exactly once at the end of the iteration process

    int my_rank = mpi::orb_rank;
    int max_rank = mpi::orb_size;
    int received = 0;                                //number of new received orbitals this call
    int sent = 0;                                    //number of new sent orbitals this call
    int maxget = max_recv;                           //max number of orbitals to receive
    if (maxget < 0) maxget = this->orbitals->size(); //unlimited

    int nstep = 1;                  // send and receive in the same iteration
    if (this->symmetric) nstep = 2; // one iteration is receive, one iteration is send

    for (int step = 0; step < nstep; step++) {
        if (this->iter >= mpi::orb_size) break; //last iteration is completed

        bool IamReceiver = true; //default, both send and receive
        bool IamSender = true;   //default, both send and receive

        // Figure out which MPI to talk to
        int other_rank = (max_rank + this->iter - my_rank) % max_rank;
        //note: my_rank= (max_rank + this->iter - other_rank)%max_rank

        int send_size = this->orb_ix_map[my_rank].size();
        int recv_size = this->orb_ix_map[other_rank].size();

        if (other_rank == my_rank) {
            // We receive from ourself
            IamSender = false; //sending data is "for free"
            for (int i = this->received_counter; i < send_size and received < maxget; i++) {
                int orb_ix = this->orb_ix_map[my_rank][i];
                Orbital &phi_i = (*this->orbitals)[orb_ix];
                this->rcv_step.push_back(step);
                this->received_orbital_index.push_back(orb_ix);
                this->received_orbitals.push_back(phi_i);
                this->received_counter++;
                received++;
            }
        } else {
            if (this->symmetric) {
                // Either send or receive
                // Figure out which one will send and receive
                //(my_rank + other_rank )%2 flips each time other_rank changes by one
                //(my_rank < other_rank) ensures that my_rank and other_rank are opposite
                // my_rank = other_rank must correspond to a receive (since it implicitely means extra work later on)
                // NB: must take care that if at one step other_rank == my_rank, the other step must not be a receive!! not trivial
                if ((my_rank + other_rank + (my_rank > other_rank)) % 2) {
                    IamReceiver = false;
                } else {
                    IamSender = false;
                }
            }

            // highest mpi rank send first receive after, the other does the opposite
            if (my_rank > other_rank) {
                //send
                for (int i = this->sent_counter; i < send_size and IamSender and sent < maxget; i++) {
                    int tag = this->orbitals->size() * this->iter + i;
                    int orb_ix = this->orb_ix_map[my_rank][i];
                    Orbital &phi_i = (*this->orbitals)[orb_ix];
                    mpi::send_orbital(phi_i, other_rank, tag);
                    this->sent_orbital_index.push_back(orb_ix);
                    this->sent_orbital_mpirank.push_back(other_rank);
                    this->sent_counter++; //total to other_rank
                    sent++;               //this call
                }
                //receive
                for (int i = this->received_counter; i < recv_size and IamReceiver and received < maxget; i++) {
                    int tag = this->orbitals->size() * this->iter + i;
                    int orb_ix = this->orb_ix_map[other_rank][i];
                    Orbital &phi_i = (*this->orbitals)[orb_ix];
                    mpi::recv_orbital(phi_i, other_rank, tag);
                    this->received_orbital_index.push_back(orb_ix);
                    this->received_orbitals.push_back(phi_i);
                    this->rcv_step.push_back(step);
                    this->received_counter++; //total from other_rank
                    received++;               //this call
                }
            } else {
                //my_rank < other_rank -> receive first, send after receive
                for (int i = this->received_counter; i < recv_size and IamReceiver and received < maxget; i++) {
                    int tag = this->orbitals->size() * this->iter + i;
                    int orb_ix = this->orb_ix_map[other_rank][i];
                    Orbital &phi_i = (*this->orbitals)[orb_ix];
                    mpi::recv_orbital(phi_i, other_rank, tag);
                    this->received_orbital_index.push_back(orb_ix);
                    this->received_orbitals.push_back(phi_i);
                    this->rcv_step.push_back(step);
                    this->received_counter++; //total from other_rank
                    received++;               //this call
                }
                //send
                for (int i = this->sent_counter; i < send_size and IamSender and sent < maxget; i++) {
                    int tag = this->orbitals->size() * this->iter + i;
                    int orb_ix = this->orb_ix_map[my_rank][i];
                    Orbital &phi_i = (*this->orbitals)[orb_ix];
                    mpi::send_orbital(phi_i, other_rank, tag);
                    this->sent_orbital_index.push_back(orb_ix);
                    this->sent_orbital_mpirank.push_back(other_rank);
                    this->sent_counter++; //total to other_rank
                    sent++;               //this call
                }
            }
        }

        if ((this->sent_counter >= send_size or not IamSender) and
            (this->received_counter >= recv_size or not IamReceiver)) {
            //all orbitals to be processed during this iteration are ready. Start next iteration
            this->sent_counter = 0;
            this->received_counter = 0;
            this->iter++;
            if (this->iter % 2 == 0) break; //must always exit after odd iterations in order to finish duties outside
        }
    }
    return true;
}

// Same as before, but own orbitals are first rotated locally before being sent
// Receive all the orbitals, for non symmetric treatment
// Receive half of the orbitals, for symmetric treatment
// Send own orbitals to half of the MPI, for symmetric treatment
// The algorithm ensures that at each iteration, each MPI is communicating
// with only one other MPI. This ensures good communication parallelism.
// NB: iter counts the number of MPI looped over:
//     there may be several calls to "next" for the same iter in case max_recv<0,
//     or double iterations for symmetric treatment.
    bool OrbitalIterator::next(const ComplexMatrix &U, double prec, int max_recv) {
    mpi::free_foreign(*this->orbitals); //delete foreign orbitals
    //NB: for now we completely delete orbitals at each iteration. Could also reuse the memory in next iteration.
    this->received_orbital_index.clear();
    this->received_orbitals.clear();
    this->sent_orbital_index.clear();
    this->sent_orbital_mpirank.clear();
    this->rcv_step.clear();

    if (this->iter >= mpi::orb_size) {
        // We have talked to all MPIs -> return
        this->iter = 0;
        return false;
    }

    //  We communicate with one MPI per iteration:
    //   - Symmetric: do two iterations, send in one step receive in the other
    //   - send ALL my orbitals to other_rank (or max_recv if set)
    //   - receive ALL orbitals owned by other_rank (or max_recv if set)
    //   - other_rank increases by one for each iteration, modulo size
    // Each MPI pair will be treated exactly once at the end of the iteration process

    double inter_prec = (mpi::numerically_exact) ? -1.0 : prec;
    int my_rank = mpi::orb_rank;
    int max_rank = mpi::orb_size;
    int received = 0;                                //number of new received orbitals this call
    int sent = 0;                                    //number of new sent orbitals this call
    int maxget = max_recv;                           //max number of orbitals to receive
    if (maxget < 0) maxget = this->orbitals->size(); //unlimited

    //    maxget = 1;                     //NB: for now only one orbital at a time

    int nstep = 1;                  // send and receive in the same iteration
    if (this->symmetric) nstep = 2; // one iteration is receive, one iteration is send

    for (int step = 0; step < nstep; step++) {
        if (this->iter >= mpi::orb_size) break; //last iteration is completed

        bool IamReceiver = true; //default, both send and receive
        bool IamSender = true;   //default, both send and receive

        // Figure out which MPI to talk to
        int other_rank = (max_rank + this->iter - my_rank) % max_rank;
        //note: my_rank= (max_rank + this->iter - other_rank)%max_rank

        //        int send_size = this->orb_ix_map[my_rank].size();
        //        int recv_size = this->orb_ix_map[other_rank].size();

        int send_size = this->orb_ix_map[other_rank].size();
        int recv_size = this->orb_ix_map[my_rank].size();

        if (other_rank == my_rank) {
            // We receive from ourself
            IamSender = false; //sending data is "for free"
            for (int i = this->received_counter; i < send_size and received < maxget; i++) {
                int orb_ix = this->orb_ix_map[my_rank][i];
                //rotate own orbitals into orb_ix
                ComplexVector coef_vec(this->orb_ix_map[my_rank].size());
                QMFunctionVector func_vec;
                for (int j = 0; j < this->orb_ix_map[my_rank].size(); j++) {
                    int idx_j = this->orb_ix_map[my_rank][j];
                    Orbital &orb_j = (*this->orbitals)[idx_j];
                    coef_vec[j] = U(orb_ix, idx_j);
                    func_vec.push_back(orb_j);
                }
                Orbital phi_i = (*this->orbitals)[orb_ix].paramCopy();
                qmfunction::linear_combination(phi_i, coef_vec, func_vec, inter_prec);

                this->rcv_step.push_back(step);
                this->received_orbital_index.push_back(orb_ix);
                this->received_orbitals.push_back(phi_i);
                this->received_counter++;
                received++;
            }
        } else {
            if (this->symmetric) {
                // Either send or receive
                // Figure out which one will send and receive
                //(my_rank + other_rank )%2 flips each time other_rank changes by one
                //(my_rank < other_rank) ensures that my_rank and other_rank are opposite
                // my_rank = other_rank must correspond to a receive (since it implicitely means extra work later on)
                // NB: must take care that if at one step other_rank == my_rank, the other step must not be a receive!! not trivial
                if ((my_rank + other_rank + (my_rank > other_rank)) % 2) {
                    IamReceiver = false;
                } else {
                    IamSender = false;
                }
            }

            // highest mpi rank send first receive after, the other does the opposite
            if (my_rank > other_rank) {
                //send
                for (int i = this->sent_counter; i < send_size and IamSender and sent < maxget; i++) {
                    int tag = this->orbitals->size() * this->iter + i;
                    int orb_ix = this->orb_ix_map[other_rank][i];
                    //rotate own orbitals into orb_ix
                    ComplexVector coef_vec(this->orb_ix_map[my_rank].size());
                    QMFunctionVector func_vec;

                    for (int j = 0; j < this->orb_ix_map[my_rank].size(); j++) {
                        int idx_j = this->orb_ix_map[my_rank][j];
                        Orbital &orb_j = (*this->orbitals)[idx_j];
                        coef_vec[j] = U(orb_ix, idx_j);
                        func_vec.push_back(orb_j);
                    }

                    Orbital phi_i = (*this->orbitals)[orb_ix].paramCopy();
                    qmfunction::linear_combination(phi_i, coef_vec, func_vec, inter_prec);

                    mpi::send_orbital(phi_i, other_rank, tag);
                    this->sent_orbital_index.push_back(orb_ix);
                    this->sent_orbital_mpirank.push_back(other_rank);
                    this->sent_counter++; //total to other_rank
                    sent++;               //this call
                }
                //receive
                for (int i = this->received_counter; i < recv_size and IamReceiver and received < maxget; i++) {
                    int tag = this->orbitals->size() * this->iter + i;
                    int orb_ix = this->orb_ix_map[my_rank][i];
                    Orbital phi_i = (*this->orbitals)[orb_ix].paramCopy();
                    mpi::recv_orbital(phi_i, other_rank, tag);
                    this->received_orbital_index.push_back(orb_ix);
                    this->received_orbitals.push_back(phi_i);
                    this->rcv_step.push_back(step);
                    this->received_counter++; //total from other_rank
                    received++;               //this call
                }
            } else {
                //my_rank < other_rank -> receive first, send after receive
                for (int i = this->received_counter; i < recv_size and IamReceiver and received < maxget; i++) {
                    int tag = this->orbitals->size() * this->iter + i;
                    int orb_ix = this->orb_ix_map[my_rank][i];
                    Orbital phi_i = (*this->orbitals)[orb_ix].paramCopy();
                    mpi::recv_orbital(phi_i, other_rank, tag);
                    this->received_orbital_index.push_back(orb_ix);
                    this->received_orbitals.push_back(phi_i);
                    this->rcv_step.push_back(step);
                    this->received_counter++; //total from other_rank
                    received++;               //this call
               }
                //send
                for (int i = this->sent_counter; i < send_size and IamSender and sent < maxget; i++) {
                    int tag = this->orbitals->size() * this->iter + i;
                    int orb_ix = this->orb_ix_map[other_rank][i];
                    //rotate own orbitals into orb_ix
                    ComplexVector coef_vec(this->orb_ix_map[my_rank].size());
                    QMFunctionVector func_vec;

                    for (int j = 0; j < this->orb_ix_map[my_rank].size(); j++) {
                        int idx_j = this->orb_ix_map[my_rank][j];
                        Orbital &orb_j = (*this->orbitals)[idx_j];
                        coef_vec[j] = U(orb_ix, idx_j);
                        func_vec.push_back(orb_j);
                    }

                    Orbital phi_i = (*this->orbitals)[orb_ix].paramCopy();
                    qmfunction::linear_combination(phi_i, coef_vec, func_vec, inter_prec);

                    mpi::send_orbital(phi_i, other_rank, tag);
                    this->sent_orbital_index.push_back(orb_ix);
                    this->sent_orbital_mpirank.push_back(other_rank);
                    this->sent_counter++; //total to other_rank
                    sent++;               //this call
                }
            }
        }

        if ((this->sent_counter >= send_size or not IamSender) and
            (this->received_counter >= recv_size or not IamReceiver)) {
            //all orbitals to be processed during this iteration are ready. Start next iteration
            this->sent_counter = 0;
            this->received_counter = 0;
            this->iter++;
            if (this->iter % 2 == 0) break; //must always exit after odd iterations in order to finish duties outside
        }
    }
    return true;
}

} //namespace mrchem
