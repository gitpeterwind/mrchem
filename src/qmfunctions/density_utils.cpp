/*
 * MRChem, a numerical real-space code for molecular electronic structure
 * calculations within the self-consistent field (SCF) approximations of quantum
 * chemistry (Hartree-Fock and Density Functional Theory).
 * Copyright (C) 2019 Stig Rune Jensen, Jonas Juselius, Luca Frediani, and contributors.
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

#include "MRCPP/Gaussians"
#include "MRCPP/Printer"
#include "MRCPP/Timer"

#include "parallel.h"

#include "Density.h"
#include "Orbital.h"
#include "QMFunction.h"
#include "density_utils.h"
#include "orbital_utils.h"
#include "qmfunction_utils.h"

using mrcpp::FunctionTree;
using mrcpp::FunctionTreeVector;
using mrcpp::Printer;
using mrcpp::Timer;

namespace mrchem {
extern mrcpp::MultiResolutionAnalysis<3> *MRA; // Global MRA

/****************************************
 * Density related standalone functions *
 ****************************************/

namespace density {
Density compute(double prec, Orbital phi, int spin);
void compute_X(double prec, Density &rho, OrbitalVector &Phi, OrbitalVector &X, int spin);
void compute_XY(double prec, Density &rho, OrbitalVector &Phi, OrbitalVector &X, OrbitalVector &Y, int spin);
double compute_occupation(Orbital &phi, int dens_spin);
} // namespace density

/** @brief Compute density as the square of an orbital
 *
 * This routine is similar to qmfunction::multiply_real(), but it uses
 * mrcpp::square(phi) instead of mrcpp::multiply(phi, phi), which makes it
 * slightly faster.
 *
 */
Density density::compute(double prec, Orbital phi, int spin) {
    double occ = density::compute_occupation(phi, spin);
    if (std::abs(occ) < mrcpp::MachineZero) return Density(false);

    Density rho(false);
    FunctionTreeVector<3> sum_vec;
    if (phi.hasReal()) {
        FunctionTree<3> *real_2 = new FunctionTree<3>(*MRA);
        mrcpp::copy_grid(*real_2, phi.real());
        mrcpp::square(prec, *real_2, phi.real());
        sum_vec.push_back(std::make_tuple(occ, real_2));
    }
    if (phi.hasImag()) {
        FunctionTree<3> *imag_2 = new FunctionTree<3>(*MRA);
        mrcpp::copy_grid(*imag_2, phi.imag());
        mrcpp::square(prec, *imag_2, phi.imag());
        sum_vec.push_back(std::make_tuple(occ, imag_2));
    }

    rho.alloc(NUMBER::Real);
    if (sum_vec.size() > 0) {
        mrcpp::build_grid(rho.real(), sum_vec);
        mrcpp::add(-1.0, rho.real(), sum_vec, 0);
        mrcpp::clear(sum_vec, true);
    } else {
        rho.real().setZero();
    }
    return rho;
}

/** @brief Compute local density as the sum of own (MPI) orbitals
 */
void density::compute_local(double prec, Density &rho, OrbitalVector &Phi, int spin) {
    int N_el = orbital::get_electron_number(Phi);
    double mult_prec = prec;       // prec for rho_i = |phi_i|^2
    double add_prec = prec / N_el; // prec for rho = sum_i rho_i

    if (not rho.hasReal()) rho.alloc(NUMBER::Real);

    // Compute local density from own orbitals
    rho.real().setZero();
    for (auto &phi_i : Phi) {
        if (mpi::my_orb(phi_i)) {
            Density rho_i = density::compute(mult_prec, phi_i, spin);
            rho.add(1.0, rho_i);
            rho.crop(add_prec);
        }
    }
}

/** @brief Compute density as the sum of squared orbitals
 *
 * MPI: Each rank first computes its own local density, which is then reduced
 *      to rank = 0 and broadcasted to all ranks.
 *
 */
void density::compute(double prec, Density &rho, OrbitalVector &Phi, int spin) {
    int N_el = orbital::get_electron_number(Phi);
    double mult_prec = prec;       // prec for rho_i = |phi_i|^2
    double add_prec = prec / N_el; // prec for rho = sum_i rho_i

    if (not rho.hasReal()) rho.alloc(NUMBER::Real);

    // For numerically identical results in MPI we must first add
    // up orbital contributions onto their union grid, and THEN
    // crop the resulting density tree to the desired precision
    double part_prec = (mpi::numerically_exact) ? -1.0 : add_prec;

    // Compute local density from own orbitals
    Density rho_loc(false);
    rho_loc.alloc(NUMBER::Real);
    rho_loc.real().setZero();
    for (auto &phi_i : Phi) {
        if (mpi::my_orb(phi_i)) {
            Density rho_i = density::compute(mult_prec, phi_i, spin);
            rho_loc.add(1.0, rho_i);
            rho_loc.crop(part_prec);
        }
    }

    // Add up local contributions into the grand master
    mpi::reduce_function(part_prec, rho_loc, mpi::comm_orb);
    if (mpi::grand_master()) {
        // If numerically exact the grid is huge at this point
        if (mpi::numerically_exact) rho_loc.crop(add_prec);
    }

    if (rho.isShared()) {
        int tag = 3141;
        // MPI grand master distributes to shared masters
        mpi::broadcast_function(rho_loc, mpi::comm_sh_group);
        if (mpi::share_master()) {
            // MPI shared masters copies the function into final memory
            mrcpp::copy_grid(rho.real(), rho_loc.real());
            mrcpp::copy_func(rho.real(), rho_loc.real());
        }
        // MPI share masters distributes to their sharing ranks
        mpi::share_function(rho, 0, tag, mpi::comm_share);
    } else {
        // MPI grand master distributes to all ranks
        mpi::broadcast_function(rho_loc, mpi::comm_orb);
        // All MPI ranks copies the function into final memory
        mrcpp::copy_grid(rho.real(), rho_loc.real());
        mrcpp::copy_func(rho.real(), rho_loc.real());
    }
}

/** @brief Compute transition density as rho = sum_i |x_i><phi_i| + |phi_i><y_i|
 *
 * MPI: Each rank first computes its own local density, which is then reduced
 *      to rank = 0 and broadcasted to all ranks. The rank distribution of Phi
 *      and X/Y must be the same.
 *
 */
void density::compute(double prec, Density &rho, OrbitalVector &Phi, OrbitalVector &X, OrbitalVector &Y, int spin) {
    if (&X == &Y) {
        density::compute_X(prec, rho, Phi, X, spin);
    } else {
        density::compute_XY(prec, rho, Phi, X, Y, spin);
    }
}

/** @brief Compute transition density as rho = sum_i |x_i><phi_i| + |phi_i><x_i|
 *
 * Exploits the fact that the resulting density must be real.
 *
 * MPI: Each rank first computes its own local density, which is then reduced
 *      to rank = 0 and broadcasted to all ranks. The rank distribution of Phi
 *      and X must be the same.
 *
 */
void density::compute_X(double prec, Density &rho, OrbitalVector &Phi, OrbitalVector &X, int spin) {
    int N_el = orbital::get_electron_number(Phi);
    double mult_prec = prec;       // prec for rho_i = |x_i><phi_i| + |phi_i><x_i|
    double add_prec = prec / N_el; // prec for rho = sum_i rho_i
    if (Phi.size() != X.size()) MSG_ERROR("Size mismatch");

    if (not rho.hasReal()) rho.alloc(NUMBER::Real);

    // For numerically identical results in MPI we must first add
    // up orbital contributions onto their union grid, and THEN
    // crop the resulting density tree to the desired precision
    double part_prec = (mpi::numerically_exact) ? -1.0 : add_prec;

    // Compute local density from own orbitals
    Density rho_loc(false);
    rho_loc.alloc(NUMBER::Real);
    rho_loc.real().setZero();
    for (int i = 0; i < Phi.size(); i++) {
        if (mpi::my_orb(Phi[i])) {
            if (not mpi::my_orb(X[i])) MSG_FATAL("Inconsistent MPI distribution");

            double occ = density::compute_occupation(Phi[i], spin);
            if (std::abs(occ) < mrcpp::MachineZero) continue; //next orbital if this one is not occupied!

            Density rho_i(false);
            qmfunction::multiply_real(rho_i, Phi[i], X[i], mult_prec);
            rho_loc.add(2.0 * occ, rho_i);
            rho_loc.crop(part_prec);
        }
    }

    // Add up shared contributions into the grand master
    mpi::reduce_function(part_prec, rho_loc, mpi::comm_orb);
    if (mpi::grand_master()) {
        // If numerically exact the grid is huge at this point
        if (mpi::numerically_exact) rho_loc.crop(add_prec);
        // MPI grand master copies the function into final memory
        mrcpp::copy_grid(rho.real(), rho_loc.real());
        mrcpp::copy_func(rho.real(), rho_loc.real());
    }
    rho_loc.release();

    if (rho.isShared()) {
        int tag = 3141;
        // MPI grand master distributes to shared masters
        mpi::broadcast_function(rho, mpi::comm_sh_group);
        // MPI share masters distributes to their sharing ranks
        mpi::share_function(rho, 0, tag, mpi::comm_share);
    } else {
        // MPI grand master distributes to all ranks
        mpi::broadcast_function(rho, mpi::comm_orb);
    }
}

void density::compute_XY(double prec, Density &rho, OrbitalVector &Phi, OrbitalVector &X, OrbitalVector &Y, int spin) {
    int N_el = orbital::get_electron_number(Phi);
    double mult_prec = prec;       // prec for rho_i = |x_i><phi_i| + |phi_i><y_i|
    double add_prec = prec / N_el; // prec for rho = sum_i rho_i
    if (Phi.size() != X.size()) MSG_ERROR("Size mismatch");
    if (Phi.size() != Y.size()) MSG_ERROR("Size mismatch");

    if (not rho.hasReal()) rho.alloc(NUMBER::Real);

    // For numerically identical results in MPI we must first add
    // up orbital contributions onto their union grid, and THEN
    // crop the resulting density tree to the desired precision
    double part_prec = (mpi::numerically_exact) ? -1.0 : add_prec;

    // Compute local density from own orbitals
    Density rho_loc(false);
    rho_loc.alloc(NUMBER::Real);
    rho_loc.real().setZero();
    for (int i = 0; i < Phi.size(); i++) {
        if (mpi::my_orb(Phi[i])) {
            if (not mpi::my_orb(X[i])) MSG_FATAL("Inconsistent MPI distribution");
            if (not mpi::my_orb(Y[i])) MSG_FATAL("Inconsistent MPI distribution");

            double occ = density::compute_occupation(Phi[i], spin);
            if (std::abs(occ) < mrcpp::MachineZero) continue; //next orbital if this one is not occupied!

            Density rho_x(false);
            Density rho_y(false);
            qmfunction::multiply(rho_x, X[i], Phi[i].dagger(), mult_prec);
            qmfunction::multiply(rho_y, Phi[i], Y[i].dagger(), mult_prec);

            rho_loc.add(occ, rho_x);
            rho_loc.add(occ, rho_y);
            rho_loc.crop(part_prec);
        }
    }
    Timer t_com;

    // Add up shared contributions into the grand master
    mpi::reduce_function(part_prec, rho_loc, mpi::comm_orb);
    if (mpi::grand_master()) {
        // If numerically exact the grid is huge at this point
        if (mpi::numerically_exact) rho_loc.crop(add_prec);
        // MPI grand master copies the function into final memory
        mrcpp::copy_grid(rho.real(), rho_loc.real());
        mrcpp::copy_func(rho.real(), rho_loc.real());
    }
    rho_loc.release();

    if (rho.isShared()) {
        int tag = 3141;
        // MPI grand master distributes to shared masters
        mpi::broadcast_function(rho, mpi::comm_sh_group);
        // MPI share masters distributes to their sharing ranks
        mpi::share_function(rho, 0, tag, mpi::comm_share);
    } else {
        // MPI grand master distributes to all ranks
        mpi::broadcast_function(rho, mpi::comm_orb);
    }
    t_com.stop();
    Printer::printTree(0, "Allreduce density", rho.getNNodes(NUMBER::Total), t_com.getWallTime());
}

void density::compute(double prec, Density &rho, mrcpp::GaussExp<3> &dens_exp, int spin) {
    if (not rho.hasReal()) rho.alloc(NUMBER::Real);
    mrcpp::project(prec, rho.real(), dens_exp);
}

double density::compute_occupation(Orbital &phi, int dens_spin) {
    double occ_a(0.0), occ_b(0.0), occ_p(0.0);
    if (phi.spin() == SPIN::Alpha) occ_a = (double)phi.occ();
    if (phi.spin() == SPIN::Beta) occ_b = (double)phi.occ();
    if (phi.spin() == SPIN::Paired) occ_p = (double)phi.occ();

    double occ(0.0);
    if (dens_spin == DENSITY::Total) occ = occ_a + occ_b + occ_p;
    if (dens_spin == DENSITY::Alpha) occ = occ_a + 0.5 * occ_p;
    if (dens_spin == DENSITY::Beta) occ = occ_b + 0.5 * occ_p;
    if (dens_spin == DENSITY::Spin) occ = occ_a - occ_b;

    return occ;
}

} //namespace mrchem
