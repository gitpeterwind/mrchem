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

#include "MRCPP/MWOperators"

#include "MRDFT.h"
#include "xc_utils.h"
#include "utils/Bank.h"

namespace mrdft {

/** @brief Compute XC potentials from densities
 *
 * This routine computes the XC energy density and potentials on the
 * union grid of all the density input functions. Functional evaluation
 * and subsequent contraction is done node by node, to avoid explicit
 * construction of the huge number of intermediate functions.
 *
 * Ordering without spin:
 * inp_vec[0] = rho_0 (unperturbed)
 * inp_vec[1] = rho_1 (first order perturbed)
 * ...
 * out_vec[0] = f_xc (XC energy density)
 * out_vec[1] = v_xc (XC potential)
 *
 * Ordering with spin:
 * inp_vec[0] = alpha_0 (unperturbed)
 * inp_vec[1] = beta_0
 * inp_vec[2] = alpha_1 (first order perturbed)
 * inp_vec[3] = beta_1
 * ...
 * out_vec[0] = f_xc (XC energy density)
 * out_vec[1] = v_xc_a (XC alpha potential)
 * out_vec[2] = v_xc_b (XC beta potential)
 */
mrcpp::FunctionTreeVector<3> MRDFT::evaluate(mrcpp::FunctionTreeVector<3> &inp) {
    grid().unify(inp);
    functional().preprocess(inp);
    mrcpp::FunctionTreeVector<3> xcInpVec = functional().setupXCInput();
    mrcpp::FunctionTreeVector<3> ctrInpVec = functional().setupCtrInput();

    // auto nOutXC = functional().getXCOutputLength();
    // mrcpp::FunctionTreeVector<3> xcOutVec = grid().generate(nOutXC);

    auto nOutCtr = functional().getCtrOutputLength();
    mrcpp::FunctionTreeVector<3> ctrOutVec = grid().generate(nOutCtr);

    // divide nNodes into parts assigned to each MPI rank
    int nNodes = grid().size();
    int d = (nNodes + mrchem::mpi::orb_size - 1) / mrchem::mpi::orb_size; // max number of nodes treated by each rank
    int n_start = mrchem::mpi::orb_rank * d;
    int n_end = std::min(nNodes, (mrchem::mpi::orb_rank + 1) * d);

    std::vector<Eigen::MatrixXd> ctrOutDataVec(n_end - n_start); // to put the XC data for each local node.

#pragma omp parallel
    {
#pragma omp for schedule(guided)
        for (int n = n_start; n < n_end; n++) {
            auto xcInpNodes = xc_utils::fetch_nodes(n, xcInpVec);
            auto xcInpData = xc_utils::compress_nodes(xcInpNodes);

            auto xcOutData = functional().evaluate(xcInpData);
            // auto xcOutNodes = xc_utils::fetch_nodes(n, xcOutVec);
            // xc_utils::expand_nodes(xcOutNodes, xcOutData);

            auto ctrInpNodes = xc_utils::fetch_nodes(n, ctrInpVec);
            auto ctrInpData = xc_utils::compress_nodes(ctrInpNodes);
            auto ctrOutData = functional().contract(xcOutData, ctrInpData);

            ctrOutDataVec[n - n_start] = std::move(ctrOutData);
            // auto ctrOutNodes = xc_utils::fetch_nodes(n, ctrOutVec);
            // xc_utils::expand_nodes(ctrOutNodes, ctrOutData);
        }
    }
    mrcpp::clear(xcInpVec, false);
    mrcpp::clear(ctrInpVec, false);

    mrchem::BankAccount ctrOutBank; // to put the ctrOutDataVec;
    // note that mpi cannot run in multiple omp threads
    int nFcs = ctrOutDataVec[0].rows();
    int nPts = ctrOutDataVec[0].cols();
    int size = nFcs * nPts;
    for (int n = n_start; n < n_end; n++) { ctrOutBank.put_data(n, size, ctrOutDataVec[n - n_start].data()); }

    // fetch all nodes from bank
    for (int n = 0; n < nNodes; n++) {
        Eigen::MatrixXd ctrOutData = Eigen::MatrixXd::Zero(nFcs, nPts);
        ctrOutBank.get_data(n, size, ctrOutData.data());
        auto ctrOutNodes = xc_utils::fetch_nodes(n, ctrOutVec);
        xc_utils::expand_nodes(ctrOutNodes, ctrOutData);
    }

    // Reconstruct raw xcfun output functions
    /*
    for (auto i = 0; i < xcOutVec.size(); i++) {
        auto &f_i = mrcpp::get_func(xcOutVec, i);
        f_i.mwTransform(mrcpp::BottomUp);
        f_i.calcSquareNorm();
    }
    mrcpp::clear(xcOutVec, true);
    */

    // Reconstruct contracted output functions
    for (auto i = 0; i < ctrOutVec.size(); i++) {
        auto &f_i = mrcpp::get_func(ctrOutVec, i);
        f_i.mwTransform(mrcpp::BottomUp);
        f_i.calcSquareNorm();
    }

    auto potOutVec = functional().postprocess(ctrOutVec);
    mrcpp::clear(ctrOutVec, true);
    functional().clear();

    return potOutVec;
}

} // namespace mrdft
