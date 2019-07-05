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

#pragma once

#include <string>
#include <vector>

#include "mrchem.h"
#include "qmfunctions/qmfunction_fwd.h"

/** @class SCF
 *
 * @brief Abstract base class for different types of SCF solvers
 *
 * The ground state and linear response SCF solvers share some common features which
 * are collected in this abstract base class. This mainly involves convergence routines
 * and printing.
 *
 */

namespace mrchem {

class SCFSolver {
public:
    SCFSolver() = default;
    virtual ~SCFSolver() = default;

    void setHistory(int hist) { this->history = hist; }
    void setRotation(int iter) { this->rotation = iter; }
    void setLocalize(bool loc) { this->localize = loc; }
    void setThreshold(double orb, double prop);
    void setOrbitalPrec(double init, double final);
    void setMaxIterations(int m_iter) { this->maxIter = m_iter; }
    void setMethodName(const std::string &name) { this->methodName = name; }

protected:
    int history{0};                      ///< Maximum length of KAIN history
    int maxIter{-1};                     ///< Maximum number of iterations
    int rotation{0};                     ///< Number of iterations between localization/diagonalization
    bool localize{false};                ///< Use localized or canonical orbitals
    double orbThrs{-1.0};                ///< Convergence threshold for norm of orbital update
    double propThrs{-1.0};               ///< Convergence threshold for property
    double orbPrec[3]{-1.0, -1.0, -1.0}; ///< Dynamic precision: [current_prec, start_prec, end_prec]
    std::string methodName;              ///< Name of electronic structure method to appear in output

    std::vector<double> error;    ///< Convergence orbital error
    std::vector<double> property; ///< Convergence property error

    virtual void reset();

    bool checkConvergence(double err_o, double err_p) const;
    bool needLocalization(int nIter, bool converged) const;
    bool needDiagonalization(int nIter, bool converged) const;

    double adjustPrecision(double error);

    double getUpdate(const std::vector<double> &vec, int i, bool absPrec) const;
    void printUpdate(int plevel, const std::string &txt, double P, double dP, double thrs) const;

    void printConvergence(bool converged) const;
    void printConvergenceHeader() const;
    void printConvergenceRow(int i) const;
    void printOrbitals(const DoubleVector &norms, const DoubleVector &errors, const OrbitalVector &Phi, int flag) const;
    void printMemory() const;
};

} // namespace mrchem