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

#pragma once

#include "MRCPP/Printer"

#include "mrchem.h"
#include "qmoperator_fwd.h"

/** @class QMOperator
 *
 * @brief Fundamental quantum mechanical operator
 *
 * Base class to handle operators and their application in the QM sense. Used to
 * build more complicated operators through the TensorOperator classes. This class
 * hierarchy should NOT be used directly, as the most important functionality is
 * protected. A proper interface is provided through RankZeroTensorOperator.
 *
 * Notes on naming conventions of derived operator classes:
 * Direct decendants of QMOperator should START with "QM", like QMPotential, QMSpin,
 * QMMomentum. Further decendants of QMPotential should END with "Potential", like
 * PositionPotential, NuclearPotential, CoulombPotential. Decendants of the
 * TensorOperators should end with "Operator" (except the perturbation operators
 * H_E_dip, H_B_dip, etc). E.i. the NuclearOperator IS a TensorOperator that
 * CONTAINS a NuclearPotential which IS a QMPotential which IS a QMOperator.
 * Capiche?
 *
 */
namespace mrchem {

class Orbital;

class QMOperator {
public:
    QMOperator()
            : apply_prec(-1.0) {}
    virtual ~QMOperator() {}

    double prec() { return this->apply_prec; }

    friend RankZeroTensorOperator;

protected:
    double apply_prec;

    void setApplyPrec(double prec) {
        if (this->apply_prec < 0.0) {
            this->apply_prec = prec;
        } else if (not isSetup(prec)) {
            MSG_ERROR("Clear operator before setup with different prec!");
        }
    }
    void clearApplyPrec() { this->apply_prec = -1.0; }

    bool isSetup(double prec) const {
        double dPrec = std::abs(this->apply_prec - prec);
        double thrs = mrcpp::MachineZero;
        return (dPrec < thrs) ? true : false;
    }

    virtual void setup(double prec) = 0;
    virtual void clear() = 0;

    virtual Orbital apply(Orbital inp) = 0;
    virtual Orbital dagger(Orbital inp) = 0;
};

} //namespace mrchem
