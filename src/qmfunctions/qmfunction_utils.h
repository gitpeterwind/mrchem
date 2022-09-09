/*
 * MRChem, a numerical real-space code for molecular electronic structure
 * calculations within the self-consistent field (SCF) approximations of quantum
 * chemistry (Hartree-Fock and Density Functional Theory).
 * Copyright (C) 2022 Stig Rune Jensen, Luca Frediani, Peter Wind and contributors.
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

#include "mrchem.h"
#include "qmfunction_fwd.h"

namespace mrchem {
namespace qmfunction {

ComplexDouble dot(mrcpp::CplxFunc bra, mrcpp::CplxFunc ket);
ComplexDouble node_norm_dot(mrcpp::CplxFunc bra, mrcpp::CplxFunc ket, bool exact);
void deep_copy(mrcpp::CplxFunc &out, mrcpp::CplxFunc &inp);
void add(mrcpp::CplxFunc &out, ComplexDouble a, mrcpp::CplxFunc inp_a, ComplexDouble b, mrcpp::CplxFunc inp_b, double prec);
void project(mrcpp::CplxFunc &out, std::function<double(const mrcpp::Coord<3> &r)> f, int type, double prec);
void project(mrcpp::CplxFunc &out, mrcpp::RepresentableFunction<3> &f, int type, double prec);
void project(mrcpp::CplxFunc &out, mrcpp::GaussExp<3> &f, int type, double prec);
void multiply(mrcpp::CplxFunc &out, mrcpp::CplxFunc inp_a, mrcpp::CplxFunc inp_b, double prec, bool absPrec = false, bool useMaxNorms = false);
void multiply_real(mrcpp::CplxFunc &out, mrcpp::CplxFunc inp_a, mrcpp::CplxFunc inp_b, double prec, bool absPrec = false, bool useMaxNorms = false);
void multiply_imag(mrcpp::CplxFunc &out, mrcpp::CplxFunc inp_a, mrcpp::CplxFunc inp_b, double prec, bool absPrec = false, bool useMaxNorms = false);
void linear_combination(mrcpp::CplxFunc &out, const ComplexVector &c, mrcpp::CplxFuncVector &inp, double prec);

} // namespace qmfunction
} // namespace mrchem
