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

#include "Permittivity.h"
#include "Cavity.h"
#include <MRCPP/MWFunctions>

namespace mrchem {

Permittivity::Permittivity(const mrchem::Cavity cavity, double epsilon_in, double epsilon_out, std::string formulation)
        : epsilon_in(epsilon_in)
        , epsilon_out(epsilon_out)
        , formulation(formulation)
        , cavity(cavity) {}

double Permittivity::evalf(const mrcpp::Coord<3> &r) const {
    auto epsilon = epsilon_in * std::exp(std::log(epsilon_out / epsilon_in) * (1 - this->cavity.evalf(r)));
    if (inverse) {
        return 1 / epsilon;
    } else {
        return epsilon;
    }
}

} // namespace mrchem
