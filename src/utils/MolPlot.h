/*
 * MRChem, a numerical real-space code for molecular electronic structure
 * calculations within the self-consistent field (SCF) approximations of quantum
 * chemistry (Hartree-Fock and Density Functional Theory).
 * Copyright (C) 2018 Stig Rune Jensen, Jonas Juselius, Luca Frediani, and contributors.
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
#include "MRCPP/Plotter"
#include "Molecule.h"
#include "Nucleus.h"

namespace mrchem {

class MolPlot : public mrcpp::Plotter<3> {
public:
    MolPlot(Molecule &mol) : mrcpp::Plotter<3>() { this->molecule = &mol; }
    virtual ~MolPlot() { this->molecule = 0; }

protected:
    Molecule *molecule;

    void writeCubeData() {
        std::ofstream &o = *this->fout;

        o << "Cube file format. Generated by MRChem.\n" << std::endl;

        int nNuclei = this->molecule->getNNuclei();
        int nPerDim = (int) floor(cbrt(this->nPoints));
        int nRealPoints = nPerDim*nPerDim*nPerDim;

        double step[3];
        for (int d = 0; d < 3; d++) {
            step[d] = (this->B[d] - this->A[d]) / (nPerDim - 1);
        }

        //	"%5d %12.6f %12.6f %12.6f\n"
        o.setf(std::ios::scientific);
        o.precision(12);
        o << nPerDim << " " << step[0] << " " << 0.0 << " " << 0.0 << std::endl;
        o << nPerDim << " " << 0.0 << " " << step[1] << " " << 0.0 << std::endl;
        o << nPerDim << " " << 0.0 << " " << 0.0 << " " << step[2] << std::endl;

        for (int i = 0; i < nNuclei; i++) {
            const Nucleus &nuc = this->molecule->getNucleus(i);
            int Z = nuc.getCharge();
            const mrcpp::Coord<3> &coord = nuc.getCoord();
            o << Z << "  0.00 " << coord[0]
              << "  " << coord[1]
              << "  " << coord[2] << std::endl;
        }

        int np = 0;
        double p = 0.0;
        double max = -1.0e10;
        double min = 1.0e10;
        double isoval = 0.0;
        for (int n = 0; n < nRealPoints; n++) {
            o << this->values[n] << " "; //12.5E
            if (n % 6 == 5)
                o << std::endl;
            if (this->values[n] < min)
                min = this->values[n];
            if (this->values[n] > max)
                max = this->values[n];
            p = abs(this->values[n]);
            if (p > 1.e-4 || p < 1.e+2) {
                np += 1;
                isoval += p;
            }
        }

        isoval = isoval / np;
        int oldprec = mrcpp::Printer::setPrecision(5);
        printout(0, "MolPlot:");
        printout(0, "   Min val:" << std::setw(13) << min);
        printout(0, "   Max val:" << std::setw(13) << max);
        println(0,  "   Isoval:" << std::setw(13) << isoval);
        mrcpp::Printer::setPrecision(oldprec);
    }
};

} //namespace mrchem

