#pragma once

#include "qmoperators/RankZeroTensorOperator.h"
#include "CoulombPotential.h"

/** @class CoulombOperator
 *
 * @brief Operator containing a single CoulombPotential
 *
 * This class is a simple TensorOperator realization of @class CoulombPotential.
 *
 */

namespace mrchem {

class CoulombOperator final : public RankZeroTensorOperator {
public:
 CoulombOperator(mrcpp::PoissonOperator *P,
                 OrbitalVector *Phi = nullptr,
                 OrbitalVector *Phi_1= nullptr,
                 int order = 1)
     : potential(P, Phi, Phi_1, order) {
        RankZeroTensorOperator &J = (*this);
        J = this->potential;
    }

    Density &getDensity() { return this->potential.getDensity(); }

    ComplexDouble trace(OrbitalVector &Phi) { return 0.5*RankZeroTensorOperator::trace(Phi); }

protected:
    CoulombPotential potential;
};

} //namespace mrchem
