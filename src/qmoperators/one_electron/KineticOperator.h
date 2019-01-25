#pragma once

#include "MomentumOperator.h"
#include "qmoperators/RankZeroTensorOperator.h"

/** @class KineticOperator
 *
 * @brief Operator for kinetic energy
 *
 * This operator is constructed as the square of the more fundamental
 * MomentumOperator. The general base class functions for calculation of
 * expectation values are overwritten, as they can be improved due to
 * symmetry.
 *
 */

namespace mrchem {

class KineticOperator final : public RankZeroTensorOperator {
public:
    KineticOperator(mrcpp::DerivativeOperator<3> &D)
            : p(D) {
        RankZeroTensorOperator &p_x = this->p[0];
        RankZeroTensorOperator &p_y = this->p[1];
        RankZeroTensorOperator &p_z = this->p[2];

        RankZeroTensorOperator &t = (*this);
        t = 0.5 * (p_x * p_x + p_y * p_y + p_z * p_z);
    }

    ComplexMatrix operator()(OrbitalVector &bra, OrbitalVector &ket);
    ComplexMatrix dagger(OrbitalVector &bra, OrbitalVector &ket);

    using RankZeroTensorOperator::operator();
    using RankZeroTensorOperator::dagger;

protected:
    MomentumOperator p;
};

} //namespace mrchem
