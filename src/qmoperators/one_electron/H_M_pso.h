#pragma once

#include "AngularMomentumOperator.h"
#include "DistanceOperator.h"
#include "qmoperators/RankOneTensorOperator.h"

namespace mrchem {

class H_M_pso final : public RankOneTensorOperator<3> {
public:
    H_M_pso(mrcpp::DerivativeOperator<3> &D, const mrcpp::Coord<3> &k)
            : r_m3(3.0, k)
            , l(D, k) {
        const double alpha_2 = PHYSCONST::alpha * PHYSCONST::alpha;
        RankOneTensorOperator<3> &h = (*this);
        h[0] = alpha_2 * r_m3 * l[0];
        h[1] = alpha_2 * r_m3 * l[1];
        h[2] = alpha_2 * r_m3 * l[2];
    }

protected:
    DistanceOperator r_m3;
    AngularMomentumOperator l;
};

} //namespace mrchem
