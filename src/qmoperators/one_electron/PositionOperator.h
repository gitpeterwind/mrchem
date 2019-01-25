#pragma once

#include "qmoperators/RankOneTensorOperator.h"
#include "qmoperators/one_electron/QMPotential.h"

namespace mrchem {

class PositionPotential final : public QMPotential {
public:
    PositionPotential(int d, const mrcpp::Coord<3> &o);

protected:
    mrcpp::AnalyticFunction<3> func;

    void setup(double prec);
    void clear();
};

class PositionOperator : public RankOneTensorOperator<3> {
public:
    PositionOperator(const mrcpp::Coord<3> &o = {0.0, 0.0, 0.0})
            : r_x(0, o)
            , r_y(1, o)
            , r_z(2, o) {
        RankOneTensorOperator &r = (*this);
        r[0] = r_x;
        r[1] = r_y;
        r[2] = r_z;
    }
    virtual ~PositionOperator() = default;

protected:
    PositionPotential r_x;
    PositionPotential r_y;
    PositionPotential r_z;
};

} //namespace mrchem
