#pragma once

#include "qmoperators/QMOperator.h"
#include "qmoperators/RankZeroTensorOperator.h"

namespace mrchem {

class QMIdentity final : public QMOperator {
public:
    QMIdentity()
            : QMOperator() {}

protected:
    void setup(double prec) { setApplyPrec(prec); }
    void clear() { clearApplyPrec(); }

    Orbital apply(Orbital inp);
    Orbital dagger(Orbital inp);
};

class IdentityOperator final : public RankZeroTensorOperator {
public:
    IdentityOperator() {
        RankZeroTensorOperator &h = (*this);
        h = I;
    }

    ComplexDouble operator()(Orbital bra, Orbital ket);
    ComplexDouble dagger(Orbital bra, Orbital ket);

    ComplexMatrix operator()(OrbitalVector &bra, OrbitalVector &ket);
    ComplexMatrix dagger(OrbitalVector &bra, OrbitalVector &ket);

    // Necessary in order to pick up base class definitions for overloaded functions
    using RankZeroTensorOperator::operator();
    using RankZeroTensorOperator::dagger;

protected:
    QMIdentity I;
};

} //namespace mrchem
