#include "CoulombOperator.h"
#include "Density.h"
#include "FunctionTree.h"

CoulombOperator::CoulombOperator(double build_prec,
                                 const MultiResolutionAnalysis<3> &mra,
                                 OrbitalVector &phi)
        : clean(mra),
          grid(mra),
          poisson(mra, build_prec, build_prec),
          project(mra),
          density(Paired),
          potential(0),
          orbitals(&phi) {
    this->potential = new Potential(mra);
}

CoulombOperator::~CoulombOperator() {
    if (this->potential == 0) MSG_ERROR("Invalid potential");
    delete this->potential;
    this->potential = 0;
}

void CoulombOperator::setup(double prec) {
    this->apply_prec = prec;
    this->clean.setPrecision(this->apply_prec);
    this->poisson.setPrecision(this->apply_prec);
    this->project.setPrecision(this->apply_prec);
}

void CoulombOperator::clear() {
    this->apply_prec = -1.0;
    this->clean.setPrecision(-1.0);
    this->poisson.setPrecision(-1.0);
    this->project.setPrecision(-1.0);
}

int CoulombOperator::printTreeSizes() const {
    NOT_IMPLEMENTED_ABORT;
}

Orbital* CoulombOperator::operator() (Orbital &orb) {
    if (this->apply_prec < 0.0) MSG_ERROR("Uninitialized operator");
    if (this->potential == 0) MSG_ERROR("XC potential not available");
    return (*this->potential)(orb);
}

Orbital* CoulombOperator::adjoint(Orbital &orb) {
    NOT_IMPLEMENTED_ABORT;
}
