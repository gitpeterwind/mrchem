#include "CoulombPotential.h"

extern MultiResolutionAnalysis<3> *MRA;

using namespace std;

void CoulombPotential::setup(double prec) {
    if (this->orbitals == 0) MSG_ERROR("Orbitals not initialized");

    CoulombOperator::setup(prec);

    {
        Timer timer;
        this->project(this->density, *this->orbitals);
        timer.stop();
        double t = timer.getWallTime();
        int n = this->density.getNNodes();
        TelePrompter::printTree(0, "Coulomb density", n, t);
    }

    Timer timer;
    FunctionTree<3> &rho = this->density.getDensity(Paired);
    if (not this->potential.hasReal()) {
        this->potential.real = new FunctionTree<3>(*MRA);
        this->apply(*this->potential.real, this->poisson, rho);
        this->potential.imag = 0;
    } else {
        NOT_IMPLEMENTED_ABORT;
//        int nNodes = this->clean(*this->potential.real);
//        this->poisson(*this->potential.real, rho, 0);
//        this->potential.imag = 0;
    }
    timer.stop();
    int n = this->potential.getNNodes();
    double t = timer.getWallTime();
    TelePrompter::printTree(0, "Coulomb potential", n, t);
    this->potential.setup(this->apply_prec);
}

void CoulombPotential::clear() {
    this->density.clear();
    this->potential.clear();
    CoulombOperator::clear();
}