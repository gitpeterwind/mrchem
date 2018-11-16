#include "MRCPP/Printer"
#include "MRCPP/Timer"

#include "PositionOperator.h"

using mrcpp::Printer;
using mrcpp::Timer;

namespace mrchem {

PositionPotential::PositionPotential(int d, const mrcpp::Coord<3> &o)
        : QMPotential(1) {
    auto f = [d, o] (const mrcpp::Coord<3> &r) -> double {
        return r[d] - o[d];
    };

    this->func.set(f);
}

void PositionPotential::setup(double prec) {
    if (this->isSetup(prec)) return;
    this->setApplyPrec(prec);

    if (this->hasReal()) MSG_ERROR("Potential not properly cleared");
    if (this->hasImag()) MSG_ERROR("Potential not properly cleared");

    Timer timer;
    this->alloc(NUMBER::Real);
    mrcpp::project(this->apply_prec, this->real(), this->func);
    timer.stop();

    int n = this->getNNodes();
    double t = timer.getWallTime();
    Printer::printTree(1, "PositionPotential", n, t);
}

void PositionPotential::clear() {
    this->free();           // delete FunctionTree pointers
    this->clearApplyPrec(); // apply_prec = -1
}

} //namespace mrchem