#include "ExchangeOperator.h"
#include "MultiResolutionAnalysis.h"
#include "OrbitalVector.h"

extern MultiResolutionAnalysis<3> *MRA; // Global MRA

using namespace std;
using namespace Eigen;

ExchangeOperator::ExchangeOperator(double prec,
                                   OrbitalVector &phi,
                                   double x_fac)
        : add(-1.0),
          mult(-1.0),
          poisson(*MRA, prec),
          apply(-1.0, MRA->getMaxScale()),
          x_factor(x_fac),
          orbitals_0(&phi),
          screen(true) {
    int nOrbs = this->orbitals_0->size();
    this->tot_norms = VectorXd::Zero(nOrbs);
    this->part_norms = MatrixXd::Zero(nOrbs, nOrbs);
}

ExchangeOperator::~ExchangeOperator() {
    this->orbitals_0 = 0;
}

void ExchangeOperator::setup(double prec) {
    QMOperator::setup(prec);
    this->add.setPrecision(prec);
    this->mult.setPrecision(prec);
    this->apply.setPrecision(prec);
}

void ExchangeOperator::clear() {
    this->add.setPrecision(-1.0);
    this->mult.setPrecision(-1.0);
    this->apply.setPrecision(-1.0);
    QMOperator::clear();
}

double ExchangeOperator::getScaledPrecision(int i, int j) const {
    double scaled_prec = this->apply_prec;
    if (getScreen()) {
        double tNorm = this->tot_norms(i);
        double pNorm = max(this->part_norms(i,j), this->part_norms(j,i));
        if (tNorm > 0.0) {
            scaled_prec *= tNorm/pNorm;
        }
    }
    return scaled_prec;
}