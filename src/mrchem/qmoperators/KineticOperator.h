#ifndef KINETICOPERATOR_H
#define KINETICOPERATOR_H

#include "QMOperator.h"
#include "MomentumOperator.h"

class MomentumOperator;

class KineticOperator : public QMOperator {
public:
    KineticOperator(double build_prec = -1.0);
    virtual ~KineticOperator();

    virtual void setup(double prec);
    virtual void clear();

    virtual int printTreeSizes() const;

    virtual Orbital* operator() (Orbital &orb_p);
    virtual Orbital* adjoint(Orbital &orb_p);

    virtual double operator() (Orbital &orb_i, Orbital &orb_j);
    virtual double adjoint(Orbital &orb_i, Orbital &orb_j);

    virtual Eigen::MatrixXd operator() (OrbitalVector &i_orbs, OrbitalVector &j_orbs);
    virtual Eigen::MatrixXd adjoint(OrbitalVector &i_orbs, OrbitalVector &j_orbs);

protected:
    MomentumOperator momentum_x;
    MomentumOperator momentum_y;
    MomentumOperator momentum_z;
};

#endif // KINETICOPERATOR_H