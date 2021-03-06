#pragma once

#include "qmfunctions/QMFunction.h"
#include "qmoperators/QMOperator.h"

/** @class QMPotential
 *
 * @brief Operator defining a multiplicative potential
 *
 * Inherits the general features of a complex function from QMFunction and
 * implements the multiplication of this function with an Orbital. The actual
 * function representing the operator needs to be implemented in the derived
 * classes, where the *re and *im FunctionTree pointers should be assigned in
 * the setup() function and deallocated in the clear() function.
 *
 */

namespace mrchem {

class QMPotential : public QMFunction, public QMOperator {
public:
    explicit QMPotential(int adap, bool shared = false);
    QMPotential(const QMPotential &pot) = delete;
    QMPotential &operator=(const QMPotential &pot) = delete;
    virtual ~QMPotential();

protected:
    int adap_build;

    virtual Orbital apply(Orbital inp);
    virtual Orbital dagger(Orbital inp);

    void calcRealPart(Orbital &out, Orbital &inp, bool dagger);
    void calcImagPart(Orbital &out, Orbital &inp, bool dagger);
};

} // namespace mrchem
