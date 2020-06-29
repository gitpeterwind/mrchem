#pragma once

#include <memory>

#include "ExchangePotential.h"
#include "qmfunctions/qmfunction_fwd.h"
#include "qmoperators/QMOperator.h"

namespace mrchem {

/** @class ExchangePotentialD1
 *
 *  @brief Hartree-Fock exchange potential defined by a set of unperturbed orbitals
 *
 * The operator is defined as the Hartree-Fock exchange arising from a
 * set of unperturbed orbitals. The OrbitalVector defining the
 * operator is fixed throughout the operator life time, but the
 * orbitals themselves are allowed to change in between each
 * application. The internal exchange potentials (the operator applied
 * to it's own orbitals) can be precomputed and stored for fast
 * retrieval.
 */

class ExchangePotentialD1 final : public ExchangePotential {
public:
    ExchangePotentialD1(std::shared_ptr<mrcpp::PoissonOperator> P, std::shared_ptr<OrbitalVector> Phi, bool s = false);
    ~ExchangePotentialD1() override = default;

    friend class ExchangeOperator;

private:
    void setupInternal(double prec);
    int testPreComputed(Orbital phi_p) const;
    Orbital calcExchange(Orbital phi_p);
    void calcInternal(int i);
    void calcInternal(int i, int j);
    void setupInternal_bank(double prec);
};

} // namespace mrchem
