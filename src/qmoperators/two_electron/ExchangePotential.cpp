#include "MRCPP/MWOperators"
#include "MRCPP/Printer"
#include "MRCPP/Timer"

#include "ExchangePotential.h"
#include "parallel.h"
#include "qmfunctions/Orbital.h"
#include "qmfunctions/OrbitalIterator.h"
#include "qmfunctions/orbital_utils.h"
#include "qmfunctions/qmfunction_utils.h"
#include "utils/print_utils.h"

using mrcpp::Printer;
using mrcpp::Timer;

using PoissonOperator = mrcpp::PoissonOperator;
using PoissonOperator_p = std::shared_ptr<mrcpp::PoissonOperator>;
using OrbitalVector_p = std::shared_ptr<mrchem::OrbitalVector>;

namespace mrchem {

/** @brief constructor
 *
 * @param[in] P Poisson operator (does not take ownership)
 * @param[in] Phi vector of orbitals which define the exchange operator
 */
ExchangePotential::ExchangePotential(PoissonOperator_p P, OrbitalVector_p Phi, double exchange_prec)
        : orbitals(Phi)
        , poisson(P) {
    int nOrbs = this->orbitals->size();
    this->exchange_prec = exchange_prec;
}

/** @brief Perform a unitary transformation among the precomputed exchange contributions
 *
 * @param[in] U unitary matrix defining the rotation
 */
void ExchangePotential::rotate(const ComplexMatrix &U) {
    if (this->exchange.size() == 0) return;
    this->exchange = orbital::rotate(this->exchange, U, this->apply_prec);

    // NOTE: The following MPI point is currently NOT implemented!
    //
    // the last parameter, 1, means MPI will send only one orbital at a time
    // (because Exchange orbitals can be large for large molecules).
    // OrbitalAdder add(this->apply_prec, this->max_scale, 1);
    // add.rotate(this->exchange, U);
}

/** @brief determines the exchange factor to be used in the calculation of the exact exchange
 *
 * @param [in] phi_i orbital defining the K operator
 * @param [in] phi_j orbital to which K is applied
 *
 * The factor is computed in terms of the occupancy of the two orbitals and in terms of the spin
 * 0.5 factors are used in order to preserve occupancy of the set of doubly occupied orbitals
 * this-> is the orbital defining the operator whereas the input orbital (orb) is the one
 * the operator is applied to
 *
 * Occupancy: Single/Double
 * Spin: alpha/beta
 *
 * K (this->) | orb (input) | factor
 * alpha      | alpha       | 1.0
 * alpha      | beta        | 0.0
 * alpha      | double      | 0.5
 * -------------------------------
 * beta       | alpha       | 0.0
 * beta       | beta        | 1.0
 * beta       | double      | 0.5
 * -------------------------------
 * double     | alpha       | 1.0
 * double     | beta        | 1.0
 * double     | double      | 1.0
 *
 */
double ExchangePotential::getSpinFactor(Orbital phi_i, Orbital phi_j) const {
    double out = 0.0;
    if (phi_i.spin() == SPIN::Paired)
        out = 1.0;
    else if (phi_j.spin() == SPIN::Paired)
        out = 0.5;
    else if (phi_i.spin() == phi_j.spin())
        out = 1.0;
    return out;
}

/** @brief Prepare operator for application
 *
 * @param[in] prec reqested precision
 *
 * This will NOT precompute the internal exchange between the orbtials defining
 * the operator, which is done explicitly using setupInternal().
 */
void ExchangePotential::setup(double prec) {
    setApplyPrec(prec);

    int nOrbs = this->orbitals->size();
}

/** @brief Clears the Exchange Operator
 *
 *  Clears deletes the precomputed exchange contributions.
 */
void ExchangePotential::clear() {
    this->exchange.clear();
    clearApplyPrec();
}

/** @brief Applies operator potential
 *
 *  @param[in] inp input orbital
 *
 * The exchange potential is applied to the given orbital. Checks first if this
 * particular exchange contribution has been precomputed.
 */
Orbital ExchangePotential::apply(Orbital inp) {
    if (this->apply_prec < 0.0) {
        MSG_ERROR("Uninitialized operator");
        return inp.paramCopy();
    }
    int i = testPreComputed(inp);
    if (i < 0) {
        if (!mpi::my_orb(inp)) {
            MSG_WARN("Not computing exchange contributions that are not mine");
            return inp.paramCopy();
        }
        println(4, "On-the-fly exchange");
        return calcExchange(inp);
    } else {
        println(4, "Precomputed exchange");
        Orbital out = this->exchange[i].paramCopy();
        qmfunction::deep_copy(out, this->exchange[i]);
        return out;
    }
}

/** @brief Applies the adjoint of the operator
 *  \param[in] inp input orbital
 *
 * NOT IMPLEMENTED
 */
Orbital ExchangePotential::dagger(Orbital inp) {
    NOT_IMPLEMENTED_ABORT;
}

/** @brief computes phi_k*Int(phi_i*phi_j^dag/|r-r'|)
 *
 *  \param[in] phi_i orbital to be multiplied by phi_j^dag
 *  \param[in] phi_j orbital to be conjugated and multiplied by phi_i
 *  \param[in] phi_k orbital to be multiplied after application of Poisson operator
 *  \param[out] phi_out_kij result
 *  \param[out] phi_out_jij (optional), result where phi_k is replaced by phi_j (i.e. phi_k not used, and phi_j used
 * twice)
 *
 * Computes the product of phi_i and complex conjugate of phi_j,
 * then applies the Poisson operator, and multiplies the result by phi_k (and optionally by phi_j).
 * The result is given in phi_out.
 */
void ExchangePotential::calc_i_Int_jk_P(double prec,
                                        Orbital &phi_i,
                                        Orbital &phi_j,
                                        Orbital &phi_k,
                                        Orbital &phi_out_kij,
                                        Orbital *phi_out_jij) {
    mrcpp::PoissonOperator &P = *this->poisson;

    Timer timer;
    auto plevel = Printer::getPrintLevel();
    phi_out_kij.free(NUMBER::Total);
    phi_out_kij = phi_k.paramCopy(); // can be different?

    // compute phi_ij = phi_i * phi_j^dag
    Timer timermult;
    Orbital phi_ij = phi_i.paramCopy();

    double precf = prec / 10; // multiplication1 precision

    qmfunction::multiply(phi_ij, phi_i, phi_j, precf, true, true);

    double norma = phi_ij.norm();
    int Ni = phi_j.getNNodes(NUMBER::Total);
    int Nj = phi_i.getNNodes(NUMBER::Total);
    int N0 = phi_ij.getNNodes(NUMBER::Total);
    timermult.stop();

    // if the product is smaller than the precision, the result is expected to be negligible
    if (phi_ij.norm() < prec) {
        phi_ij.release();
        return;
    }

    mrcpp::FunctionTreeVector<3> phi_opt_vec; // used to steer precision of Poisson application
    if (phi_k.hasReal()) phi_opt_vec.push_back(std::make_tuple(1.0, &phi_k.real()));
    if (phi_k.hasImag()) phi_opt_vec.push_back(std::make_tuple(1.0, &phi_k.imag()));
    if (phi_out_jij != nullptr) {
        if (phi_j.hasReal()) phi_opt_vec.push_back(std::make_tuple(1.0, &phi_j.real()));
        if (phi_j.hasImag()) phi_opt_vec.push_back(std::make_tuple(1.0, &phi_j.imag()));
    }

    // compute V_ij = P[phi_ij]
    Orbital V_ij = phi_k.paramCopy();
    Timer timerV;

    precf = prec * 10; // Poisson application precision.

    if (phi_ij.hasReal()) {
        V_ij.alloc(NUMBER::Real);
        mrcpp::apply(precf, V_ij.real(), P, phi_ij.real(), phi_opt_vec, -1, true);
    }
    if (phi_ij.hasImag()) {
        V_ij.alloc(NUMBER::Imag);
        mrcpp::apply(precf, V_ij.imag(), P, phi_ij.imag(), phi_opt_vec, -1, true); // NB: phi_opt.real() must be used
    }

    phi_ij.release();
    double normV = V_ij.norm();
    timerV.stop();
    int N2 = V_ij.getNNodes(NUMBER::Total);

    // compute phi_out_kij = phi_k * V_ij
    Timer timermult2;

    precf = prec / 100; // multiplication2 precision.

    phi_out_kij = phi_k.paramCopy();
    qmfunction::multiply(phi_out_kij, phi_k, V_ij, precf, true, true);

    int N3 = phi_out_kij.getNNodes(NUMBER::Total);
    double normjij = phi_out_kij.norm();
    timermult2.stop();
    Timer timermult3;
    int N4 = 0;
    double normiij = 0.0;
    if (phi_out_jij != nullptr) {
        // compute phi_out_jij = phi_j * V_ij
        *phi_out_jij = phi_j.paramCopy();
        qmfunction::multiply(*phi_out_jij, phi_j, V_ij, precf, true, true);
        N4 = phi_out_jij->getNNodes(NUMBER::Total);
    }
    timermult3.stop();
    V_ij.release();

    println(4,
            " time " << (int)((float)timer.elapsed() * 1000) << " ms "
                     << " mult1:" << (int)((float)timermult.elapsed() * 1000) << " Pot:"
                     << (int)((float)timerV.elapsed() * 1000) << " mult2:" << (int)((float)timermult2.elapsed() * 1000)
                     << " " << (int)((float)timermult3.elapsed() * 1000) << " Nnodes: " << Ni << " " << Nj << " " << N0
                     << " " << N2 << " " << N3 << " " << N4 << " norms " << norma << " " << normV << " " << normjij
                     << "  " << normiij);
}

/** @brief Test if a given contribution has been precomputed
 *
 * @param[in] phi_p orbital for which the check is performed
 *
 * If the given contribution has been precomputed, it is simply copied,
 * without additional recalculation.
 */
int ExchangePotential::testPreComputed(Orbital phi_p) const {
    const OrbitalVector &Phi = *this->orbitals;
    const OrbitalVector &Ex = this->exchange;

    int out = -1;
    if (Ex.size() == Phi.size()) {
        for (int i = 0; i < Phi.size(); i++) {
            if (&Phi[i].real() == &phi_p.real() and &Phi[i].imag() == &phi_p.imag()) {
                out = i;
                break;
            }
        }
    }
    return out;
}

} // namespace mrchem
