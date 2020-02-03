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
ExchangePotential::ExchangePotential(PoissonOperator_p P, OrbitalVector_p Phi, bool s)
        : screen(s)
        , orbitals(Phi)
        , poisson(P) {
    int nOrbs = this->orbitals->size();
    this->tot_norms = DoubleVector::Zero(nOrbs);
    this->part_norms = DoubleMatrix::Zero(nOrbs, nOrbs);
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
    if (tot_norms.size() != nOrbs) this->tot_norms = DoubleVector::Zero(nOrbs);
    if (part_norms.rows() != nOrbs) this->part_norms = DoubleMatrix::Zero(nOrbs, nOrbs);
    if (part_norms.cols() != nOrbs) this->part_norms = DoubleMatrix::Zero(nOrbs, nOrbs);
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

/** @brief Computes the exchange potential on the fly
 *
 *  \param[in] inp input orbital
 *
 * The exchange potential is computed and applied on the fly to the given orbital.
 */
Orbital ExchangePotential::calcExchange(Orbital phi_p) {
    Timer timer;

    double prec = this->apply_prec;
    OrbitalVector &Phi = *this->orbitals;
    mrcpp::PoissonOperator &P = *this->poisson;

    ComplexVector coef_vec(Phi.size());
    QMFunctionVector func_vec;

    OrbitalIterator iter(Phi);
    while (iter.next()) {
        for (int i = 0; i < iter.get_size(); i++) {
            Orbital &phi_i = iter.orbital(i);

            double spin_fac = getSpinFactor(phi_i, phi_p);
            if (std::abs(spin_fac) < mrcpp::MachineZero) continue;

            // compute phi_ip = phi_i^dag * phi_p
            Orbital phi_ip = phi_p.paramCopy();
            qmfunction::multiply(phi_ip, phi_i.dagger(), phi_p, -1.0);

            // compute V_ip = P[phi_ip]
            Orbital V_ip = phi_p.paramCopy();
            if (phi_ip.hasReal()) {
                V_ip.alloc(NUMBER::Real);
                mrcpp::apply(prec, V_ip.real(), P, phi_ip.real());
            }
            if (phi_ip.hasImag()) {
                V_ip.alloc(NUMBER::Imag);
                mrcpp::apply(prec, V_ip.imag(), P, phi_ip.imag());
            }
            phi_ip.release();

            // compute phi_iip = phi_i * V_ip
            Orbital phi_iip = phi_p.paramCopy();
            qmfunction::multiply(phi_iip, phi_i, V_ip, -1.0);

            coef_vec(i) = spin_fac / phi_i.squaredNorm();
            func_vec.push_back(phi_iip);
        }
    }

    // compute ex_p = sum_i c_i*phi_iip
    Orbital ex_p = phi_p.paramCopy();
    qmfunction::linear_combination(ex_p, coef_vec, func_vec, -1.0);
    print_utils::qmfunction(3, "Applied exchange", ex_p, timer);

    return ex_p;
}

/** @brief precomputes the exchange potential
 *
 *  @param[in] phi_p input orbital
 *
 * The exchange potential is (pre)computed among the orbitals that define the operator
 */
void ExchangePotential::setupInternal(double prec) {

    if (mpi::bank_size > 0) {
        setupInternal_bank(prec);
        return;
    }
    setApplyPrec(prec);

    if (this->exchange.size() != 0) MSG_ERROR("Exchange not properly cleared");

    OrbitalVector &Phi = *this->orbitals;
    OrbitalVector &Ex = this->exchange;

    // Diagonal must come first because it's NOT in-place
    Timer timer;
    for (int i = 0; i < Phi.size(); i++) calcInternal(i);

    // Off-diagonal must come last because it IS in-place
    OrbitalIterator iter(Phi, true); // symmetric iterator
    Orbital ex_rcv;
    while (iter.next(1)) { // one orbital at the time
        if (iter.get_size() > 0) {
            Orbital &phi_i = iter.orbital(0);
            int idx = iter.idx(0);
            for (int j = 0; j < Phi.size(); j++) {
                if (mpi::my_orb(phi_i) and j <= idx) continue; // compute only i<j for own block
                Orbital &phi_j = (*this->orbitals)[j];
                if (mpi::my_orb(phi_j)) calcInternal(idx, j, phi_i, phi_j);
            }
            // must send exchange_i to owner and receive exchange computed by other
            if (iter.get_step(0) and not mpi::my_orb(phi_i))
                mpi::send_function(Ex[idx], phi_i.rankID(), idx, mpi::comm_orb);

            if (iter.get_sent_size()) {
                // get exchange from where we sent orbital to
                int idx_sent = iter.get_idx_sent(0);
                int sent_rank = iter.get_rank_sent(0);
                mpi::recv_function(ex_rcv, sent_rank, idx_sent, mpi::comm_orb);
                Ex[idx_sent].add(1.0, ex_rcv);
            }

            if (not iter.get_step(0) and not mpi::my_orb(phi_i))
                mpi::send_function(Ex[idx], phi_i.rankID(), idx, mpi::comm_orb);
            if (not mpi::my_orb(Ex[idx])) Ex[idx].free(NUMBER::Total);
        } else {
            if (iter.get_sent_size()) { // must receive exchange computed by other
                // get exchange from where we sent orbital to
                int idx_sent = iter.get_idx_sent(0);
                int sent_rank = iter.get_rank_sent(0);
                mpi::recv_function(ex_rcv, sent_rank, idx_sent, mpi::comm_orb);
                Ex[idx_sent].add(1.0, ex_rcv);
            }
        }
        ex_rcv.free(NUMBER::Total);
    }

    // Collect info from the calculation
    for (int i = 0; i < Phi.size(); i++) {
        if (mpi::my_orb(Phi[i])) this->tot_norms(i) = Ex[i].norm();
    }

    mpi::allreduce_vector(this->tot_norms, mpi::comm_orb);  // to be checked
    mpi::allreduce_matrix(this->part_norms, mpi::comm_orb); // to be checked

    auto n = orbital::get_n_nodes(Ex);
    auto m = orbital::get_size_nodes(Ex);
    auto t = timer.elapsed();
    mrcpp::print::tree(2, "Hartree-Fock exchange", n, m, t);
}

/** @brief precomputes the exchange potential
 *
 *  @param[in] phi_p input orbital
 *
 * The exchange potential is (pre)computed among the orbitals that define the operator
 */
void ExchangePotential::setupInternal_bank(double prec) {
    setApplyPrec(prec);

    if (this->exchange.size() != 0) MSG_ERROR("Exchange not properly cleared");

    OrbitalVector &Phi = *this->orbitals;
    OrbitalVector &Ex = this->exchange;

    // Diagonal must come first because it's NOT in-place
    Timer timer;
    for (int i = 0; i < Phi.size(); i++) calcInternal(i);
    if (mpi::grand_master())
        std::cout << " Exchange time diagonal " << (int)((float)timer.elapsed() * 1000) << " ms " << std::endl;

    // 1) Save all orbitals in Bank
    Timer timerS;
    for (int i = 0; i < Phi.size(); i++) {
        if (not mpi::my_orb(Phi[i])) continue;
        Orbital &Phi_i = (*this->orbitals)[i];
        mpi::orb_bank.put_orb(i, Phi_i);
    }
    mpi::barrier(mpi::comm_orb);
    timerS.stop();

    Orbital ex_rcv;
    int N = (int)Phi.size();
    int count = 0;
    int foundcount = 0;
    for (int j = 0; j < N; j++) {
        if (not mpi::my_orb(Phi[j])) continue; // compute only own j
        for (int i = 0; i < N; i++) {
            if (i == j) continue;
            // compute only half of matrix (in band where i-j < size/2 %size)
            if (i > j + (N - 1) / 2 and i > j) continue;
            if (i > j - (N + 1) / 2 and i < j) continue;
            Orbital phi_i;
            int ss = mpi::orb_bank.get_orb(i, phi_i);
            Orbital &phi_j = (*this->orbitals)[j];
            //            if (mpi::my_orb(phi_j)) calcInternal(j, i, phi_j, phi_i);
            Orbital phi_jij;
            Orbital phi_iij;
            if (mpi::my_orb(phi_j)) {
                double j_fac = getSpinFactor(phi_j, phi_i);
                // calcInternal_test(j, i, phi_j, phi_i,phi_iij, phi_jij);
                calc_i_Int_jk_P(prec, phi_i, phi_j, phi_i, phi_iij, &phi_jij);
                Ex[j].add(j_fac, phi_iij);
                phi_iij.release();

                if (phi_jij.getNNodes(NUMBER::Total) > 0) {
                    // must store exchange_i
                    if (not mpi::my_orb(Phi[i])) {
                        // if(not mpi::my_orb(Phi[i]) and Ex[i].getNNodes(NUMBER::Total) > 0){
                        timerS.resume();
                        //               mpi::orb_bank.put_orb(i+(j+1)*N, Ex[i]);
                        //               Ex[i].free(NUMBER::Total);
                        timerS.stop();
                        mpi::orb_bank.put_orb(i + (j + 1) * N, phi_jij);
                        phi_jij.release();
                    } else {
                        double i_fac = getSpinFactor(phi_i, phi_j);
                        Ex[i].add(i_fac, phi_jij);
                        phi_jij.release();
                    }
                }
            }
            count++;
            if (count % 100 == 0) {
                timerS.resume();
                // regularly fetch available exchange computed by other, so as to not overload the bank.
                for (int jj = 0; jj < N; jj++) {
                    if (not mpi::my_orb(Phi[jj])) continue; // compute only own j
                    for (int ii = 0; ii < N; ii++) {
                        if (ii == jj) continue;
                        if (mpi::my_orb(Phi[ii])) continue; // fetch only other's i
                        // Fetch other half of matrix (in band where i-j > size/2 %size)
                        if ((ii > jj + (N - 1) / 2 and ii > jj) or (ii > jj - (N + 1) / 2 and ii < jj)) {
                            double j_fac = getSpinFactor(Phi[jj], Phi[ii]);
                            int found = mpi::orb_bank.get_orb_del(jj + (ii + 1) * N, ex_rcv);
                            foundcount += found;
                            if (found) Ex[jj].add(j_fac, ex_rcv);
                        }
                    }
                    Ex[jj].real().crop(prec);
                }
                ex_rcv.free(NUMBER::Total);
                if (mpi::grand_master())
                    std::cout << " FETCHED " << foundcount << " Exchange contributions from bank "
                              << (int)((float)timerS.elapsed() * 1000) << " ms " << std::endl;
                timerS.stop();
            }
        }
    }
    if (mpi::grand_master())
        std::cout << " Time exchanges compute " << (int)((float)timer.elapsed() * 1000) << " ms " << std::endl;
    mpi::barrier(mpi::comm_orb);
    if (mpi::grand_master())
        std::cout << " Time exchanges all compute " << (int)((float)timer.elapsed() * 1000) << " ms " << std::endl;
    timerS.resume();
    for (int j = 0; j < N; j++) {
        if (not mpi::my_orb(Phi[j])) continue; // compute only own j
        for (int i = 0; i < N; i++) {
            if (i == j) continue;
            if (mpi::my_orb(Phi[i])) continue; // fetch only other's i
            // Fetch other half of matrix (in band where i-j > size/2 %size)
            if ((i > j + (N - 1) / 2 and i > j) or (i > j - (N + 1) / 2 and i < j)) {
                double j_fac = getSpinFactor(Phi[j], Phi[i]);
                int found = mpi::orb_bank.get_orb_del(j + (i + 1) * N, ex_rcv);
                foundcount += found;
                if (found) Ex[j].add(j_fac, ex_rcv);
            }
        }
        Ex[j].real().crop(prec);
        ex_rcv.free(NUMBER::Total);
    }
    if (mpi::grand_master())
        std::cout << " fetched in total " << foundcount << " Exchange contributions from bank " << std::endl;
    timerS.stop();
    if (mpi::grand_master())
        std::cout << " Time send/rcv exchanges " << (int)((float)timerS.elapsed() * 1000) << " ms " << std::endl;
    // Collect info from the calculation
    for (int i = 0; i < Phi.size(); i++) {
        if (mpi::my_orb(Phi[i])) this->tot_norms(i) = Ex[i].norm();
    }
    mpi::orb_bank.clear_all(mpi::orb_rank, mpi::comm_orb);

    mpi::allreduce_vector(this->tot_norms, mpi::comm_orb);  // to be checked
    mpi::allreduce_matrix(this->part_norms, mpi::comm_orb); // to be checked

    auto n = orbital::get_n_nodes(Ex);
    auto m = orbital::get_size_nodes(Ex);
    auto t = timer.elapsed();
    mrcpp::print::tree(2, "Hartree-Fock exchange", n, m, t);
}

/** @brief Computes the diagonal part of the internal exchange potential
 *
 *  \param[in] i orbital index
 *
 * The diagonal term K_ii is computed.
 */
void ExchangePotential::calcInternal(int i) {
    Orbital &phi_i = (*this->orbitals)[i];

    if (mpi::my_orb(phi_i)) {
        double prec = std::min(getScaledPrecision(i, i), 1.0e-1);
        mrcpp::PoissonOperator &P = *this->poisson;

        // compute phi_ii = phi_i^dag * phi_i
        Orbital phi_ii = phi_i.paramCopy();
        qmfunction::multiply(phi_ii, phi_i.dagger(), phi_i, prec);

        // compute V_ii = P[phi_ii]
        Orbital V_ii = phi_i.paramCopy();
        if (phi_ii.hasReal()) {
            V_ii.alloc(NUMBER::Real);
            mrcpp::apply(prec, V_ii.real(), P, phi_ii.real());
        }
        if (phi_ii.hasImag()) {
            V_ii.alloc(NUMBER::Imag);
            mrcpp::apply(prec, V_ii.imag(), P, phi_ii.imag());
        }
        phi_ii.release();

        // compute phi_iii = phi_i * V_ii
        Orbital phi_iii = phi_i.paramCopy();
        qmfunction::multiply(phi_iii, phi_i, V_ii, prec);
        phi_iii.rescale(1.0 / phi_i.squaredNorm());
        this->part_norms(i, i) = phi_iii.norm();
        this->exchange.push_back(phi_iii);
    } else {
        // put empty orbital to fill the exchange vector
        Orbital phi_iii = phi_i.paramCopy();
        this->exchange.push_back(phi_iii);
    }
}

/** @brief computes the off-diagonal part of the exchange potential
 *
 *  \param[in] i first orbital index
 *  \param[in] j second orbital index
 *
 * The off-diagonal terms K_ij and K_ji are computed.
 */
void ExchangePotential::calcInternal(int i, int j, Orbital &phi_i, Orbital &phi_j) {
    mrcpp::PoissonOperator &P = *this->poisson;
    OrbitalVector &Phi = *this->orbitals;
    OrbitalVector &Ex = this->exchange;
    Timer timer;

    if (i == j) MSG_ABORT("Cannot handle diagonal term");
    if (Ex.size() != Phi.size()) MSG_ABORT("Size mismatch");
    if (phi_i.hasImag() or phi_j.hasImag()) MSG_ABORT("Orbitals must be real");

    double i_fac = getSpinFactor(phi_i, phi_j);
    double j_fac = getSpinFactor(phi_j, phi_i);

    double thrs = mrcpp::MachineZero;
    if (std::abs(i_fac) < thrs or std::abs(j_fac) < thrs) return;

    double prec = this->apply_prec;
    prec = std::min(prec, 1.0e-1); // very low precision does not work properly

    // compute phi_ij = phi_i^dag * phi_j (dagger NOT used, orbitals must be real!)
    Timer timermult;
    Orbital phi_ij = phi_i.paramCopy();
    double precf = prec;
    phi_ij.alloc(NUMBER::Real);
    //    qmfunction::multiply(phi_ij, phi_i, phi_j, prec);
    mrcpp::multiply(precf, phi_ij.real(), 1.0, phi_i.real(), phi_j.real(), -1, true);
    double norma = phi_ij.norm();
    int N0 = phi_ij.getNNodes(NUMBER::Total);
    timermult.stop();

    if (phi_ij.norm() < prec) {
        phi_ij.release();
        return;
    }
    Timer timerd;
    Orbital phi_ii = phi_i.paramCopy();
    Orbital phi_jj = phi_j.paramCopy();
    phi_ii.alloc(NUMBER::Real);
    // phi_jj.alloc(NUMBER::Real);
    mrcpp::copy_grid(phi_ii.real(), phi_i.real());
    mrcpp::copy_func(phi_ii.real(), phi_i.real());
    // mrcpp::copy_grid(phi_jj.real(), phi_j.real());

    // mrcpp::multiply(precf, phi_ii.real(), 1.0, phi_i.real(), phi_i.real(), 0, false);
    // mrcpp::multiply(precf, phi_jj.real(), 1.0, phi_j.real(), phi_j.real(), 0, false);
    // mrcpp::copy_grid(phi_ii.real(), phi_i.real());
    // mrcpp::copy_func(phi_ii.real(), phi_i.real());
    // mrcpp::phi_ii.real().square();
    // phi_ii.add(1.0, phi_jj);

    phi_ii.absadd(1.0, phi_j);
    phi_ii.real().makeMaxSquareNorms();
    int N1 = phi_ii.getNNodes(NUMBER::Total);
    timerd.stop();

    // compute V_ij = P[phi_ij]
    Orbital V_ij = phi_i.paramCopy();
    Orbital V_ji = phi_i.paramCopy();
    Timer timerV;
    precf = prec * 0.2;
    precf = prec * 10;
    precf = std::min(precf, 1.0e-1); // very low precision does not work properly
    if (phi_ij.hasReal()) {
        V_ij.alloc(NUMBER::Real);
        // V_ji.alloc(NUMBER::Real);
        // mrcpp::apply(prec, V_ij.real(), P, phi_ij.real());
        // mrcpp::apply(precf, V_ij.real(), P, phi_ij.real(), -1, true);
        // use density weighted precision
        mrcpp::apply(precf, V_ij.real(), P, phi_ij.real(), phi_ii.real(), -1, true);
        // std::cout<<mpi::orb_rank<<" "<<i<<" "<<j<<" applyok "<<std::endl;
    }
    double normV = V_ij.norm();
    timerV.stop();
    int N2 = V_ij.getNNodes(NUMBER::Total);
    if (phi_ij.hasImag()) {
        MSG_ABORT("Orbitals must be real");
        V_ij.alloc(NUMBER::Imag);
        mrcpp::apply(prec, V_ij.imag(), P, phi_ij.imag());
    }
    phi_ij.release();

    // compute phi_jij = phi_j * V_ij
    Timer timermult2;
    precf = prec;
    Orbital phi_jij = phi_j.paramCopy();
    phi_jij.alloc(NUMBER::Real);
    mrcpp::copy_grid(phi_jij.real(), phi_j.real());
    // qmfunction::multiply(phi_jij, phi_j, V_ij, prec);
    mrcpp::multiply(precf, phi_jij.real(), 1.0, phi_j.real(), V_ij.real(), -1, true);
    int N3 = phi_jij.getNNodes(NUMBER::Total);
    phi_jij.rescale(1.0 / phi_j.squaredNorm());
    double normjij = phi_jij.norm();
    this->part_norms(j, i) = phi_jij.norm();
    timermult2.stop();

    // compute phi_iij = phi_i * V_ij
    Timer timermult3;
    Orbital phi_iij = phi_i.paramCopy();
    phi_iij.alloc(NUMBER::Real);
    mrcpp::copy_grid(phi_iij.real(), phi_i.real());
    mrcpp::multiply(precf, phi_iij.real(), 1.0, phi_i.real(), V_ij.real(), -1, true);
    int N4 = phi_iij.getNNodes(NUMBER::Total);
    timermult3.stop();
    phi_iij.rescale(1.0 / phi_i.squaredNorm());
    this->part_norms(i, j) = phi_iij.norm();
    V_ij.release();
    V_ji.release();
    double normiij = phi_iij.norm();

    // compute x_i += phi_jij
    Ex[i].add(i_fac, phi_jij);
    phi_jij.release();

    // compute x_j += phi_iij
    Ex[j].add(j_fac, phi_iij);
    phi_iij.release();
    if (mpi::grand_master())
        std::cout << i << " " << j << " prec " << prec << " time " << (int)((float)timer.elapsed() * 1000) << " ms "
                  << " mult1:" << (int)((float)timermult.elapsed() * 1000)
                  << " dens:" << (int)((float)timerd.elapsed() * 1000)
                  << " Pot:" << (int)((float)timerV.elapsed() * 1000)
                  << " mult2:" << (int)((float)timermult2.elapsed() * 1000) << " "
                  << (int)((float)timermult3.elapsed() * 1000) << " Nnodes: " << N0 << " " << N1 << " " << N2 << " "
                  << N3 << " " << N4 << " norms " << norma << " " << normV << " " << normjij << "  " << normiij
                  << std::endl;
}

/** @brief computes the off-diagonal part of the exchange potential
 *
 *  \param[in] i first orbital index
 *  \param[in] j second orbital index
 *
 * The off-diagonal terms K_ij and K_ji are computed.
 */
void ExchangePotential::calcInternal_test(int i,
                                          int j,
                                          Orbital &phi_i,
                                          Orbital &phi_j,
                                          Orbital &phi_jij,
                                          Orbital &phi_iij) {
    mrcpp::PoissonOperator &P = *this->poisson;
    OrbitalVector &Phi = *this->orbitals;
    OrbitalVector &Ex = this->exchange;
    Timer timer;

    if (i == j) MSG_ABORT("Cannot handle diagonal term");
    if (Ex.size() != Phi.size()) MSG_ABORT("Size mismatch");
    if (phi_i.hasImag() or phi_j.hasImag()) MSG_ABORT("Orbitals must be real");

    double i_fac = getSpinFactor(phi_i, phi_j);
    double j_fac = getSpinFactor(phi_j, phi_i);
    phi_iij.alloc(NUMBER::Real);
    phi_jij.alloc(NUMBER::Real);

    double thrs = mrcpp::MachineZero;
    if (std::abs(i_fac) < thrs or std::abs(j_fac) < thrs) return;

    double prec = this->apply_prec;
    prec = std::min(prec, 1.0e-1); // very low precision does not work properly

    // compute phi_ij = phi_i^dag * phi_j (dagger NOT used, orbitals must be real!)
    Timer timermult;
    Orbital phi_ij = phi_i.paramCopy();
    double precf = prec;
    phi_ij.alloc(NUMBER::Real);
    //    qmfunction::multiply(phi_ij, phi_i, phi_j, prec);
    mrcpp::multiply(precf, phi_ij.real(), 1.0, phi_i.real(), phi_j.real(), -1, true);
    double norma = phi_ij.norm();
    int N0 = phi_ij.getNNodes(NUMBER::Total);
    timermult.stop();

    if (phi_ij.norm() < prec) {
        phi_ij.release();
        return;
    }
    Timer timerd;
    Orbital phi_ii = phi_i.paramCopy();
    Orbital phi_jj = phi_j.paramCopy();
    phi_ii.alloc(NUMBER::Real);
    // phi_jj.alloc(NUMBER::Real);
    mrcpp::copy_grid(phi_ii.real(), phi_i.real());
    mrcpp::copy_func(phi_ii.real(), phi_i.real());
    // mrcpp::copy_grid(phi_jj.real(), phi_j.real());

    // mrcpp::multiply(precf, phi_ii.real(), 1.0, phi_i.real(), phi_i.real(), 0, false);
    // mrcpp::multiply(precf, phi_jj.real(), 1.0, phi_j.real(), phi_j.real(), 0, false);
    // mrcpp::copy_grid(phi_ii.real(), phi_i.real());
    // mrcpp::copy_func(phi_ii.real(), phi_i.real());
    // mrcpp::phi_ii.real().square();
    // phi_ii.add(1.0, phi_jj);

    phi_ii.absadd(1.0, phi_j);
    phi_ii.real().makeMaxSquareNorms();
    int N1 = phi_ii.getNNodes(NUMBER::Total);
    timerd.stop();

    // compute V_ij = P[phi_ij]
    Orbital V_ij = phi_i.paramCopy();
    Orbital V_ji = phi_i.paramCopy();
    Timer timerV;
    precf = prec * 0.2;
    precf = prec * 10;
    precf = std::min(precf, 1.0e-1); // very low precision does not work properly
    if (phi_ij.hasReal()) {
        V_ij.alloc(NUMBER::Real);
        // V_ji.alloc(NUMBER::Real);
        // mrcpp::apply(prec, V_ij.real(), P, phi_ij.real());
        // mrcpp::apply(precf, V_ij.real(), P, phi_ij.real(), -1, true);
        // use density weighted precision
        mrcpp::apply(precf, V_ij.real(), P, phi_ij.real(), phi_ii.real(), -1, true);
        // std::cout<<mpi::orb_rank<<" "<<i<<" "<<j<<" applyok "<<std::endl;
    }
    double normV = V_ij.norm();
    timerV.stop();
    int N2 = V_ij.getNNodes(NUMBER::Total);
    if (phi_ij.hasImag()) {
        MSG_ABORT("Orbitals must be real");
        V_ij.alloc(NUMBER::Imag);
        mrcpp::apply(prec, V_ij.imag(), P, phi_ij.imag());
    }
    phi_ij.release();

    // compute phi_jij = phi_j * V_ij
    Timer timermult2;
    precf = prec;
    //    Orbital phi_jij = phi_j.paramCopy();
    // phi_jij.alloc(NUMBER::Real);
    mrcpp::copy_grid(phi_jij.real(), phi_j.real());
    // qmfunction::multiply(phi_jij, phi_j, V_ij, prec);
    mrcpp::multiply(precf, phi_jij.real(), 1.0, phi_j.real(), V_ij.real(), -1, true);
    int N3 = phi_jij.getNNodes(NUMBER::Total);
    phi_jij.rescale(1.0 / phi_j.squaredNorm());
    double normjij = phi_jij.norm();
    this->part_norms(j, i) = phi_jij.norm();
    timermult2.stop();

    // compute phi_iij = phi_i * V_ij
    Timer timermult3;
    //    Orbital phi_iij = phi_i.paramCopy();
    // phi_iij.alloc(NUMBER::Real);
    mrcpp::copy_grid(phi_iij.real(), phi_i.real());
    mrcpp::multiply(precf, phi_iij.real(), 1.0, phi_i.real(), V_ij.real(), -1, true);
    int N4 = phi_iij.getNNodes(NUMBER::Total);
    timermult3.stop();
    phi_iij.rescale(1.0 / phi_i.squaredNorm());
    this->part_norms(i, j) = phi_iij.norm();
    V_ij.release();
    V_ji.release();
    double normiij = phi_iij.norm();

    // compute x_i += phi_jij
    //    Ex[i].add(i_fac, phi_jij);
    // phi_jij.release();

    // compute x_j += phi_iij
    // Ex[j].add(j_fac, phi_iij);
    // phi_iij.release();
    if (mpi::grand_master())
        std::cout << i << " " << j << " prec " << prec << " time " << (int)((float)timer.elapsed() * 1000) << " ms "
                  << " mult1:" << (int)((float)timermult.elapsed() * 1000)
                  << " dens:" << (int)((float)timerd.elapsed() * 1000)
                  << " Pot:" << (int)((float)timerV.elapsed() * 1000)
                  << " mult2:" << (int)((float)timermult2.elapsed() * 1000) << " "
                  << (int)((float)timermult3.elapsed() * 1000) << " Nnodes: " << N0 << " " << N1 << " " << N2 << " "
                  << N3 << " " << N4 << " norms " << norma << " " << normV << " " << normjij << "  " << normiij
                  << std::endl;
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
 * then applies the Poisson operator, and multiplies the result by phi_k.
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
    phi_out_kij.free(NUMBER::Total);
    phi_out_kij = phi_k.paramCopy(); // can be different?

    double i_fac = getSpinFactor(phi_i, phi_j);
    double j_fac = getSpinFactor(phi_j, phi_i);

    double thrs = mrcpp::MachineZero;
    if (std::abs(i_fac) < thrs or std::abs(j_fac) < thrs) return;

    // compute phi_ij = phi_i * phi_j^dag
    Timer timermult;
    Orbital phi_ij = phi_i.paramCopy();
    Orbital phi_ij_tmp = phi_i.paramCopy();
    phi_ij.alloc(NUMBER::Real);
    phi_ij_tmp.alloc(NUMBER::Real);
    if (phi_i.hasImag() or phi_j.hasImag()) {
        phi_ij.alloc(NUMBER::Imag);
        phi_ij_tmp.alloc(NUMBER::Imag);
    }
    double precf = prec / 100; // could use other precision locally
    qmfunction::multiply(phi_ij, phi_i, phi_j, -1.0);
    /*    if (phi_i.hasReal() and phi_j.hasReal() ) {
        mrcpp::multiply(precf, phi_ij.real(), 1.0, phi_i.real(), phi_j.real(), -1, true);
    }
    if (phi_i.hasImag() and phi_j.hasImag() ) { // multiply by +1.0 for complex conjugate and i*i
        mrcpp::multiply(precf, phi_ij_tmp.real(), 1.0, phi_i.real(), phi_j.real(), -1, true);
        phi_ij.real().add(1.0,phi_ij_tmp.real());
    }
    if (phi_i.hasReal() and phi_j.hasImag() ) { // multiply by -1.0 for complex conjugate of j
        mrcpp::multiply(precf, phi_ij.imag(), -1.0, phi_i.real(), phi_j.imag(), -1, true);
    }
    if (phi_i.hasImag() and phi_j.hasReal() ) { // multiply by 1.0 for complex conjugate of j
        mrcpp::multiply(precf, phi_ij_tmp.imag(), 1.0, phi_i.imag(), phi_j.real(), -1, true);
        phi_ij.imag().add(1.0,phi_ij_tmp.real());
        }*/

    double norma = phi_ij.norm();
    int N0 = phi_ij.getNNodes(NUMBER::Total);
    timermult.stop();

    // if the product is smaller than the precision, the result is expected to be negligible
    if (phi_ij.norm() < prec) {
        phi_ij.release();
        phi_ij_tmp.release();
        return;
    }

    Timer timerd;
    Orbital phi_opt; // phi_opt is used to optimize the Poisson application
    qmfunction::deep_copy(phi_opt, phi_k);
    if (phi_out_jij != nullptr) {
        // compute abs(phi_k) + abs(phi_j) , to be used to optimize the target grid of Poisson application
        phi_opt.absadd(1.0, phi_j);
    }

    // Sum the imag and real part in the real part
    // We only use the sum, because the result of Poisson application is to be multiplied by both
    // the real and imag part of phi_k (and we want to apply Poisson only once)
    if (phi_opt.hasImag()) phi_opt.real().absadd(1.0, phi_opt.imag());

    // Calculate the maximum of the norms between each node and its descendants
    phi_opt.real().makeMaxSquareNorms();

    int N1 = phi_opt.getNNodes(NUMBER::Total);
    timerd.stop();

    // compute V_ij = P[phi_ij]
    Orbital V_ij = phi_k.paramCopy();
    Timer timerV;
    precf = prec * 1; // use other precision locally for Poisson
    mrcpp::multiply(precf, phi_ij.real(), 1.0, phi_i.real(), phi_j.real(), -1, true);
    if (phi_ij.hasReal()) {
        V_ij.alloc(NUMBER::Real);
        mrcpp::apply(precf, V_ij.real(), P, phi_ij.real(), phi_opt.real(), -1, true);
    }
    if (phi_ij.hasImag()) {
        V_ij.alloc(NUMBER::Imag);
        mrcpp::apply(precf, V_ij.imag(), P, phi_ij.imag(), phi_opt.real(), -1, true); // NB: phi_opt.real() must be used
    }
    phi_opt.release();
    phi_ij.release();
    phi_ij_tmp.release();
    double normV = V_ij.norm();
    timerV.stop();
    int N2 = V_ij.getNNodes(NUMBER::Total);

    // compute phi_out_kij = phi_k * V_ij
    Timer timermult2;
    precf = prec; // could use other precision locally
    phi_out_kij = phi_k.paramCopy();
    phi_out_kij.alloc(NUMBER::Real);
    // force using at least same grid. Otherwise multiplication is not safe
    mrcpp::copy_grid(phi_out_kij.real(), phi_k.real());
    mrcpp::multiply(precf, phi_out_kij.real(), 1.0, phi_k.real(), V_ij.real(), -1, true);
    if (phi_k.hasImag() and V_ij.hasImag()) {
        Orbital phi_tmp = phi_k.paramCopy();
        phi_tmp.alloc(NUMBER::Real);
        mrcpp::copy_grid(phi_tmp.real(), phi_k.imag());
        mrcpp::multiply(precf, phi_tmp.real(), 1.0, phi_k.imag(), V_ij.imag(), -1, true);
        phi_out_kij.add(-1.0, phi_tmp);
    }
    if (phi_k.hasImag() or V_ij.hasImag()) phi_out_kij.alloc(NUMBER::Imag);
    if (phi_k.hasImag() and V_ij.hasReal()) {
        Orbital phi_tmp = phi_k.paramCopy();
        phi_tmp.alloc(NUMBER::Imag);
        mrcpp::copy_grid(phi_tmp.imag(), phi_k.imag());
        mrcpp::multiply(precf, phi_tmp.imag(), 1.0, phi_k.imag(), V_ij.real(), -1, true);
        phi_out_kij.imag().add(1.0, phi_tmp.imag());
        phi_tmp.free(NUMBER::Total);
    }
    if (phi_k.hasReal() and V_ij.hasImag()) {
        Orbital phi_tmp = phi_k.paramCopy();
        phi_tmp.alloc(NUMBER::Imag);
        mrcpp::copy_grid(phi_tmp.imag(), phi_k.real());
        mrcpp::multiply(precf, phi_tmp.imag(), 1.0, phi_k.real(), V_ij.imag(), -1, true);
        phi_out_kij.imag().add(1.0, phi_tmp.imag());
        phi_tmp.free(NUMBER::Total);
    }
    int N3 = phi_out_kij.getNNodes(NUMBER::Total);

    double normjij = phi_out_kij.norm();
    timermult2.stop();
    Timer timermult3;
    int N4 = 0;
    double normiij = 0.0;
    if (phi_out_jij != nullptr) {
        if (phi_k.hasImag() or V_ij.hasImag()) NOT_IMPLEMENTED_ABORT;
        // compute phi_out_jij = phi_j * V_ij
        *phi_out_jij = phi_j.paramCopy();
        phi_out_jij->alloc(NUMBER::Real);
        mrcpp::copy_grid(phi_out_jij->real(), phi_j.real());
        mrcpp::multiply(precf, phi_out_jij->real(), 1.0, phi_j.real(), V_ij.real(), -1, true);
        normiij = phi_out_jij->norm();
        N4 = phi_out_jij->getNNodes(NUMBER::Total);
    }
    timermult3.stop();
    V_ij.release();

    // compute x_i += phi_jij
    // Ex[i].add(i_fac, phi_jij);
    // phi_jij.release();

    // compute x_j += phi_iij
    // Ex[j].add(j_fac, phi_iij);
    // phi_iij.release();
    if (mpi::grand_master())
        std::cout << " prec " << prec << " time " << (int)((float)timer.elapsed() * 1000) << " ms "
                  << " mult1:" << (int)((float)timermult.elapsed() * 1000)
                  << " dens:" << (int)((float)timerd.elapsed() * 1000)
                  << " Pot:" << (int)((float)timerV.elapsed() * 1000)
                  << " mult2:" << (int)((float)timermult2.elapsed() * 1000) << " "
                  << (int)((float)timermult3.elapsed() * 1000) << " Nnodes: " << N0 << " " << N1 << " " << N2 << " "
                  << N3 << " " << N4 << " norms " << norma << " " << normV << " " << normjij << "  " << normiij
                  << std::endl;
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

>>>>>>> working Poisson, uncleaned
/** @brief scale the relative precision based on norm
 *
 * The internal norms are saved between SCF iterations so that they can
 * be used to estimate the size of the different contributions to the total
 * exchange. The relative precision of the Poisson terms is scaled to give a
 * consistent _absolute_ pecision in the final output.
 */
double ExchangePotential::getScaledPrecision(int i, int j) const {
    double scaled_prec = this->apply_prec;
    if (this->screen) {
        double tNorm = this->tot_norms(i);
        double pNorm = std::max(this->part_norms(i, j), this->part_norms(j, i));
        if (tNorm > 0.0) scaled_prec *= tNorm / pNorm;
    }
    return scaled_prec;
}

<<<<<<< HEAD
=======
/** @brief determines the exchange factor to be used in the calculation of the exact exchange
 *
 * @param [in] orb input orbital to which K is applied
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
    if (phi_j.spin() == SPIN::Paired)
        out = 1.0;
    else if (phi_i.spin() == SPIN::Paired)
        out = 0.5;
    else if (phi_i.spin() == phi_j.spin())
        out = 1.0;
    return out;
}

} // namespace mrchem
