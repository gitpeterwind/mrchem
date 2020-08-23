#include "MRCPP/MWOperators"
#include "MRCPP/Printer"
#include "MRCPP/Timer"

#include "ExchangePotentialD1.h"
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
ExchangePotentialD1::ExchangePotentialD1(PoissonOperator_p P, OrbitalVector_p Phi, bool s)
        : ExchangePotential(P, Phi, s) {}

/** @brief precomputes the exchange potential
 *
 *  @param[in] phi_p input orbital
 *
 * The exchange potential is (pre)computed among the orbitals that define the operator
 */
void ExchangePotentialD1::setupInternal(double prec) {

    if (mpi::bank_size > 0) {
        // use cheaper method using bank
        //             setupInternal_bank(prec);
        setupInternal_bankqueue(prec);
        return;
    }
    setApplyPrec(prec);

    if (this->exchange.size() != 0) MSG_ERROR("Exchange not properly cleared");

    OrbitalVector &Phi = *this->orbitals;
    OrbitalVector &Ex = this->exchange;

    // Diagonal must come first because it's NOT in-place
    Timer timer;
    for (int i = 0; i < Phi.size(); i++) {
        Orbital &phi_i = (*this->orbitals)[i];
        Orbital phi_iii = phi_i.paramCopy();
        if (mpi::my_orb(phi_i)) { calc_i_Int_jk_P(prec, phi_i, phi_i, phi_i, phi_iii); }
        this->exchange.push_back(phi_iii);
    }

    // Off-diagonal
    OrbitalIterator iter(Phi, true); // symmetric iterator
    Orbital ex_rcv;
    while (iter.next(1)) { // one orbital at the time
        if (iter.get_size() > 0) {
            Orbital &phi_i = iter.orbital(0);
            int idx = iter.idx(0); // index of orbital phi_i
            for (int j = 0; j < Phi.size(); j++) {
                if (mpi::my_orb(phi_i) and j <= idx) continue; // compute only i<j for own block
                Orbital &phi_j = (*this->orbitals)[j];
                if (mpi::my_orb(phi_j)) {
                    Orbital phi_jij;
                    Orbital phi_iij;
                    // compute phi_iij and phi_jij in one operation
                    calc_i_Int_jk_P(prec, phi_i, phi_j, phi_i, phi_iij, &phi_jij);
                    double i_fac = getSpinFactor(phi_i, phi_j);
                    double j_fac = getSpinFactor(phi_j, phi_i);
                    Ex[j].add(j_fac, phi_iij);
                    phi_iij.release();
                    Ex[idx].add(i_fac, phi_jij);
                    phi_jij.release();
                }
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
void ExchangePotentialD1::setupInternal_bank(double prec) {
    setApplyPrec(prec);

    if (this->exchange.size() != 0) MSG_ERROR("Exchange not properly cleared");
    auto plevel = Printer::getPrintLevel();

    OrbitalVector &Phi = *this->orbitals;
    OrbitalVector &Ex = this->exchange;

    double precf = prec / std::min(10.0, std::sqrt(1.0 * Phi.size())); // since we sum over orbitals

    // Initialize this->exchange and compute own diagonal elements
    Timer timer;
    for (int i = 0; i < Phi.size(); i++) {
        //      calcInternal(i);
        Orbital &phi_i = (*this->orbitals)[i];
        Orbital phi_iii = phi_i.paramCopy();
        if (mpi::my_orb(phi_i)) { calc_i_Int_jk_P(precf, phi_i, phi_i, phi_i, phi_iii); }
        this->exchange.push_back(phi_iii);
    }

    println(2, " Exchange time diagonal " << (int)((float)timer.elapsed() * 1000) << " ms ");

    // Save all orbitals in Bank, so that they can be accessed asynchronously
    Timer timerS;
    mpi::orb_bank.clear_all(mpi::orb_rank, mpi::comm_orb);
    for (int i = 0; i < Phi.size(); i++) {
        if (not mpi::my_orb(Phi[i])) continue;
        Orbital &Phi_i = (*this->orbitals)[i];
        mpi::orb_bank.put_orb(i, Phi_i);
    }

    timerS.stop();

    bool use_sym = true;
    Orbital ex_rcv;
    int N = Phi.size();
    int foundcount = 0;
    int totsize = 0; // total size (kB) of all tree saved in bank by this process so far (until read)
    int maxSize = 5000 * 1024 * mpi::bank_size /
                  mpi::orb_size; // size allowed to be stored before going to the bank for withdrawals (5000 MB).
    for (int j = 0; j < N; j++) {
        if (not mpi::my_orb(Phi[j])) continue; // compute only own j
        for (int i = 0; i < N; i++) {
            if (i == j) continue;
            // compute only half of matrix (in band where i-j < size/2 %size)
            if (i > j + (N - 1) / 2 and i > j and use_sym) continue;
            if (i > j - (N + 1) / 2 and i < j and use_sym) continue;
            Orbital phi_i;
            int ss = mpi::orb_bank.get_orb(i, phi_i, 1);
            Orbital &phi_j = (*this->orbitals)[j];
            Orbital phi_jij;
            Orbital phi_iij;
            if (mpi::my_orb(phi_j)) {
                if (use_sym) {
                    // compute phi_iij and phi_jij in one operation
                    calc_i_Int_jk_P(precf, phi_i, phi_j, phi_i, phi_iij, &phi_jij);
                } else {
                    // compute only own contributions (phi_jij is not computed)
                    calc_i_Int_jk_P(precf, phi_i, phi_j, phi_i, phi_iij);
                }
                double j_fac = getSpinFactor(phi_j, phi_i);
                Ex[j].add(j_fac, phi_iij);
                phi_iij.free(NUMBER::Total);

                if (phi_jij.getNNodes(NUMBER::Total) > 0) {
                    if (not mpi::my_orb(Phi[i])) {
                        // must store contribution to exchange_i in bank
                        timerS.resume();
                        totsize += phi_jij.real().getSizeNodes();
                        if (phi_jij.norm() > prec) { mpi::orb_bank.put_orb(i + (j + 1) * N, phi_jij); }
                        phi_jij.free(NUMBER::Total);
                        timerS.stop();
                    } else {
                        // must add contribution to exchange_i
                        if (phi_jij.norm() > prec) {
                            double i_fac = getSpinFactor(phi_i, phi_j);
                            Ex[i].add(i_fac, phi_jij);
                        }
                        phi_jij.free(NUMBER::Total);
                    }
                }
            }
            // Periodically the bank is visited and all contributions to own exchange that are stored there
            // and ready to use are fetched, added to exchange (and deleted from bank).
            // This must not be done too often for not waisting time, but often enough so that the bank does not fill
            // too much.
            if (totsize > maxSize) { // rough estimate of how full the bank is
                timerS.resume();
                totsize = 0; // reset counter
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
                println(4,
                        " fetched " << foundcount << " Exchange contributions from bank "
                                    << (int)((float)timerS.elapsed() * 1000) << " ms ");
                timerS.stop();
            }
        }
    }
    println(2, " Time exchanges compute " << (int)((float)timer.elapsed() * 1000) << " ms ");
    mpi::barrier(mpi::comm_orb);
    println(2, " Time exchanges all mpi finished " << (int)((float)timer.elapsed() * 1000) << " ms ");
    timerS.resume();
    for (int j = 0; j < N and use_sym; j++) {  // If symmetri is not used, there is nothing to fetch in bank
        if (not mpi::my_orb(Phi[j])) continue; // fetch only own j
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
    println(2, " fetched in total " << foundcount << " Exchange contributions from bank ");
    timerS.stop();
    println(2, " Time send/rcv exchanges " << (int)((float)timerS.elapsed() * 1000) << " ms ");

    mpi::orb_bank.clear_all(mpi::orb_rank, mpi::comm_orb);

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
void ExchangePotentialD1::setupInternal_bankqueue(double prec) {
    setApplyPrec(prec);

    if (this->exchange.size() != 0) MSG_ERROR("Exchange not properly cleared");
    auto plevel = Printer::getPrintLevel();
    mpi::barrier(mpi::comm_orb); // for testing

    OrbitalVector &Phi = *this->orbitals;
    OrbitalVector &Ex = this->exchange;
    int N = Phi.size();
    double precf = prec / std::min(10.0, std::sqrt(1.0 * N)); // since we sum over orbitals

    // Initialize this->exchange and compute own diagonal elements
    Timer timer;

    for (int i = 0; i < N; i++) {
        //      calcInternal(i);
        Orbital &phi_i = (*this->orbitals)[i];
        Orbital phi_iii = phi_i.paramCopy();
        if (mpi::my_orb(phi_i)) { calc_i_Int_jk_P(precf, phi_i, phi_i, phi_i, phi_iii); }
        this->exchange.push_back(phi_iii);
    }

    println(2, " Exchange time diagonal " << (int)((float)timer.elapsed() * 1000) << " ms ");

    // Save all orbitals in Bank, so that they can be accessed asynchronously
    Timer timerS,t_bankr,t_add,t_task,t_wait,t_int;
    t_bankr.stop();
    t_add.stop();
    t_task.stop();
    t_wait.stop();
    t_int.stop();
    mpi::orb_bank.clear_all(mpi::orb_rank, mpi::comm_orb);
    for (int i = 0; i < N; i++) {
        if (not mpi::my_orb(Phi[i])) continue;
        Orbital &Phi_i = (*this->orbitals)[i];
        mpi::orb_bank.put_orb_n(i, Phi_i, 3);
    }
    timerS.stop();

    // make a set of tasks
    // We use symmetry: each pair (i,j) must be used once only. Only j<i
    // Divide into square blocks, with the diagonal blocks taken at the end (because they are faster to compute)
    int block_size = 6; // NB: block_size*block_size intermediate exchange results are stored temporarily
    int iblocks =  (N + block_size - 1) / block_size;
    int ntasks = ((iblocks+1) * iblocks) / 2;
    std::vector<std::vector<int>> itasks(ntasks); // the i values (orbitals) of each block
    std::vector<std::vector<int>> jtasks(ntasks); // the j values (orbitals) of each block

    int task = 0;
    for (int i = 0; i < N; i+=block_size) {
        for (int j = 0; j < i; j+=block_size) {
	    // save all i,j related to one block
            for (int jj = 0; jj < block_size and j+jj < N; jj++) { // note that j+block_size is alway <N
                int jjj = j + jj;
                if (j+block_size <= N) jjj = j + (jj + mpi::orb_rank)%block_size;
                jtasks[task].push_back(jjj);
             }
            for (int ii = 0; ii < block_size and i+ii < N; ii++) {
                int iii = i + ii;
                if (i+block_size <= N) iii = i + (ii + mpi::orb_rank)%block_size;
                itasks[task].push_back(iii);
            }
            task++;
       }
    }
    // add diagonal blocks:
    for (int i = 0; i < N; i+=block_size) {
        // note that not we store all ii and jj in the block, but must use only half of them
        int j = i;
        for (int jj = j; jj < j+block_size and jj < N; jj++) {
            jtasks[task].push_back(jj);
        }
        for (int ii = i; ii < i+block_size and ii < N; ii++) {
            itasks[task].push_back(ii);
        }
        task++;
    }
    ntasks = task;
    t_task.resume();
    mpi::orb_bank.init_tasks(ntasks, mpi::orb_rank, mpi::comm_orb);
    t_task.stop();
    int foundcount = 0;
    int count = 0;
    ComplexVector coef_vec(N);
    for (int i = 0; i<N; i++) coef_vec(i) = 1.0;
    // one task is for one block with block_size x block_size exchange integrals (except diagonal block which has less)
    // Each integral provides two exchange contributions: iij and jij
    // The contributions are summed up within block, and ready results from other blocks are added sometimes.
    while(true) { // fetch new tasks until all are completed
        int it;
	t_task.resume();
        mpi::orb_bank.get_task(&it);
 	t_task.stop();
        if(it<0)break;
        count++;
        QMFunctionVector ifunc_vec;
        OrbitalVector iorb_vec;
        int nodesize=0;
	// we fetch all required i (but only one j at a time)
        for (int i = 0; i < itasks[it].size(); i++) {
            int iorb = itasks[it][i];
            Orbital phi_i;
	    t_bankr.resume();
            mpi::orb_bank.get_orb_n(iorb, phi_i, 0);
	    t_bankr.stop();
            iorb_vec.push_back(phi_i);
            nodesize+=phi_i.getSizeNodes(NUMBER::Total);
        }
        if(mpi::orb_rank==0)std::cout<<mpi::orb_rank<<" fetched total " <<nodesize/1024  <<" MB for "<<itasks[it].size()<<" orbitals "<<std::endl;
	std::vector<QMFunctionVector> jijfunc_vec(N);
        for (int j = 0; j < jtasks[it].size(); j++) {
            int jorb = jtasks[it][j];
            Orbital phi_j;
	    t_bankr.resume();
            mpi::orb_bank.get_orb_n(jorb, phi_j, 0);
	    t_bankr.stop();
	    QMFunctionVector iijfunc_vec;
            for (int i = 0; i < iorb_vec.size(); i++) {
                if(itasks[it][i] <= jorb) continue; //only j<i is computed
                int iorb = itasks[it][i];
                Orbital &phi_i = iorb_vec[i];
                Orbital phi_jij;
                Orbital phi_iij;
                int jj = 0;
                bool phi_jij_saved = false;
                // compute phi_iij and phi_jij in one operation
                t_int.resume();
                calc_i_Int_jk_P(precf, phi_i, phi_j, phi_i, phi_iij, &phi_jij);
                t_int.stop();
                if(phi_iij.norm() > prec) iijfunc_vec.push_back(phi_iij);
                if(phi_jij.norm() > prec) jijfunc_vec[iorb].push_back(phi_jij);

                if((it/iblocks) % 4 == 3) { // every x column only
                    // include also results which may be ready
                    t_task.resume();
                    // include all Ex[jorb] contributions already stored
                    std::vector<int> jvec = mpi::orb_bank.get_readytasks(jorb, 0);
                    t_task.stop();
                    if(jvec.size() > 3) {
                        for (int id : jvec) {
                            Orbital phi_iij_extra;
                            t_bankr.resume();
                            int ok = mpi::orb_bank.get_orb_n(id, phi_iij_extra, 1);
                            t_bankr.stop();
                            if (ok) {
                                t_task.resume();
                                mpi::orb_bank.del_readytask(id, jorb); // tell the task manager that this task should not be used anymore
                                t_task.stop();
				iijfunc_vec.push_back(phi_iij_extra);
				foundcount++;
                            }
                            if(iijfunc_vec.size() > 1.5 * block_size ) break; // we do not want too long vectors
                        }
                    }
                }
            }
	    // sum all iij contributions: (fixed j sum over i)
	    auto tmp_iij = phi_j.paramCopy();
            t_add.resume();
            qmfunction::linear_combination(tmp_iij, coef_vec, iijfunc_vec, prec);
            tmp_iij.crop(prec);
            t_add.stop();
            if (tmp_iij.norm() > prec) {
	        timerS.resume();
                int iorb = itasks[it][0]; // any iorb would do
                int idiij = N+jorb+N*iorb; // N first indices are reserved for original orbitals;
                mpi::orb_bank.put_orb_n(idiij, tmp_iij, 1);
		timerS.stop();
 		t_task.resume();
		mpi::orb_bank.put_readytask(idiij, jorb);
		t_task.stop();
            }
            tmp_iij.free(NUMBER::Total);
            for (int jj = 0; jj<iijfunc_vec.size(); jj++) iijfunc_vec[jj].free(NUMBER::Total);
        }

	// sum all contributions jij (fixed i sum over j)
	for (int i = 0; i < iorb_vec.size(); i++) {
            // include all Ex[iorb] contributions already stored
	    t_task.resume();
            int iorb = itasks[it][i];
	    std::vector<int> jvec = mpi::orb_bank.get_readytasks(iorb, 0);
	    t_task.stop();
	    if(jvec.size() > 3) {
	        for (int id : jvec) {
		    Orbital phi_jij_extra;
		    t_bankr.resume();
		    int ok = mpi::orb_bank.get_orb_n(id, phi_jij_extra, 1);
		    t_bankr.stop();
		    if (ok) {
 		        t_task.resume();
			mpi::orb_bank.del_readytask(id, iorb); // tell the task manager that this task should not be used anymore
			t_task.stop();
			jijfunc_vec[iorb].push_back(phi_jij_extra);
			foundcount++;
		    }
		    if(jijfunc_vec[iorb].size() > 1.5 * block_size ) break; // we do not want too long vectors
		}
	    }
            Orbital &phi_i = iorb_vec[i];
	    auto tmp_jij = phi_i.paramCopy();
	    t_add.resume();
	    qmfunction::linear_combination(tmp_jij, coef_vec, jijfunc_vec[iorb], prec);
	    tmp_jij.crop(prec);
	    t_add.stop();
	    if (tmp_jij.norm() > prec) {
  	        timerS.resume();
                int jorb = jtasks[it][0];//any jorb from task would do
                //The diagonal blocks can have (iorb,jtasks[it][0])=(itasks[it][0],jorb), therefore N*N to distinguish them
                int idjij = N+iorb+N*jorb+N*N; // N first indices are reserved for original orbitals;
		mpi::orb_bank.put_orb_n(idjij, tmp_jij, 1);
		timerS.stop();
		t_task.resume();
		mpi::orb_bank.put_readytask(idjij, iorb);
		t_task.stop();
	    }
	    tmp_jij.free(NUMBER::Total);
	    for (int jj = 0; jj<jijfunc_vec[iorb].size(); jj++) jijfunc_vec[iorb][jj].free(NUMBER::Total);
	}
    }
    println(2, " Time exchanges compute " << (int)((float)timer.elapsed() * 1000) << " ms. "<<" Treated "<<count<<" tasks"<<" blocksize "<<block_size);
    mpi::barrier(mpi::comm_orb);
    println(2, " Time exchanges compute all " << (int)((float)timer.elapsed() * 1000) << " ms ");
    // sum all contributions that may not yet be summed, and put them into Ex[j]
    for (int j = 0; j < N ; j++) {  //
        if (not mpi::my_orb(Phi[j])) continue; // fetch only own j
	t_task.resume();
	// include all Ex[jorb] contributions already stored
	std::vector<int> jvec = mpi::orb_bank.get_readytasks(j, 0);
        QMFunctionVector iijfunc_vec;
	t_task.stop();
 	for (int id : jvec) {
	    Orbital phi_iij;
	    t_bankr.resume();
	    int ok = mpi::orb_bank.get_orb_n(id, phi_iij, 1);
	    t_bankr.stop();
	    if (ok) {
	        t_task.resume();
		mpi::orb_bank.del_readytask(id, j); // tell the task manager that this task should not be used anymore
		t_task.stop();
	        iijfunc_vec.push_back(phi_iij);
		foundcount++;
	    } else {
                std::cout<<mpi::orb_rank<<"did not find exchange contributions for  " << j<<" expected "<<jvec.size()<<" id "<<id<<std::endl;
                MSG_ABORT("did not find contribution for my exchange");
            }
	}
	// sum all found contributions:
	t_add.resume();
        auto tmp_j = Ex[j].paramCopy();
	qmfunction::linear_combination(tmp_j, coef_vec, iijfunc_vec, prec);
        Ex[j].add(1.0, tmp_j);
	Ex[j].crop(prec);
	t_add.stop();
	for (int jj = 0; jj<iijfunc_vec.size(); jj++) iijfunc_vec[jj].free(NUMBER::Total);
   }
    println(2, " fetched in total " << foundcount << " Exchange contributions from bank.  Total time "<<(int)((float)timer.elapsed() * 1000) << " ms ");

    println(2, " Time integrate " << (int)((float)t_int.elapsed() * 1000) << " ms. Time write " << (int)((float)timerS.elapsed() * 1000) << " ms. Time read "<< (int)((float)t_bankr.elapsed() * 1000) << " ms. Time add "<<(int)((float)t_add.elapsed() * 1000));

    mpi::orb_bank.clear_all(mpi::orb_rank, mpi::comm_orb);

    auto n = orbital::get_n_nodes(Ex);
    auto m = orbital::get_size_nodes(Ex);
    auto t = timer.elapsed();
    mrcpp::print::tree(2, "Hartree-Fock exchange", n, m, t);
}

/** @brief Test if a given contribution has been precomputed
 *
 * @param[in] phi_p orbital for which the check is performed
 *
 * If the given contribution has been precomputed, it is simply copied,
 * without additional recalculation.
 */
int ExchangePotentialD1::testPreComputed(Orbital phi_p) const {
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

/** @brief Computes the exchange potential on the fly
 *
 *  \param[in] inp input orbital
 *
 * The exchange potential is computed and applied on the fly to the given orbital.
 * The orbitals must have been previously stored in bank.
 */
Orbital ExchangePotentialD1::calcExchange(Orbital phi_p) {
    Timer timer;
    double prec = this->apply_prec;
    OrbitalVector &Phi = *this->orbitals;
    mrcpp::PoissonOperator &P = *this->poisson;

    std::vector<ComplexDouble> coef_vec;
    QMFunctionVector func_vec;

    for (int i = 0; i < Phi.size(); i++) {
        Orbital &phi_i = Phi[i];
        if (!mpi::my_orb(phi_i)) mpi::orb_bank.get_orb(i, phi_i, 1);
        double spin_fac = getSpinFactor(phi_i, phi_p);
        if (std::abs(spin_fac) < mrcpp::MachineZero) continue;

        // compute phi_ip = phi_i^dag * phi_p
        Orbital phi_ip = phi_p.paramCopy();
        qmfunction::multiply(phi_ip, phi_i, phi_p, -1.0);

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

        coef_vec.push_back(spin_fac / phi_i.squaredNorm());
        func_vec.push_back(phi_iip);

        if (!mpi::my_orb(phi_i)) phi_i.free(NUMBER::Total);
    }

    // compute ex_p = sum_i c_i*phi_iip
    Orbital ex_p = phi_p.paramCopy();
    Eigen::Map<ComplexVector> coefs(coef_vec.data(), coef_vec.size());
    qmfunction::linear_combination(ex_p, coefs, func_vec, -1.0);
    print_utils::qmfunction(3, "Applied exchange", ex_p, timer);
    return ex_p;
}

/** @brief Computes the diagonal part of the internal exchange potential
 *
 *  \param[in] i orbital index
 *
 * The diagonal term K_ii is computed.
 */
void ExchangePotentialD1::calcInternal(int i) {
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
void ExchangePotentialD1::calcInternal(int i, int j) {
    mrcpp::PoissonOperator &P = *this->poisson;
    Orbital &phi_i = (*this->orbitals)[i];
    Orbital &phi_j = (*this->orbitals)[j];
    OrbitalVector &Phi = *this->orbitals;
    OrbitalVector &Ex = this->exchange;

    if (i == j) MSG_ABORT("Cannot handle diagonal term");
    if (Ex.size() != Phi.size()) MSG_ABORT("Size mismatch");
    if (phi_i.hasImag() or phi_j.hasImag()) MSG_ABORT("Orbitals must be real");

    double spinFactor = getSpinFactor(phi_j, phi_i);

    double thrs = mrcpp::MachineZero;
    if (std::abs(spinFactor) < thrs) { return; }

    // set correctly scaled precision for components ij and ji
    double prec = std::min(getScaledPrecision(i, j), getScaledPrecision(j, i));
    if (prec > 1.0e00) return;     // orbital does not contribute within the threshold
    prec = std::min(prec, 1.0e-1); // very low precision does not work properly

    // compute phi_ij = phi_i^dag * phi_j (dagger NOT used, orbitals must be real!)
    Orbital phi_ij = phi_i.paramCopy();
    qmfunction::multiply(phi_ij, phi_i, phi_j, prec);

    // compute V_ij = P[phi_ij]
    Orbital V_ij = phi_i.paramCopy();
    if (phi_ij.hasReal()) {
        V_ij.alloc(NUMBER::Real);
        mrcpp::apply(prec, V_ij.real(), P, phi_ij.real());
    }
    if (phi_ij.hasImag()) {
        MSG_ABORT("Orbitals must be real");
        V_ij.alloc(NUMBER::Imag);
        mrcpp::apply(prec, V_ij.imag(), P, phi_ij.imag());
    }
    phi_ij.release();

    // compute phi_jij = phi_j * V_ij
    Orbital phi_jij = phi_j.paramCopy();
    qmfunction::multiply(phi_jij, phi_j, V_ij, prec);
    phi_jij.rescale(1.0 / phi_j.squaredNorm());

    // compute x_i += phi_jij
    Ex[i].add(spinFactor, phi_jij);
    phi_jij.release();
}

} // namespace mrchem
