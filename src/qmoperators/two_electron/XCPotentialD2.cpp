#include "MRCPP/MWOperators"
#include "MRCPP/Printer"
#include "MRCPP/Timer"

#include "XCPotential.h"
#include "XCPotentialD2.h"
#include "qmfunctions/Density.h"
#include "qmfunctions/Orbital.h"
#include "qmfunctions/density_utils.h"
#include "qmfunctions/orbital_utils.h"

using mrcpp::ABGVOperator;
using mrcpp::DerivativeOperator;
using mrcpp::FunctionTree;
using mrcpp::Printer;
using mrcpp::Timer;

namespace mrchem {

/** @brief Constructor
 *
 * @param[in] F XCFunctional pointer
 * @param[in] Phi Vector of orbitals
 *
 * Based on the order and spin the correct nr. of potential functions is determined.
 * Then the functional is set up for subsequent calculations, fixing some internals of
 * xcfun when F.evalSetup is invoked.
 */
XCPotentialD2::XCPotentialD2(mrdft::XCFunctional *F, OrbitalVector *Phi, OrbitalVector *X, OrbitalVector *Y)
        : XCPotential(F, Phi)
        , orbitals_x(X)
        , orbitals_y(Y)
        , pertDensity_t(nullptr)
        , pertDensity_a(nullptr)
        , pertDensity_b(nullptr) {}

XCPotentialD2::~XCPotentialD2() {
    mrcpp::clear(this->potentials, true);
    if (this->pertDensity_t != nullptr) MSG_FATAL("Operator not properly cleared");
    if (this->pertDensity_a != nullptr) MSG_FATAL("Operator not properly cleared");
    if (this->pertDensity_b != nullptr) MSG_FATAL("Operator not properly cleared");
}

/** @brief Prepare the operator for application
 * 
 * @param[in] prec Apply precision
 *
 * Sequence of steps required to compute the XC potentials:
 *
 * 1) Compute density
 * 2) Setup xcfun input functions (gradients etc.)
 * 3) Evaluate xcfun
 * 4) Compute XC potential(s) from xcfun output
 *
 */
void XCPotentialD2::setup(double prec) {
    if (isSetup(prec)) return;
    setApplyPrec(prec);
    //setupDensity(prec);
    setupPerturbedDensity(prec);
    //setupPotential(prec);
}

/** @brief Clears all data in the XCPotentialD2 object */
void XCPotentialD2::clear() {
    this->energy = 0.0;
    if (this->pertDensity_t != nullptr) delete this->pertDensity_t;
    if (this->pertDensity_a != nullptr) delete this->pertDensity_a;
    if (this->pertDensity_b != nullptr) delete this->pertDensity_b;
    this->pertDensity_t = nullptr;
    this->pertDensity_a = nullptr;
    this->pertDensity_b = nullptr;
    clearApplyPrec();
}

//LUCA This does not work in the case of a non spin separated functional used for an open-shell system!!
void XCPotentialD2::setupPerturbedDensity(double prec) {
    if (this->orbitals_x == nullptr) MSG_ERROR("Orbitals not initialized");
    if (this->orbitals_y == nullptr) MSG_ERROR("Orbitals not initialized");
    OrbitalVector &Phi = *this->orbitals;
    OrbitalVector &X = *this->orbitals_x;
    OrbitalVector &Y = *this->orbitals_y;

    if (this->functional->isSpinSeparated()) {
        this->pertDensity_a = new Density(false);
        this->pertDensity_b = new Density(false);
        Density &dRho_a = *this->pertDensity_a;
        Density &dRho_b = *this->pertDensity_b;
        Timer time_a;
        density::compute(prec, dRho_a, Phi, X, Y, DENSITY::Alpha);
        time_a.stop();
        Printer::printTree(0, "XC perturbed alpha density", dRho_a.getNNodes(NUMBER::Total), time_a.getWallTime());

        Timer time_b;
        density::compute(prec, dRho_b, Phi, X, Y, DENSITY::Beta);
        time_b.stop();
        Printer::printTree(0, "XC perturbed beta density", dRho_b.getNNodes(NUMBER::Total), time_b.getWallTime());

        // Extend to union grid
        while (mrcpp::refine_grid(dRho_a.real(), dRho_b.real())) {}
        while (mrcpp::refine_grid(dRho_b.real(), dRho_a.real())) {}
    } else {
        this->pertDensity_t = new Density(false);
        Density &dRho_t = *this->pertDensity_t;
        Timer time_t;
        density::compute(prec, dRho_t, Phi, X, Y, DENSITY::Total);
        time_t.stop();
        Printer::printTree(0, "XC perturbed total density", dRho_t.getNNodes(NUMBER::Total), time_t.getWallTime());
    }
}

/** @brief Compute XC potential(s)
 *
 * @param prec Precision used in refinement of density grid
 *
 * This will invoke a sequence of steps in the XCFunctional to compute the final
 * XC potential(s) that define this operator. Assuming the density has alredy been
 * computed:
 *
 * 1) Setup xcfun input functions (gradients etc.)
 * 2) Evaluate xcfun
 * 3) Compute XC energy by integrating energy density
 * 4) Compute XC potential(s) from xcfun output functions
 * 5) Remove excess grid nodes based on precision
 * 6) Add extra grid nodes based on precision
 * 7) Clear internal functions in XCFunctional (density grid is kept)
 *
 */
void XCPotentialD2::setupPotential(double prec) {
    if (this->functional == nullptr) MSG_ERROR("XCFunctional not initialized");
    if (not this->functional->hasDensity()) MSG_ERROR("XC density not initialized");
    if (this->potentials.size() != 0) MSG_ERROR("Potential not properly cleared");

    int inpNodes = this->functional->getNNodes();
    int inpPoints = this->functional->getNPoints();

    this->functional->setup();
    this->functional->evaluate();
    this->potentials = this->functional->calcPotential();
    this->functional->clear();

    int newNodes = this->functional->getNNodes() - inpNodes;
    int newPoints = this->functional->getNPoints() - inpPoints;

    println(0, " XC grid size   " << std::setw(26) << inpNodes << std::setw(17) << inpPoints);
    println(0, " XC grid change " << std::setw(26) << newNodes << std::setw(17) << newPoints);
}

/** @brief Return FunctionTree for the XC spin potential
 *
 * @param[in] type Which spin potential to return (alpha, beta or total)
 */
FunctionTree<3> &XCPotentialD2::getPotential(int orbitalSpin, int densitySpin) {
    int pot_idx = XCPotentialD2::getPotentialIndex(orbitalSpin, densitySpin);
    return mrcpp::get_func(this->potentials, pot_idx);
}

int XCPotentialD2::getPotentialIndex(int orbitalSpin, int densitySpin) {

    int spinFunctional = this->functional->isSpinSeparated() ? 1 : 0;

    //Potential order (spin separated): v_aa, v_ab, v_bb
    //Potential order (spin restricted, total density only): v_rr

    int functional_case = spinFunctional; // 0  1
    functional_case += orbitalSpin << 1;  // 0  2  4  6
    functional_case += densitySpin << 3;  // 0  8 16 24

    //OS Paired Alpha Beta
    //DS Total  Spin  Alpha Beta

    // SF    OS             DS          case   Index
    switch (functional_case) {
        //  0     0 (paired)     0 (total)     0
        case (0):
            return 0;
        //  1     0 (paired)     0 (total)     1    not valid
        //  0     1 (alpha )     0 (total)     2
        case (2):
            return 0;
        //  1     1 (alpha )     0 (total)     3    not valid
        //  0     2 (beta  )     0 (total)     4
        case (4):
            return 0;
        //  1     2 (beta  )     0 (total)     5    not valid
        //  0     3 (unused)     0 (total)     6    not valid
        //  1     3 (unused)     0 (total)     7    not valid
        //  0     0 (paired)     1 (spin )     8    not valid
        //  1     0 (paired)     1 (spin )     9    not valid
        //  0     1 (alpha )     1 (spin )    10    not valid
        //  1     1 (alpha )     1 (spin )    11    not valid
        //  0     2 (beta  )     1 (spin )    12    not valid
        //  1     2 (beta  )     1 (spin )    13    not valid
        //  0     3 (unused)     1 (spin )    14    not valid
        //  1     3 (unused)     1 (spin )    15    not valid
        //  0     0 (paired)     2 (alpha)    16    not implemented
        //  1     0 (paired)     2 (alpha)    17    not implemented
        //  0     1 (alpha )     2 (alpha)    18    not valid
        //  1     1 (alpha )     2 (alpha)    19
        case (19):
            return 0;
        //  0     2 (beta  )     2 (alpha)    20    not valid
        //  1     2 (beta  )     2 (alpha)    21
        case (21):
            return 1;
        //  0     3 (unused)     2 (alpha)    22    not valid
        //  1     3 (unused)     2 (alpha)    23    not valid
        //  0     0 (paired)     3 (beta )    24    not implemented
        //  1     0 (paired)     3 (beta )    25    not implemented
        //  0     1 (alpha )     3 (beta )    26    not valid
        //  1     1 (alpha )     3 (beta )    27
        case (27):
            return 1;
        //  0     2 (beta  )     3 (beta )    28    not valid
        //  1     2 (beta  )     3 (beta )    29
        case (29):
            return 2;
        //  0     3 (unused)     3 (beta )    30    not valid
        //  1     3 (unused)     3 (beta )    31    not valid
        default:
            NOT_IMPLEMENTED_ABORT;
    }

    return -1;
}

/** @brief XCPotentialD2 application
 *
 * @param[in] phi Orbital to which the potential is applied
 *
 * The operator is applied by choosing the correct potential function
 * which is then assigned to the real function part of the operator
 * base-class before the base class function is called.
 */
Orbital XCPotentialD2::apply(Orbital phi) {
    bool spinSeparated = this->functional->isSpinSeparated();
    bool totalDens = (pertDensity_t != nullptr);
    bool alphaDens = (pertDensity_a != nullptr);
    bool betaDens = (pertDensity_b != nullptr);

    if (totalDens and (alphaDens or betaDens))
        MSG_ERROR("Total density and spin separated densities are both available");

    FunctionTree<3> *Vrho = new FunctionTree<3>(*MRA);
    mrcpp::FunctionTreeVector<3> components;
    if (not spinSeparated and totalDens) {
        FunctionTree<3> *component = buildComponent(phi.spin(), DENSITY::Total, pertDensity_t->real());
        components.push_back(std::make_tuple(1.0, component));
    } else if (spinSeparated) {
        if (totalDens) NOT_IMPLEMENTED_ABORT;
        if (alphaDens) {
            FunctionTree<3> *component = buildComponent(phi.spin(), DENSITY::Alpha, pertDensity_a->real());
            components.push_back(std::make_tuple(1.0, component));
        }
        if (betaDens) {
            FunctionTree<3> *component = buildComponent(phi.spin(), DENSITY::Beta, pertDensity_b->real());
            components.push_back(std::make_tuple(1.0, component));
        }
    } else {
        NOT_IMPLEMENTED_ABORT;
    }

    mrcpp::build_grid(*Vrho, components); //LUCA just using "add" results in loss of precision.
    mrcpp::add(-1.0, *Vrho, components);
    this->setReal(Vrho);
    Orbital Vrhophi = QMPotential::apply(phi);
    this->setReal(nullptr);
    delete Vrho;
    mrcpp::clear(components, true);
    return Vrhophi;
}

FunctionTree<3> *XCPotentialD2::buildComponent(int orbital_spin, int density_spin, FunctionTree<3> &pert_dens) {
    FunctionTree<3> *component = nullptr;
    if (this->functional->isLDA()) {
        component = buildComponentLDA(orbital_spin, density_spin, pert_dens);
    } else if (this->functional->useGamma()) {
        component = buildComponentGamma(orbital_spin, density_spin, pert_dens);
    } else {
        component = buildComponentGrad(orbital_spin, density_spin, pert_dens);
    }
    return component;
}

FunctionTree<3> *XCPotentialD2::buildComponentGrad(int orbital_spin, int density_spin, FunctionTree<3> &pert_dens) {
    mrcpp::DerivativeOperator<3> *derivative = new mrcpp::ABGVOperator<3>(*MRA, 0.0, 0.0);
    FunctionTree<3> &rho = this->getDensity(DENSITY::Total);
    mrcpp::FunctionTreeVector<3> densities;
    densities.push_back(std::make_tuple(1.0, &rho));
    densities.push_back(std::make_tuple(1.0, &pert_dens));
    mrcpp::FunctionTreeVector<3> grad_eta = mrcpp::gradient(*derivative, pert_dens);
    mrcpp::FunctionTreeVector<3> d2fdrdg(potentials.begin() + 1, potentials.begin() + 4);
    mrcpp::FunctionTreeVector<3> d2fdgdgx(potentials.begin() + 4, potentials.begin() + 7);
    mrcpp::FunctionTreeVector<3> d2fdgdgy;
    d2fdgdgy.push_back(potentials[5]); //yx --> xy
    d2fdgdgy.push_back(potentials[7]); //yy
    d2fdgdgy.push_back(potentials[8]); //yz
    mrcpp::FunctionTreeVector<3> d2fdgdgz;
    d2fdgdgz.push_back(potentials[6]); //zx --> xz
    d2fdgdgz.push_back(potentials[8]); //zy --> yz
    d2fdgdgz.push_back(potentials[9]); //zz

    mrcpp::FunctionTree<3> *prod1 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod1, densities);
    mrcpp::multiply(-1.0, *prod1, 1.0, mrcpp::get_func(potentials, 0), pert_dens);

    mrcpp::FunctionTree<3> *prod2 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod2, densities);
    mrcpp::dot(-1.0, *prod2, d2fdrdg, grad_eta);

    mrcpp::FunctionTree<3> *prod3 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod3, densities);
    mrcpp::multiply(-1.0, *prod3, 1.0, mrcpp::get_func(potentials, 1), pert_dens);

    mrcpp::FunctionTree<3> *prod4 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod4, densities);
    mrcpp::multiply(-1.0, *prod4, 1.0, mrcpp::get_func(potentials, 2), pert_dens);

    mrcpp::FunctionTree<3> *prod5 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod5, densities);
    mrcpp::multiply(-1.0, *prod5, 1.0, mrcpp::get_func(potentials, 3), pert_dens);

    mrcpp::FunctionTree<3> *prod6 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod6, densities);
    mrcpp::dot(-1.0, *prod6, d2fdgdgx, grad_eta);

    mrcpp::FunctionTree<3> *prod7 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod7, densities);
    mrcpp::dot(-1.0, *prod7, d2fdgdgy, grad_eta);

    mrcpp::FunctionTree<3> *prod8 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod8, densities);
    mrcpp::dot(-1.0, *prod8, d2fdgdgz, grad_eta);
    mrcpp::clear(grad_eta, true);

    mrcpp::FunctionTree<3> *sum_36 = new FunctionTree<3>(*MRA);
    mrcpp::FunctionTree<3> *sum_47 = new FunctionTree<3>(*MRA);
    mrcpp::FunctionTree<3> *sum_58 = new FunctionTree<3>(*MRA);

    mrcpp::FunctionTreeVector<3> temp_div1;
    temp_div1.push_back(std::make_tuple(1.0, prod3));
    temp_div1.push_back(std::make_tuple(1.0, prod4));
    temp_div1.push_back(std::make_tuple(1.0, prod5));
    mrcpp::FunctionTreeVector<3> temp_div2;
    temp_div2.push_back(std::make_tuple(1.0, prod6));
    temp_div2.push_back(std::make_tuple(1.0, prod7));
    temp_div2.push_back(std::make_tuple(1.0, prod8));

    mrcpp::FunctionTree<3> *div1 = new FunctionTree<3>(*MRA);
    mrcpp::divergence(*div1, *derivative, temp_div1);
    mrcpp::clear(temp_div1, true);
    mrcpp::FunctionTree<3> *div2 = new FunctionTree<3>(*MRA);
    mrcpp::divergence(*div2, *derivative, temp_div2);
    mrcpp::clear(temp_div2, true);

    mrcpp::FunctionTreeVector<3> temp_sum;
    temp_sum.push_back(std::make_tuple(1.0, prod1));
    temp_sum.push_back(std::make_tuple(1.0, prod2));
    temp_sum.push_back(std::make_tuple(-1.0, div1));
    temp_sum.push_back(std::make_tuple(-1.0, div2));

    mrcpp::FunctionTree<3> *component = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*component, densities);
    mrcpp::add(-1.0, *component, temp_sum);
    mrcpp::clear(temp_sum, true);

    mrcpp::clear(d2fdrdg, false);
    mrcpp::clear(d2fdgdgx, false);
    mrcpp::clear(d2fdgdgy, false);
    mrcpp::clear(d2fdgdgz, false);
    return component;
}

FunctionTree<3> *XCPotentialD2::buildComponentGamma(int orbital_spin, int density_spin, FunctionTree<3> &pert_dens) {
    mrcpp::DerivativeOperator<3> *derivative = new mrcpp::ABGVOperator<3>(*MRA, 0.0, 0.0);
    FunctionTree<3> &rho = this->getDensity(DENSITY::Total);
    mrcpp::FunctionTreeVector<3> densities;
    densities.push_back(std::make_tuple(1.0, &rho));
    densities.push_back(std::make_tuple(1.0, &pert_dens));
    mrcpp::FunctionTreeVector<3> grad_rho = mrcpp::gradient(*derivative, rho);
    mrcpp::FunctionTreeVector<3> grad_eta = mrcpp::gradient(*derivative, pert_dens);
    mrcpp::FunctionTree<3> *gr_dot_gpr = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*gr_dot_gpr, densities); // All functions shall use the same grid here...
    mrcpp::dot(-1.0, *gr_dot_gpr, grad_rho, grad_eta);

    mrcpp::FunctionTree<3> *prod1 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod1, densities);
    mrcpp::multiply(-1.0, *prod1, 1.0, mrcpp::get_func(potentials, 1), pert_dens);

    mrcpp::FunctionTree<3> *prod2 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod2, densities);
    mrcpp::multiply(-1.0, *prod2, 1.0, mrcpp::get_func(potentials, 2), *gr_dot_gpr);

    mrcpp::FunctionTree<3> *prod3 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod3, densities);
    mrcpp::multiply(-1.0, *prod3, 1.0, mrcpp::get_func(potentials, 2), pert_dens);

    mrcpp::FunctionTree<3> *prod4 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*prod4, densities);
    mrcpp::multiply(-1.0, *prod4, 1.0, mrcpp::get_func(potentials, 3), *gr_dot_gpr);
    delete gr_dot_gpr;

    mrcpp::FunctionTree<3> *div3 = calcGradDotPotDensVec(*prod3, grad_rho);
    delete prod3;
    mrcpp::FunctionTree<3> *div4 = calcGradDotPotDensVec(*prod4, grad_rho);
    delete prod4;
    mrcpp::clear(grad_rho, true);

    mrcpp::FunctionTree<3> *div2 = calcGradDotPotDensVec(mrcpp::get_func(potentials, 0), grad_eta);
    mrcpp::clear(grad_eta, true);

    mrcpp::FunctionTree<3> *sum_42 = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*sum_42, densities);
    mrcpp::add(-1.0, *sum_42, 2.0, *div4, 1.0, *div2);
    delete div2;
    delete div4;

    mrcpp::FunctionTreeVector<3> temp_fun;

    temp_fun.push_back(std::make_tuple(1.0, prod1));
    temp_fun.push_back(std::make_tuple(2.0, prod2));
    temp_fun.push_back(std::make_tuple(-2.0, div3));
    temp_fun.push_back(std::make_tuple(-2.0, sum_42));

    mrcpp::FunctionTree<3> *component = new FunctionTree<3>(*MRA);
    mrcpp::build_grid(*component, densities);
    mrcpp::add(-1.0, *component, temp_fun);

    mrcpp::clear(densities, false);
    mrcpp::clear(temp_fun, true);
    delete derivative;
    return component;
}

//Copied from XCFunctional. Ideally it should end up in mrcpp
FunctionTree<3> *XCPotentialD2::calcGradDotPotDensVec(mrcpp::FunctionTree<3> &V, mrcpp::FunctionTreeVector<3> &rho) {
    mrcpp::DerivativeOperator<3> *derivative = new mrcpp::ABGVOperator<3>(*MRA, 0.0, 0.0);
    mrcpp::FunctionTreeVector<3> vec;
    for (int d = 0; d < rho.size(); d++) {
        mrcpp::FunctionTree<3> &rho_d = mrcpp::get_func(rho, d);
        mrcpp::FunctionTree<3> *Vrho = new FunctionTree<3>(*MRA);
        mrcpp::copy_grid(*Vrho, rho_d);
        mrcpp::multiply(-1.0, *Vrho, 1.0, V, rho_d);
        vec.push_back(std::make_tuple(1.0, Vrho));
    }
    mrcpp::FunctionTree<3> *result = new FunctionTree<3>(*MRA);
    mrcpp::divergence(*result, *derivative, vec);
    mrcpp::clear(vec, true);
    return result;
}

FunctionTree<3> *XCPotentialD2::buildComponentLDA(int orbital_spin, int density_spin, FunctionTree<3> &pert_dens) {
    FunctionTree<3> *tmp = new FunctionTree<3>(*MRA);
    FunctionTree<3> &V = getPotential(orbital_spin, density_spin);
    mrcpp::build_grid(*tmp, V);
    mrcpp::build_grid(*tmp, pert_dens);
    mrcpp::multiply(-1.0, *tmp, 1.0, V, pert_dens);
    return tmp;
}

} // namespace mrchem
