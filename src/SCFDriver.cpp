#include <Eigen/Eigenvalues>
#include <fstream>

#include "MRCPP/MWFunctions"
#include "MRCPP/MWOperators"
#include "MRCPP/Printer"
#include "MRCPP/Timer"

#include "getkw/Getkw.hpp"
#include "getkw/Keyword.hpp"
#include "getkw/Section.hpp"

#include "initial_guess/core.h"
#include "initial_guess/gto.h"
#include "initial_guess/sad.h"
#include "utils/math_utils.h"

#include "qmfunctions/Density.h"
#include "qmfunctions/Orbital.h"

#include "SCFDriver.h"
#include "chemistry/Molecule.h"

#include "qmfunctions/density_utils.h"
#include "qmfunctions/orbital_utils.h"

#include "scf_solver/EnergyOptimizer.h"
#include "scf_solver/KAIN.h"
#include "scf_solver/LinearResponseSolver.h"
#include "scf_solver/OrbitalOptimizer.h"

#include "qmoperators/one_electron/ElectricFieldOperator.h"
#include "qmoperators/one_electron/KineticOperator.h"
#include "qmoperators/one_electron/NuclearOperator.h"
#include "qmoperators/two_electron/CoulombOperator.h"
#include "qmoperators/two_electron/ExchangeOperator.h"
#include "qmoperators/two_electron/FockOperator.h"
#include "qmoperators/two_electron/XCOperator.h"

#include "properties/DipoleMoment.h"
#include "properties/GeometryDerivatives.h"
#include "properties/Magnetizability.h"
#include "properties/Polarizability.h"

#include "qmoperators/one_electron/H_BB_dia.h"
#include "qmoperators/one_electron/H_B_dip.h"
#include "qmoperators/one_electron/H_E_dip.h"
#include "qmoperators/one_electron/H_M_pso.h"
#include "qmoperators/one_electron/NuclearGradientOperator.h"
#include "qmoperators/one_electron/X_rm3.h"

using mrcpp::Printer;
using mrcpp::Timer;
using namespace std;

namespace mrchem {

extern mrcpp::MultiResolutionAnalysis<3> *MRA; // Global MRA

SCFDriver::SCFDriver(Getkw &input) {
    max_scale = MRA->getMaxScale();
    rel_prec = input.get<double>("rel_prec");
    nuc_prec = input.get<double>("nuc_prec");

    gauge = input.getDblVec("MRA.gauge_origin");
    center_of_mass = input.get<bool>("MRA.center_of_mass");
    center_of_charge = input.get<bool>("MRA.center_of_charge");

    diff_kin = input.get<std::string>("Derivatives.kinetic");
    diff_orb = input.get<std::string>("Derivatives.h_orb");
    diff_pso = input.get<std::string>("Derivatives.h_pso");

    calc_scf_energy = input.get<bool>("Properties.scf_energy");
    calc_dipole_moment = input.get<bool>("Properties.dipole_moment");
    calc_quadrupole_moment = input.get<bool>("Properties.quadrupole_moment");
    calc_magnetizability = input.get<bool>("Properties.magnetizability");
    calc_nmr_shielding = input.get<bool>("Properties.nmr_shielding");
    calc_hyperfine_coupling = input.get<bool>("Properties.hyperfine_coupling");
    calc_spin_spin_coupling = input.get<bool>("Properties.spin_spin_coupling");
    calc_polarizability = input.get<bool>("Properties.polarizability");
    calc_hyperpolarizability = input.get<bool>("Properties.hyperpolarizability");
    calc_optical_rotation = input.get<bool>("Properties.optical_rotation");
    calc_geometry_derivatives = input.get<bool>("Properties.geometry_derivatives");

    nmr_perturbation = input.get<std::string>("NMRShielding.perturbation");
    nmr_nucleus_k = input.getIntVec("NMRShielding.nucleus_k");
    hfcc_nucleus_k = input.getIntVec("HyperfineCoupling.nucleus_k");
    sscc_nucleus_k = input.getIntVec("SpinSpinCoupling.nucleus_k");
    sscc_nucleus_l = input.getIntVec("SpinSpinCoupling.nucleus_l");
    pol_velocity = input.get<bool>("Polarizability.velocity");
    pol_frequency = input.getDblVec("Polarizability.frequency");
    optrot_velocity = input.get<bool>("OpticalRotation.velocity");
    optrot_frequency = input.getDblVec("OpticalRotation.frequency");
    optrot_perturbation = input.get<std::string>("OpticalRotation.perturbation");

    mol_charge = input.get<int>("Molecule.charge");
    mol_multiplicity = input.get<int>("Molecule.multiplicity");
    mol_coords = input.getData("Molecule.coords");

    wf_restricted = input.get<bool>("WaveFunction.restricted");
    wf_method = input.get<std::string>("WaveFunction.method");

    if (wf_method == "DFT") {
        dft_spin = input.get<bool>("DFT.spin");
        dft_use_gamma = input.get<bool>("DFT.use_gamma");
        dft_cutoff = input.get<double>("DFT.density_cutoff");
        dft_func_coefs = input.getDblVec("DFT.func_coefs");
        dft_func_names = input.getData("DFT.functionals");
    }

    scf_run = input.get<bool>("SCF.run");
    scf_start = input.get<std::string>("SCF.initial_guess");
    scf_kain = input.get<int>("SCF.kain");
    scf_max_iter = input.get<int>("SCF.max_iter");
    scf_rotation = input.get<int>("SCF.rotation");
    scf_canonical = input.get<bool>("SCF.canonical");
    scf_write_orbitals = input.get<bool>("SCF.write_orbitals");
    scf_orbital_thrs = input.get<double>("SCF.orbital_thrs");
    scf_property_thrs = input.get<double>("SCF.property_thrs");
    scf_lambda_thrs = input.get<double>("SCF.lambda_thrs");
    scf_orbital_prec = input.getDblVec("SCF.orbital_prec");

    kin_free_run = input.get<bool>("KineticFree.run");
    kin_free_max_iter = input.get<int>("KineticFree.max_iter");
    kin_free_canonical = input.get<bool>("KineticFree.canonical");
    kin_free_orb_thrs = input.get<double>("KineticFree.orbital_thrs");
    kin_free_prop_thrs = input.get<double>("KineticFree.property_thrs");

    rsp_run = input.get<bool>("Response.run");
    rsp_start = input.get<std::string>("Response.initial_guess");
    rsp_kain = input.get<int>("Response.kain");
    rsp_max_iter = input.get<int>("Response.max_iter");
    rsp_canonical = input.get<bool>("Response.canonical");
    rsp_write_orbitals = input.get<bool>("Response.write_orbitals");
    rsp_orbital_thrs = input.get<double>("Response.orbital_thrs");
    rsp_property_thrs = input.get<double>("Response.property_thrs");
    rsp_directions = input.getIntVec("Response.directions");
    rsp_orbital_prec = input.getDblVec("Response.orbital_prec");

    ext_electric = input.get<bool>("ExternalField.electric_run");
    ext_magnetic = input.get<bool>("ExternalField.magnetic_run");
    if (ext_electric) {
        std::vector<double> tmp = input.getDblVec("ExternalField.electric_field");
        ext_electric_field[0] = tmp[0];
        ext_electric_field[1] = tmp[1];
        ext_electric_field[2] = tmp[2];
    }
    if (ext_magnetic) {
        std::vector<double> tmp = input.getDblVec("ExternalField.magnetic_field");
        ext_magnetic_field[0] = tmp[0];
        ext_magnetic_field[1] = tmp[1];
        ext_magnetic_field[2] = tmp[2];
    }

    file_start_orbitals = input.get<std::string>("Files.start_orbitals");
    file_final_orbitals = input.get<std::string>("Files.final_orbitals");
    file_start_x_orbs = input.get<std::string>("Files.start_x_orbs");
    file_final_x_orbs = input.get<std::string>("Files.final_x_orbs");
    file_start_y_orbs = input.get<std::string>("Files.start_y_orbs");
    file_final_y_orbs = input.get<std::string>("Files.final_y_orbs");
    file_basis_set = input.get<std::string>("Files.basis_set");
    file_dens_mat_a = input.get<std::string>("Files.dens_mat_a");
    file_dens_mat_b = input.get<std::string>("Files.dens_mat_b");
    file_fock_mat = input.get<std::string>("Files.fock_mat");
    file_energy_vec = input.get<std::string>("Files.energy_vec");
    file_mo_mat_a = input.get<std::string>("Files.mo_mat_a");
    file_mo_mat_b = input.get<std::string>("Files.mo_mat_b");

    r_O[0] = 0.0;
    r_O[1] = 0.0;
    r_O[2] = 0.0;

    kain = 0;
    kain_x = 0;
    kain_y = 0;

    P = 0;
    PH_1 = 0;
    PH_2 = 0;
    ABGV_00 = 0;
    ABGV_55 = 0;

    molecule = 0;
    nuclei = 0;
    phi = 0;

    T = 0;
    V = 0;
    J = 0;
    K = 0;
    XC = 0;
    Vext = 0;
    fock = 0;

    phi_np1 = 0;
    J_np1 = 0;
    K_np1 = 0;
    XC_np1 = 0;
    fock_np1 = 0;

    phi_x = 0;
    phi_y = 0;
    dJ = 0;
    dK = 0;
    dXC = 0;
    d_fock = 0;

    xcfun = 0;

    h_E = 0;
    h_B = 0;
    h_M = 0;
}

bool SCFDriver::sanityCheck() const {
    if (scf_lambda_thrs > 0.0) {
        MSG_ERROR("Recycling of HelmholtzOperators currently disabled");
        return true;
    }
    if (wf_restricted and mol_multiplicity != 1) {
        MSG_ERROR("Restricted open-shell not implemented");
        return false;
    }
    if (calc_quadrupole_moment) {
        MSG_ERROR("Quadrupole moment not implemented");
        return false;
    }
    if (calc_hyperpolarizability) {
        MSG_ERROR("Hyperpolarizability not implemented");
        return false;
    }
    if (calc_optical_rotation) {
        MSG_ERROR("Optical rotation not implemented");
        return false;
    }
    if (calc_nmr_shielding) {
        MSG_ERROR("NMR shielding not implemented");
        return false;
    }
    if (calc_spin_spin_coupling) {
        MSG_ERROR("Spin-spin coupling not implemented");
        return false;
    }
    if (calc_hyperfine_coupling) {
        MSG_ERROR("Hyperfine coupling not implemented");
        return false;
    }
    return true;
}

void SCFDriver::setup() {
    // Setting up molecule
    molecule = new Molecule(mol_coords, mol_charge, mol_multiplicity);
    molecule->printGeometry();
    nuclei = &molecule->getNuclei();

    
    // Setting up empty orbitals
    phi = new OrbitalVector;

    // Defining gauge origin
    const mrcpp::Coord<3> &COM = molecule->getCenterOfMass();
    const mrcpp::Coord<3> &COC = molecule->getCenterOfCharge();
    if (center_of_mass) {
        r_O[0] = COM[0];
        r_O[1] = COM[1];
        r_O[2] = COM[2];
    } else if (center_of_charge) {
        r_O[0] = COC[0];
        r_O[1] = COC[1];
        r_O[2] = COC[2];
    } else {
        r_O[0] = gauge[0];
        r_O[1] = gauge[1];
        r_O[2] = gauge[2];
    }

    // Setting up MW operators
    P = new mrcpp::PoissonOperator(*MRA, rel_prec);
    PH_1 = new mrcpp::PHOperator<3>(*MRA, 1); // first derivative
    PH_2 = new mrcpp::PHOperator<3>(*MRA, 2); // second derivative
    ABGV_00 = new mrcpp::ABGVOperator<3>(*MRA, 0.0, 0.0);
    ABGV_55 = new mrcpp::ABGVOperator<3>(*MRA, 0.5, 0.5);

    // Setting up perturbation operators
    int nNucs = molecule->getNNuclei();
    h_E = new H_E_dip(r_O);
    h_B = new H_B_dip(*(useDerivative(diff_orb)), r_O);
    h_M = new H_M_pso *[nNucs];
    for (int k = 0; k < nNucs; k++) {
        const mrcpp::Coord<3> &r_K = molecule->getNucleus(k).getCoord();
        h_M[k] = new H_M_pso(*(useDerivative(diff_pso)), r_K);
    }

    // Setting up properties
    if (nmr_nucleus_k[0] < 0) {
        nmr_nucleus_k.clear();
        for (int k = 0; k < nNucs; k++) nmr_nucleus_k.push_back(k);
    }
    if (sscc_nucleus_k[0] < 0) {
        sscc_nucleus_k.clear();
        for (int k = 0; k < nNucs; k++) sscc_nucleus_k.push_back(k);
    }
    if (sscc_nucleus_l[0] < 0) {
        sscc_nucleus_l.clear();
        for (int l = 0; l < nNucs; l++) sscc_nucleus_l.push_back(l);
    }
    if (hfcc_nucleus_k[0] < 0) {
        hfcc_nucleus_k.clear();
        for (int k = 0; k < nNucs; k++) hfcc_nucleus_k.push_back(k);
    }

    if (calc_scf_energy) molecule->initSCFEnergy();
    if (calc_dipole_moment) molecule->initDipoleMoment();
    if (calc_quadrupole_moment) molecule->initQuadrupoleMoment();
    if (calc_geometry_derivatives) molecule->initGeometryDerivatives();
    if (calc_magnetizability) {
        molecule->initMagnetizability();
        for (int d = 0; d < 3; d++) {
            if (rsp_directions[d] == 0) continue;
            rsp_calculations.push_back(h_B, 0.0, true, d, "H_B");
        }
    }
    if (calc_nmr_shielding) {
        for (int k = 0; k < nmr_nucleus_k.size(); k++) {
            int K = nmr_nucleus_k[k];
            molecule->initNMRShielding(K);
            if (nmr_perturbation == "B") {
                for (int d = 0; d < 3; d++) {
                    if (rsp_directions[d] == 0) continue;
                    rsp_calculations.push_back(h_B, 0.0, true, d, "H_B");
                }
            } else {
                const mrcpp::Coord<3> &r_K = molecule->getNucleus(K).getCoord();
                for (int d = 0; d < 3; d++) {
                    if (rsp_directions[d] == 0) continue;
                    std::string name = "H_M" + std::to_string(K);
                    rsp_calculations.push_back(h_M[K], 0.0, true, d, name);
                }
            }
        }
    }
    if (calc_hyperfine_coupling) {
        for (int k = 0; k < hfcc_nucleus_k.size(); k++) {
            int K = hfcc_nucleus_k[k];
            molecule->initHyperFineCoupling(K);
        }
    }
    if (calc_spin_spin_coupling) {
        for (int k = 0; k < sscc_nucleus_k.size(); k++) {
            int K = sscc_nucleus_k[k];
            for (int l = 0; l < sscc_nucleus_l.size(); l++) {
                int L = sscc_nucleus_l[l];
                if (K != L) molecule->initSpinSpinCoupling(K, L);
            }
        }
    }
    if (calc_polarizability) {
        for (int i = 0; i < pol_frequency.size(); i++) {
            double omega = pol_frequency[i];
            molecule->initPolarizability(omega);
            for (int d = 0; d < 3; d++) {
                if (rsp_directions[d] == 0) continue;
                rsp_calculations.push_back(h_E, omega, false, d, "H_E");
            }
        }
    }
    if (calc_optical_rotation) {
        for (int i = 0; i < optrot_frequency.size(); i++) {
            double omega = optrot_frequency[i];
            molecule->initOpticalRotation(omega);
            NOT_IMPLEMENTED_ABORT;
        }
    }

    // Setting up SCF
    if (scf_kain > 0) kain = new KAIN(scf_kain);
    if (rsp_kain > 0) kain_x = new KAIN(rsp_kain);
    if (rsp_kain > 0) kain_y = new KAIN(rsp_kain);

    // Setting up Fock operator
    T = new KineticOperator(*(useDerivative(diff_kin)));
    V = new NuclearOperator(*nuclei, nuc_prec);
    // All cases need kinetic energy and nuclear potential
    fock = new FockOperator(T, V);
    // For Hartree, HF and DFT we need the coulomb part
    if (wf_method == "Hartree" or wf_method == "HF" or wf_method == "DFT") {
        J = new CoulombOperator(P, phi);
        fock->setCoulombOperator(J);
    }
    // For HF we need the full HF exchange
    if (wf_method == "HF") {
        K = new ExchangeOperator(P, phi);
        fock->setExchangeOperator(K);
    }
    //For DFT we need the XC operator
    if (wf_method == "DFT") {
        xcfun = setupFunctional(MRDFT::Gradient);
        XC = new XCOperator(xcfun, phi);
        fock->setXCOperator(XC);

        //For hybrid DFT we need a partial HF exchange
        if (xcfun->isHybrid()) {
            K = new ExchangeOperator(P, phi, xcfun->amountEXX());
            fock->setExchangeOperator(K);
        }
    }
    //HACK we need a better way to decide whether to initialize the external potential operator
    if (ext_electric) Vext = new ElectricFieldOperator(ext_electric_field, r_O);
    fock->setExtOperator(Vext);
    fock->build();
}

/** @brief choose the right derivative operator to use
 *
 * param[in] nam of the chosen derivative operator
 *
 * Returns the pointer to the correct derivative operator
 *
 */
mrcpp::DerivativeOperator<3> *SCFDriver::useDerivative(string derivative_name) {
    if (derivative_name == "PH_1") return PH_1;
    if (derivative_name == "ABGV_00") return ABGV_00;
    if (derivative_name == "ABGV_55") return ABGV_55;
    MSG_FATAL("No such derivative operator");
}

void SCFDriver::clear() {
    for (int k = 0; k < molecule->getNNuclei(); k++) {
        if (h_M[k] != 0) delete h_M[k];
    }
    if (h_M != 0) delete[] h_M;
    if (h_B != 0) delete h_B;
    if (h_E != 0) delete h_E;

    if (xcfun != 0) delete xcfun;

    if (fock != 0) delete fock;
    if (Vext != 0) delete Vext;
    if (XC != 0) delete XC;
    if (K != 0) delete K;
    if (J != 0) delete J;
    if (V != 0) delete V;
    if (T != 0) delete T;

    if (phi != 0) delete phi;
    if (molecule != 0) delete molecule;

    if (ABGV_55 != 0) delete ABGV_55;
    if (ABGV_00 != 0) delete ABGV_00;
    if (PH_2 != 0) delete PH_2;
    if (PH_1 != 0) delete PH_1;
    if (P != 0) delete P;

    if (kain != 0) delete kain;
    if (kain_x != 0) delete kain_x;
    if (kain_y != 0) delete kain_y;
}

/** Setup n+1 Fock operator for energy optimization */
void SCFDriver::setup_np1() {
    phi_np1 = new OrbitalVector;
    *phi_np1 = orbital::param_copy(*phi);

    fock_np1 = new FockOperator(T, V);
    // For Hartree, HF and DFT we need the coulomb part
    if (wf_method == "Hartree" or wf_method == "HF" or wf_method == "DFT") {
        J_np1 = new CoulombOperator(P, phi_np1);
        fock_np1->setCoulombOperator(J_np1);
    }
    // For HF we need the full HF exchange
    if (wf_method == "HF") {
        K_np1 = new ExchangeOperator(P, phi);
        fock_np1->setExchangeOperator(K_np1);
    }
    //For DFT we need the XC operator
    if (wf_method == "DFT") {
        xcfun = setupFunctional(MRDFT::Gradient);
        XC_np1 = new XCOperator(xcfun, phi_np1);
        fock_np1->setXCOperator(XC_np1);
        //For hybrid DFT we need a partial HF exchange
        if (xcfun->isHybrid()) {
            K_np1 = new ExchangeOperator(P, phi_np1, xcfun->amountEXX());
            fock_np1->setExchangeOperator(K_np1);
        }
    }
    fock_np1->build();
}

void SCFDriver::clear_np1() {
    if (fock_np1 != 0) delete fock_np1;
    if (XC_np1 != 0) delete XC_np1;
    if (K_np1 != 0) delete K_np1;
    if (J_np1 != 0) delete J_np1;
    if (phi_np1 != 0) delete phi_np1;
}

void SCFDriver::setupInitialGroundState() {
    double prec = scf_orbital_prec[0];
    if (scf_start == "GTO")
        if (wf_restricted)
            *phi = initial_guess::gto::setup(prec, *molecule, file_basis_set, file_mo_mat_a);
        else
            *phi = initial_guess::gto::setup(prec, *molecule, file_basis_set, file_mo_mat_a, file_mo_mat_b);
    else if (scf_start == "CORE_SZ")
        *phi = initial_guess::core::setup(prec, *molecule, wf_restricted, 1);
    else if (scf_start == "CORE_DZ")
        *phi = initial_guess::core::setup(prec, *molecule, wf_restricted, 2);
    else if (scf_start == "CORE_TZ")
        *phi = initial_guess::core::setup(prec, *molecule, wf_restricted, 3);
    else if (scf_start == "CORE_QZ")
        *phi = initial_guess::core::setup(prec, *molecule, wf_restricted, 4);
    else if (scf_start == "SAD_SZ")
        *phi = initial_guess::sad::setup(prec, *molecule, wf_restricted, 1);
    else if (scf_start == "SAD_DZ")
        *phi = initial_guess::sad::setup(prec, *molecule, wf_restricted, 2);
    else if (scf_start == "SAD_TZ")
        *phi = initial_guess::sad::setup(prec, *molecule, wf_restricted, 3);
    else if (scf_start == "SAD_QZ")
        *phi = initial_guess::sad::setup(prec, *molecule, wf_restricted, 4);
    else if (scf_start == "MW")
        *phi = orbital::load_orbitals(file_start_orbitals);
    else
        MSG_FATAL("Invalid initial guess");
    orbital::print(*phi);
}

OrbitalOptimizer *SCFDriver::setupOrbitalOptimizer() {
    OrbitalOptimizer *optimizer = new OrbitalOptimizer(kain);
    optimizer->setMaxIterations(scf_max_iter);
    optimizer->setRotation(scf_rotation);
    optimizer->setCanonical(scf_canonical);
    optimizer->setThreshold(scf_orbital_thrs, scf_property_thrs);
    optimizer->setOrbitalPrec(scf_orbital_prec[0], scf_orbital_prec[1]);

    return optimizer;
}

EnergyOptimizer *SCFDriver::setupEnergyOptimizer() {
    EnergyOptimizer *optimizer = new EnergyOptimizer();
    optimizer->setMaxIterations(kin_free_max_iter);
    optimizer->setCanonical(kin_free_canonical);
    optimizer->setThreshold(kin_free_orb_thrs, kin_free_prop_thrs);
    optimizer->setOrbitalPrec(rel_prec, rel_prec);

    return optimizer;
}

LinearResponseSolver *SCFDriver::setupLinearResponseSolver(bool dynamic) {
    LinearResponseSolver *lrs = 0;
    if (dynamic) {
        lrs = new LinearResponseSolver(kain_x, kain_y);
    } else {
        lrs = new LinearResponseSolver(kain_x);
    }
    lrs->setMaxIterations(rsp_max_iter);
    lrs->setThreshold(rsp_orbital_thrs, rsp_property_thrs);
    lrs->setOrbitalPrec(rsp_orbital_prec[0], rsp_orbital_prec[1]);

    return lrs;
}

void SCFDriver::setupPerturbedOrbitals(const ResponseCalculation &rsp_calc) {
    if (phi == 0) MSG_ERROR("Orbitals not initialized");

    phi_x = new OrbitalVector;
    *phi_x = orbital::param_copy(*phi);
    if (rsp_start == "MW") *phi_x = orbital::load_orbitals(file_start_x_orbs, rsp_calc.getFileSuffix());
    if (rsp_calc.isDynamic()) {
        phi_y = new OrbitalVector;
        *phi_y = orbital::param_copy(*phi);
        if (rsp_start == "MW") *phi_y = orbital::load_orbitals(file_start_y_orbs, rsp_calc.getFileSuffix());
    } else {
        phi_y = phi_x;
    }
}

void SCFDriver::clearPerturbedOrbitals(bool dynamic) {
    if (not dynamic) phi_y = 0;
    if (phi_x != 0) {
        delete phi_x;
        phi_x = 0;
    }
    if (phi_y != 0) {
        delete phi_y;
        phi_y = 0;
    }
}

void SCFDriver::setupPerturbedOperators(const ResponseCalculation &rsp_calc) {
    if (phi == nullptr) MSG_ERROR("Orbitals not initialized");
    if (phi_x == nullptr) MSG_ERROR("X orbitals not initialized");
    if (phi_y == nullptr) MSG_ERROR("Y orbitals not initialized");

    double xFac = 0.0;
    if (wf_method == "HF") {
        xFac = 1.0;
    } else if (wf_method == "DFT") {
        xFac = xcfun->amountEXX();
        xcfun = setupFunctional(MRDFT::Hessian);
        dXC = new XCOperator(xcfun, phi, phi_x, phi_y);
    }
    if (xFac > mrcpp::MachineZero) NOT_IMPLEMENTED_ABORT;

    dJ = new CoulombOperator(P, phi, phi_x, phi_y);

    
    int d = rsp_calc.dir;
    RankOneTensorOperator<3> &dH = *rsp_calc.pert;
    if (rsp_calc.isDynamic()) {
        NOT_IMPLEMENTED_ABORT;
    }

    d_fock = new FockOperator(0, 0, dJ, dK, dXC);
    d_fock->perturbation() += dH[d];
    d_fock->build();
}

void SCFDriver::clearPerturbedOperators() {
    if (d_fock != 0) delete d_fock;
    if (dXC != 0) delete dXC;
    if (dK != 0) delete dK;
    if (dJ != 0) delete dJ;

    d_fock = 0;
    dXC = 0;
    dK = 0;
    dJ = 0;
}

void SCFDriver::run() {
    if (not sanityCheck()) return;

    bool converged = runGroundState();
    if (converged) {
        for (int i = 0; i < rsp_calculations.size(); i++) {
            const ResponseCalculation &rsp_calc = rsp_calculations[i];
            runLinearResponse(rsp_calc);
        }
    }

    printEigenvalues(*phi, F);
    molecule->printGeometry();
    molecule->printProperties();
}

bool SCFDriver::runGroundState() {
    if (phi == 0) MSG_ERROR("Orbitals not initialized");
    if (fock == 0) MSG_ERROR("Fock operator not initialized");
    bool converged = true;

    // Setup initial guess
    setupInitialGroundState();

    // Optimize orbitals
    if (scf_run) {
        OrbitalOptimizer *solver = setupOrbitalOptimizer();
        solver->setup(*fock, *phi, F);
        converged = solver->optimize();
        solver->clear();
        delete solver;
    } else {
        fock->setup(rel_prec);
        F = (*fock)(*phi, *phi);
        fock->clear();
    }

    if (scf_canonical) {
        orbital::diagonalize(rel_prec / 10, *phi, F);
    } else {
        ComplexMatrix U = orbital::localize(rel_prec / 10, *phi);
        F = U * F * U.adjoint();
    }

    // Optimize energy
    if (kin_free_run) {
        setup_np1();

        EnergyOptimizer *solver = setupEnergyOptimizer();
        solver->setup(*fock, *phi, F, *fock_np1, *phi_np1);
        converged = solver->optimize();
        solver->clear();

        clear_np1();
        delete solver;
    }

    if (scf_write_orbitals) orbital::save_orbitals(*phi, file_final_orbitals);

    // Compute requested properties
    if (converged) calcGroundStateProperties();

    return converged;
}

void SCFDriver::runLinearResponse(const ResponseCalculation &rsp_calc) {
    bool dynamic = rsp_calc.isDynamic();
    setupPerturbedOrbitals(rsp_calc);
    setupPerturbedOperators(rsp_calc);

    d_fock->getXCOperator()->setupDensity(rel_prec); //Luca: maybe this is not the best place to do this....
    d_fock->getXCOperator()->setupPotential(rel_prec);

    bool converged = true;
    if (rsp_run) {
        LinearResponseSolver *solver = setupLinearResponseSolver(dynamic);
        solver->setupUnperturbed(rsp_orbital_prec[1], fock, phi, &F);
        solver->setup(d_fock, phi_x);
        converged = solver->optimize(); //LUCA Here is a fock.setup()
        solver->clear();
        solver->clearUnperturbed();
        delete solver;
    }
    if (rsp_write_orbitals) {
        orbital::save_orbitals(*phi_x, file_final_x_orbs, rsp_calc.getFileSuffix());
        if (dynamic) orbital::save_orbitals(*phi_y, file_final_y_orbs, rsp_calc.getFileSuffix());
    }

    // Compute requested properties
    if (converged) calcLinearResponseProperties(rsp_calc);

    clearPerturbedOperators();
    clearPerturbedOrbitals(dynamic);
}

void SCFDriver::calcGroundStateProperties() {
    if (calc_scf_energy) {
        fock->setup(rel_prec);
        Printer::printHeader(0, "Calculating SCF energy");
        Timer timer;
        SCFEnergy &energy = molecule->getSCFEnergy();
        energy = fock->trace(*phi, F);
        timer.stop();
        Printer::printFooter(0, timer, 2);
        fock->clear();
    }
    if (calc_dipole_moment) {
        Printer::printHeader(0, "Calculating dipole moment");
        Timer timer;
        DoubleVector &nuc = molecule->getDipoleMoment().getNuclear();
        DoubleVector &el = molecule->getDipoleMoment().getElectronic();
        H_E_dip mu(r_O);
        mu.setup(rel_prec);
        nuc = mu.trace(*nuclei).real();
        el = mu.trace(*phi).real();
        mu.clear();
        timer.stop();
        Printer::printFooter(0, timer, 2);
    }
    if (calc_geometry_derivatives) {
        Printer::printHeader(0, "Calculating geometry derivatives");
        Timer timer;
        DoubleMatrix &nuc = molecule->getGeometryDerivatives().getNuclear();
        DoubleMatrix &el = molecule->getGeometryDerivatives().getElectronic();
        DoubleVector vecsum = DoubleVector::Zero(3);
        DoubleVector torque = DoubleVector::Zero(3);

        for (int k = 0; k < nuclei->size(); k++) {
            const Nucleus &nuc_k = (*nuclei)[k];
            double Z_k = nuc_k.getCharge();
            const mrcpp::Coord<3> &R_k = nuc_k.getCoord();
            Nuclei nucs;
            nucs.push_back("H", R_k);
            NuclearGradientOperator r_rm3(nuc_k, 1.0e-2);
            r_rm3.setup(1.0e-4);
            for (int l = 0; l < nuclei->size(); l++) {
                if (l == k) continue;
                const Nucleus &nuc_l = (*nuclei)[l];
                double Z_l = nuc_l.getCharge();
                const mrcpp::Coord<3> &R_l = nuc_l.getCoord();
                double r_x = (R_k[0] - R_l[0]);
                double r_y = (R_k[1] - R_l[1]);
                double r_z = (R_k[2] - R_l[2]);
                double R_kl = std::pow(math_utils::calc_distance(R_k, R_l), 3.0);
                ;
                nuc(k, 0) -= Z_k * Z_l * r_x / R_kl;
                nuc(k, 1) -= Z_k * Z_l * r_y / R_kl;
                nuc(k, 2) -= Z_k * Z_l * r_z / R_kl;
            }
            el.row(k) = r_rm3.trace(*phi).real();
            r_rm3.clear();
            vecsum += el.row(k);
            vecsum += nuc.row(k);
            torque[0] += R_k[1] * (el(k, 2) + nuc(k, 2)) - R_k[2] * (el(k, 1) + nuc(k, 1));
            torque[1] += R_k[2] * (el(k, 0) + nuc(k, 0)) - R_k[0] * (el(k, 2) + nuc(k, 2));
            torque[2] += R_k[0] * (el(k, 1) + nuc(k, 1)) - R_k[1] * (el(k, 0) + nuc(k, 0));
        }
        println(0, "nuclear part    ");
        println(0, nuc);
        println(0, "electronic part ");
        println(0, el);
        println(0, "Total force acting on nuclei");
        println(0, vecsum.transpose());
        println(0, "Torque acting on nuclei");
        println(0, torque.transpose());
        timer.stop();
        Printer::printFooter(0, timer, 2);
    }
    if (calc_magnetizability) {
        Printer::printHeader(0, "Calculating diamagnetic magnetizability");
        Timer timer;
        DoubleMatrix &dia = molecule->getMagnetizability().getDiamagnetic();
        H_BB_dia h(r_O);
        h.setup(rel_prec);
        dia = -h.trace(*phi).real();
        h.clear();
        timer.stop();
        Printer::printFooter(0, timer, 2);
    }
    /*
    if (calc_nmr_shielding) {
        Printer::printHeader(0, "Calculating diamagnetic NMR shielding");
        Timer timer;
        for (int k = 0; k < nmr_nucleus_k.size(); k++) {
            int K = nmr_nucleus_k[k];
            NMRShielding &nmr = molecule->getNMRShielding(K);
            MatrixXd &dia = nmr.getDiamagnetic();
            const mrcpp::Coord<3> &r_K = nmr.getNucleus().getCoord();
            H_BM_dia h(r_O, r_K);
            h.setup(rel_prec);
            dia = h.trace(*phi);
            h.clear();
        }
        timer.stop();
        Printer::printFooter(0, timer, 2);
    }
    if (calc_spin_spin_coupling) {
        Printer::printHeader(0, "Calculating diamagnetic spin-spin coupling");
        Timer timer;
        for (int k = 0; k < sscc_nucleus_k.size(); k++) {
            int K = sscc_nucleus_k[k];
            for (int l = 0; l < sscc_nucleus_l.size(); l++) {
                int L = sscc_nucleus_l[l];
                if (K == L) continue;
                SpinSpinCoupling &sscc = molecule->getSpinSpinCoupling(K, L);
                MatrixXd &dia = sscc.getDiamagnetic();
                const mrcpp::Coord<3> &r_K = sscc.getNucleusK().getCoord();
                const mrcpp::Coord<3> &r_L = sscc.getNucleusL().getCoord();
                NOT_IMPLEMENTED_ABORT;
            }
        }
        timer.stop();
        Printer::printFooter(0, timer, 2);
    }
    if (calc_hyperfine_coupling) {
        Printer::printHeader(0, "Calculating HyperFine Coupling Constant");
        Timer timer;

        for (int k = 0; k < hfcc_nucleus_k.size(); k++) {
            int K = hfcc_nucleus_k[k];
            HyperFineCoupling &hfc = molecule->getHyperFineCoupling(K);
            const Nuclei &nucs = molecule->getNuclei();
            const Nucleus &nuc = nucs[K];
            const mrcpp::Coord<3> &r_K = nuc.getCoord();
            NOT_IMPLEMENTED_ABORT;
        }
        timer.stop();
        Printer::printFooter(0, timer, 2);
    }
    */
}

void SCFDriver::calcLinearResponseProperties(const ResponseCalculation &rsp_calc) {
    int j = rsp_calc.dir;
    double freq = rsp_calc.freq;

    if (calc_magnetizability and rsp_calc.pert == h_B) {
        Printer::printHeader(0, "Calculating paramagnetic magnetizability");
        Timer timer;
        DoubleMatrix &para = molecule->getMagnetizability().getParamagnetic();
        h_B->setup(rel_prec);
        para.row(j) = -h_B->trace(*phi, *phi_x, *phi_y).real();
        h_B->clear();
        timer.stop();
        Printer::printFooter(0, timer, 2);
    }

    if (calc_polarizability and rsp_calc.pert == h_E) {
        Printer::printHeader(0, "Calculating polarizability");
        Timer timer;
        DoubleMatrix &tensor = molecule->getPolarizability(freq).get();
        h_E->setup(rel_prec);
        tensor.row(j) = -h_E->trace(*phi, *phi_x, *phi_y).real();
        h_E->clear();
        timer.stop();
        Printer::printFooter(0, timer, 2);
    }
    /*
    if (calc_nmr_shielding) {
        if (nmr_perturbation == "B" and rsp_calc.pert == h_B) {
            Timer timer;
            Printer::printHeader(0, "Calculating paramagnetic NMR shielding ");
            for (int k = 0; k < nmr_nucleus_k.size(); k++) {
                int K = nmr_nucleus_k[k];
                MatrixXd &para = molecule->getNMRShielding(K).getParamagnetic();
                h_M[K]->setup(rel_prec);
                para.row(j) = -h_M[K]->trace(*phi, *phi_x, *phi_y);
                h_M[K]->clear();
            }
            timer.stop();
            Printer::printFooter(0, timer, 2);
        }
        if (nmr_perturbation == "M") {
            for (int k = 0; k < nmr_nucleus_k.size(); k++) {
                int K = nmr_nucleus_k[k];
                if (rsp_calc.pert == h_M[K]) {
                    Timer timer;
                    Printer::printHeader(0, "Calculating paramagnetic NMR shielding");

                    MatrixXd &para = molecule->getNMRShielding(K).getParamagnetic();
                    h_B->setup(rel_prec);
                    para.col(j) = -h_B->trace(*phi, *phi_x, *phi_y);
                    h_B->clear();

                    timer.stop();
                    Printer::printFooter(0, timer, 2);
                }
            }
        }
    }
    if (calc_spin_spin_coupling) {
        Printer::printHeader(0, "Calculating paramagnetic spin-spin coupling");
        Timer timer;
        NOT_IMPLEMENTED_ABORT;
        timer.stop();
        Printer::printFooter(0, timer, 2);
    }
    */
}

void SCFDriver::printEigenvalues(OrbitalVector &Phi, ComplexMatrix &f_mat) {
    int oldprec = Printer::setPrecision(5);
    Printer::printHeader(0, "Fock matrix");
    println(0, f_mat.real());
    Printer::printSeparator(0, '=', 2);

    Printer::printHeader(0, "Orbital energies");
    println(0, "    n  spin  occ                            epsilon  ");
    Printer::printSeparator(0, '-');
    Eigen::SelfAdjointEigenSolver<ComplexMatrix> es(f_mat.cols());
    es.compute(f_mat);

    Printer::setPrecision(15);
    DoubleVector epsilon = es.eigenvalues();
    for (int i = 0; i < epsilon.size(); i++) {
        printout(0, setw(5) << i);
        printout(0, setw(5) << Phi[i].printSpin());
        printout(0, setw(5) << Phi[i].occ());
        printout(0, setw(44) << epsilon(i) << endl);
    }
    Printer::printSeparator(0, '=', 2);
    Printer::setPrecision(oldprec);
}

void SCFDriver::extendRotationMatrix(const OrbitalVector &orbs, ComplexMatrix &O) {
    NOT_IMPLEMENTED_ABORT;
    /*
    int nPaired = orbs.getNPaired();
    int nAlpha  = orbs.getNAlpha();
    int nBeta   = orbs.getNBeta();
    int nCols   = O.cols();

    if (nBeta > nAlpha) {
        MSG_ERROR("Inconsistent orbital set: too many beta orbitals");
    }

    O.conservativeResize(nPaired + nAlpha + nBeta, NoChange);
    O.block(nPaired + nAlpha, 0, nBeta, nCols) = O.block(nPaired, 0, nBeta, nCols);
    */
}

/** @brief Build initial density grid for DFT
 *
 * This will refine the density grid in XCFunctional around each nuclear site
 * of the molecule. Uses the refinement algorithm for Gaussians in MRCPP by
 * placing a narrow Gaussian function on each atom with exponent set to the
 * square of the nuclear charge. This grid will be adaptively refined during
 * the SCF procedure.
 */
void SCFDriver::setupInitialGrid(mrdft::XCFunctional &func, const Molecule &mol) {
    Printer::printHeader(0, "Initialize DFT grid");
    println(0, " Nr  Element                        nNodes          nPoints");
    Printer::printSeparator(0, '-');
    Timer timer;
    const Nuclei &nucs = mol.getNuclei();
    for (int k = 0; k < nucs.size(); k++) {
        func.buildGrid(nucs[k].getCharge(), nucs[k].getCoord().data());
        printout(0, std::setw(3) << k);
        printout(0, std::setw(7) << nucs[k].getElement().getSymbol());
        printout(0, std::setw(32) << func.getNNodes());
        printout(0, std::setw(17) << func.getNPoints() << "\n");
    }
    timer.stop();
    Printer::printFooter(0, timer, 2);
}

/** @brief helper routine to set up the correct parameters in the functional before using it
 *
 * param[in] order the requested order of the derivative (order=1 for SCF)
 *
 */

mrdft::XCFunctional *SCFDriver::setupFunctional(int order) {
    mrdft::XCFunctional *fun = new mrdft::XCFunctional(*MRA, dft_spin);
    for (int i = 0; i < dft_func_names.size(); i++) {
        double f_coef = dft_func_coefs[i];
        const std::string &f_name = dft_func_names[i];
        fun->setFunctional(f_name, f_coef);
    }
    fun->setUseGamma(dft_use_gamma);
    fun->setDensityCutoff(dft_cutoff);
    fun->evalSetup(order);
    //setupInitialGrid(*fun, *molecule);
    return fun;
}

} //namespace mrchem
