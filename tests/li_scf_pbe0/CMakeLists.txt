set(_li_scf_pbe0_launcher " ")

if(ENABLE_MPI)
    set(_li_scf_pbe0_launcher "mpirun -np 2")
endif()

add_integration_test(
  NAME "Li_SCF_PBE0"
  LABELS "mrchem;li_scf_pbe0;Li_SCF_PBE0;dft;scf;energy"
  COST 100
  LAUNCH_AGENT ${_li_scf_pbe0_launcher}
  )
