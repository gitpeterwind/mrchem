target_sources(mrchem PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/chemistry_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PeriodicTable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Molecule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Cavity.cpp
    )

install(
  FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/periodictable.py
  DESTINATION
    ${PYTHON_SITE_INSTALL_DIR}
  )
