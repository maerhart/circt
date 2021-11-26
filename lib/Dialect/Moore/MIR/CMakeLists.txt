add_circt_dialect_library(CIRCTMooreMIR
  MIRDialect.cpp
  MIROps.cpp
  MIRTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Moore/MIR

  DEPENDS
  MLIRMIRIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
   )

add_dependencies(circt-headers MLIRMIRIncGen)
