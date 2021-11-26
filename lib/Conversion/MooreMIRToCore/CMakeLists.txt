add_circt_conversion_library(CIRCTMooreMIRToCore
  MooreMIRToCore.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTMooreMIR
  CIRCTLLHD
  CIRCTHW
  CIRCTComb
  MLIRTransforms
)
