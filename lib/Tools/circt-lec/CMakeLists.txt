add_circt_library(CIRCTLECTransforms
  LowerToLEC.cpp

  DEPENDS
  CIRCTLECTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSMT
  CIRCTHW
  CIRCTComb
  CIRCTCombToSMT

  MLIRIR
  MLIRSupport
  MLIRTransforms
)
