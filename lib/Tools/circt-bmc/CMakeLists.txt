add_circt_library(CIRCTBMCTransforms
  LowerToBMC.cpp

  DEPENDS
  CIRCTBMCTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSMT
  CIRCTHW
  CIRCTSeq
  CIRCTComb
  CIRCTVerif
  CIRCTCombToSMT

  MLIRIR
  MLIRSupport
  MLIRSCFDialect
  MLIRTransforms
)
