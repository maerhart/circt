add_circt_conversion_library(CIRCTLowerInterop
  LowerInterop.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSystemC
  CIRCTHW
  CIRCTComb
  MLIREmitCDialect
  MLIRTransforms
)
