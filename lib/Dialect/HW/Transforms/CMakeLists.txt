add_circt_dialect_library(CIRCTHWTransforms
  HWPrintInstanceGraph.cpp
  HWSpecialize.cpp
  LowerInterop.cpp
  CreateWrapper.cpp

  DEPENDS
  CIRCTHWTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils


  CIRCTSystemC
  CIRCTHW
  CIRCTComb
  MLIREmitCDialect
  MLIRTransforms
)
