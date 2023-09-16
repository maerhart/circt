add_circt_dialect_library(CIRCTCombTransforms
  LowerComb.cpp
  VectorizationInterfacesImpl.cpp

  DEPENDS
  CIRCTCombTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTArcExternalInterfaces
  CIRCTHW
  CIRCTSV
  CIRCTComb
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
