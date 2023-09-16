add_circt_dialect_library(CIRCTArcExternalInterfaces
  RuntimeCostEstimateInterfaceImpl.cpp
  VectorizationInterfaces.cpp

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  MLIRIR
  MLIRSCFDialect
)
