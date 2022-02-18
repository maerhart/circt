add_circt_dialect_library(CIRCTLLHDTransforms
  TemporalRegions.cpp
  PassRegistration.cpp
  ProcessLoweringPass.cpp
  FunctionEliminationPass.cpp
  MemoryToBlockArgumentPass.cpp
  EarlyCodeMotionPass.cpp
  DNFUtil.cpp
  BlockArgumentToMux.cpp

  DEPENDS
  CIRCTLLHDTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTLLHD
  CIRCTHW
  CIRCTComb
  MLIRIR
  MLIRStandard
  MLIRControlFlow
  MLIRTransformUtils
)
