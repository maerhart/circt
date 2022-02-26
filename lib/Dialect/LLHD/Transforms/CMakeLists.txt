add_circt_dialect_library(CIRCTLLHDTransforms
  TemporalRegions.cpp
  PassRegistration.cpp
  ProcessLoweringPass.cpp
  FunctionEliminationPass.cpp
  MemoryToBlockArgumentPass.cpp
  EarlyCodeMotionPass.cpp
  BlockArgumentToMux.cpp
  DesequentializationPass.cpp
  TemporalCodeMotionPass.cpp

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
