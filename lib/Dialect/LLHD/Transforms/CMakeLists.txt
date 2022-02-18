add_circt_dialect_library(CIRCTLLHDTransforms
  BlockArgumentToMux.cpp
  EarlyCodeMotionPass.cpp
  FunctionEliminationPass.cpp
  LLHDTransformsUtil.cpp
  MemoryToBlockArgumentPass.cpp
  ProcessLoweringPass.cpp
  TemporalCodeMotionPass.cpp
  TemporalRegions.cpp

  DEPENDS
  CIRCTLLHDTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTLLHD
  MLIRIR
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRTransformUtils
)
