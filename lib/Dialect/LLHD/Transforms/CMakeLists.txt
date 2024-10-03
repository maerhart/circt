add_circt_dialect_library(CIRCTLLHDTransforms
  BlockArgumentToMux.cpp
  DesequentializationPass.cpp
  EarlyCodeMotionPass.cpp
  FunctionEliminationPass.cpp
  MemoryToBlockArgumentPass.cpp
  ProcessLoweringPass.cpp
  ResolveDynamicSignalAliasesPass.cpp
  Sig2RegPass.cpp
  TemporalCodeMotionPass.cpp
  TemporalRegions.cpp
  UnrollPass.cpp
  LLHDTransformsUtil.cpp

  DEPENDS
  CIRCTLLHDTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  CIRCTLLHD
  MLIRIR
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRTransformUtils
  MLIRControlFlowToSCF
)
