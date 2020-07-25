add_mlir_dialect_library(MLIRLLHDTransforms
  PassRegistration.cpp
  ProcessLoweringPass.cpp
  TemporalCodeMotionPass.cpp
  FunctionEliminationPass.cpp

  DEPENDS
  MLIRLLHDTransformsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  MLIRLLHD
  MLIRLLHDAnalysis
  MLIRTransformUtils
)
