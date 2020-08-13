add_mlir_dialect_library(MLIRLLHDTransforms
  TemporalRegions.cpp
  PassRegistration.cpp
  ProcessLoweringPass.cpp
  FunctionEliminationPass.cpp
  EarlyCodeMotionPass.cpp
  InstanceInliningPass.cpp

  DEPENDS
  MLIRLLHDTransformsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  MLIRLLHD
  MLIRTransformUtils
)
