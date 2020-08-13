add_mlir_dialect_library(MLIRLLHDTransforms
  TemporalRegions.cpp
  EarlyCodeMotionPass.cpp
  DNFUtil.cpp
  PassRegistration.cpp
  ProcessLoweringPass.cpp
  TemporalCodeMotionPass.cpp
  FunctionEliminationPass.cpp
  EarlyCodeMotionPass.cpp
  BlockArgumentToSelect.cpp
  MemoryToBlockArgumentPass.cpp
  DesequentializationPass.cpp
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
