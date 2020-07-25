add_mlir_dialect_library(MLIRLLHDTransforms
  DNFUtil.cpp
  PassRegistration.cpp
  ProcessLoweringPass.cpp
  FunctionEliminationPass.cpp
  BlockArgumentToSelect.cpp

  DEPENDS
  MLIRLLHDTransformsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  MLIRLLHD
  MLIRTransformUtils
)
