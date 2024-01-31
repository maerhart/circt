add_circt_library(CIRCTHandshakeToSMT
  HandshakeToSMT.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HandshakeToSMT

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHandshake
  CIRCTSMT
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
  )
