add_circt_library(CIRCTSystemCTransforms
  InteropOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIREmitCDialect
  CIRCTHW
  CIRCTSystemC
)
