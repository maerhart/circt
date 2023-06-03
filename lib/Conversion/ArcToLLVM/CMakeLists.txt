add_circt_pdll_library(CIRCTArcToLLVMConversionPDLLPatternsIncGen
  ArcToLLVMConversion.pdll
  ArcToLLVMConversionPDLLPatterns.h.inc
)

add_circt_conversion_library(CIRCTArcToLLVM
  LowerArcToLLVM.cpp

  DEPENDS
  CIRCTConversionPassIncGen
  CIRCTArcToLLVMConversionPDLLPatternsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTComb
  CIRCTCombToLLVM
  CIRCTHWToLLVM
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRSCFToControlFlow
  MLIRTransforms
)

target_include_directories(CIRCTArcToLLVM
  PRIVATE
  ${CIRCT_LIBRARY_DIR}/Conversion/ArcToLLVM
)
