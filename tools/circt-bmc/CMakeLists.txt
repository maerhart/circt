add_llvm_tool(circt-bmc
  circt-bmc.cpp
)

target_link_libraries(circt-bmc
  PRIVATE
  CIRCTArcTransforms
  CIRCTLECTransforms
  CIRCTSMTToZ3LLVM
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRExecutionEngine
  MLIRExecutionEngineUtils
  MLIRFuncInlinerExtension
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)

# Correct the runpath when linking shared libraries.
if(BUILD_SHARED_LIBS)
  set_target_properties(circt-bmc PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )
endif()

llvm_update_compile_flags(circt-bmc)
mlir_check_all_link_libraries(circt-bmc)
