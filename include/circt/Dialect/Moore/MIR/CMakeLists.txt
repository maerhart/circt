add_circt_dialect(MIR moore)
add_circt_doc(MIR -gen-dialect-doc MIR Dialects/Moore/)

set(LLVM_TARGET_DEFINITIONS MIR.td)

mlir_tablegen(MIREnums.h.inc -gen-enum-decls)
mlir_tablegen(MIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMIREnumsIncGen)
add_dependencies(circt-headers MLIRMIREnumsIncGen)

mlir_tablegen(MIRStructs.h.inc -gen-struct-attr-decls)
mlir_tablegen(MIRStructs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRMIRStructsIncGen)
add_dependencies(circt-headers MLIRMIRStructsIncGen)
