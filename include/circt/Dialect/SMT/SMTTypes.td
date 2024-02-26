//===- SMTTypes.td - SMT dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTTYPES_TD
#define CIRCT_DIALECT_SMT_SMTTYPES_TD

include "circt/Dialect/SMT/SMTDialect.td"
include "mlir/IR/AttrTypeBase.td"

class SMTTypeDef<string name> : TypeDef<SMTDialect, name> { }

def BoolType : SMTTypeDef<"Bool"> {
  let mnemonic = "bool";
  let assemblyFormat = "";
}

def IntType : SMTTypeDef<"Integer"> {
  let mnemonic = "int";
  let assemblyFormat = "";
}

def PatternType : SMTTypeDef<"Pattern"> {
  let mnemonic = "pattern";
  let assemblyFormat = "";
}

def BitVectorType : SMTTypeDef<"BitVector"> {
  let mnemonic = "bv";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
}

def SolverType : SMTTypeDef<"Solver"> {
  let mnemonic = "solver";
  let assemblyFormat = "";
}

def ArrayType : SMTTypeDef<"Array"> {
  let mnemonic = "array";
  let parameters = (ins "mlir::Type":$domainType, "mlir::Type":$rangeType);
  let assemblyFormat = "`<` `[` $domainType `->` $rangeType `]` `>`";
}

def SMTFunctionType : SMTTypeDef<"SMTFunction"> {
  let mnemonic = "func";
  let parameters = (ins ArrayRefParameter<"mlir::Type", "domain types">:$domainTypes, "mlir::Type":$rangeType);
  let assemblyFormat = "`<` `(` $domainTypes `)` `->` $rangeType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "llvm::ArrayRef<mlir::Type>":$domainTypes, "mlir::Type":$rangeType), [{
      return $_get(rangeType.getContext(), domainTypes, rangeType);
    }]>
  ];
}

def SortType : SMTTypeDef<"Sort"> {
  let mnemonic = "sort";
  let parameters = (ins "mlir::StringAttr":$identifier, OptionalArrayRefParameter<"mlir::Type", "sort parameters">:$sortParams);
  let assemblyFormat = "`<` $identifier (`[` $sortParams^ `]`)? `>`";

  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$identifier, "llvm::ArrayRef<mlir::Type>":$sortParams), [{
      $_get($_ctxt, mlir::StringAttr::get($_ctxt, identifier), sortParams);
    }]>,
    TypeBuilder<(ins "llvm::StringRef":$identifier), [{
      $_get($_ctxt, mlir::StringAttr::get($_ctxt, identifier), llvm::ArrayRef<mlir::Type>{});
    }]>,
  ];
}

#endif // CIRCT_DIALECT_SMT_SMTTYPES_TD
