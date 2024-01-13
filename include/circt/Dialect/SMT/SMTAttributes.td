//===- SMTAttributes.td - Attributes for SMT dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SMT dialect specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTATTRIBUTES_TD
#define CIRCT_DIALECT_SMT_SMTATTRIBUTES_TD

include "circt/Dialect/SMT/SMTDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def BitVectorAttr : AttrDef<SMTDialect, "BitVector", [TypedAttrInterface]> {
  let mnemonic = "bv";
  let parameters = (ins "unsigned":$value,
                        AttributeSelfTypeParameter<"">:$type);

  let assemblyFormat = "`<` $value `>`";
}

#endif // CIRCT_DIALECT_SMT_SMTATTRIBUTES_TD
