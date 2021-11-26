//===- MIRExpressions.td - Moore MIR expression ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for Moore MIR expressions.
//
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/InferTypeOpInterface.td"


def ConstantOp : MIROp<"mir.constant", [NoSideEffect]> {
  let summary = "A constant of value 'x'";

  let arguments = (ins I32Attr:$value);
  let results = (outs MooreMIRRValueType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def VariableDeclOp : MIROp<"mir.vardecl", []> {
  let summary = "A variable declaration";

  let arguments = (ins StrAttr:$name, I32Attr:$init);
  let results = (outs MooreMIRLValueType:$result);
  let assemblyFormat = "$name `=` $init attr-dict `:` type($result)";
}
