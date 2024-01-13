//===- SMTOps.td - SMT dialect operations ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTOPS_TD
#define CIRCT_DIALECT_SMT_SMTOPS_TD

include "circt/Dialect/SMT/SMTDialect.td"
include "circt/Dialect/SMT/SMTAttributes.td"
include "circt/Dialect/SMT/SMTTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SMTOp<string mnemonic, list<Trait> traits = []> :
  Op<SMTDialect, mnemonic, traits>;

class SMTBVOp<string mnemonic, list<Trait> traits = []> :
  SMTOp<"bv." # mnemonic, traits>;

def ConstantOp : SMTBVOp<"constant", [
  Pure,
  ConstantLike,
  FirstAttrDerivedResultType,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "Produce a constant bit-vector";

  let arguments = (ins BitVectorAttr:$value);
  let results = (outs BitVectorType:$result);
  let assemblyFormat = "qualified($value) attr-dict";

  let hasFolder = true;
  let hasVerifier = true;
}

class UnaryBVOp<string mnemonic> : SMTBVOp<mnemonic, [
  Pure,
  SameOperandsAndResultType
]> {
  let arguments = (ins BitVectorType:$input);
  let results = (outs BitVectorType:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($result))";
}

class BinaryBVOp<string mnemonic> : SMTBVOp<mnemonic, [
  Pure,
  SameOperandsAndResultType
]> {
  let arguments = (ins BitVectorType:$lhs, BitVectorType:$rhs);
  let results = (outs BitVectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
}

// Arithmetic
def NegOp  : UnaryBVOp<"neg">;
def AddOp  : BinaryBVOp<"add">;
def SubOp  : BinaryBVOp<"sub">;
def MulOp  : BinaryBVOp<"mul">;
def URemOp : BinaryBVOp<"urem">;
def SRemOp : BinaryBVOp<"srem">;
def UModOp : BinaryBVOp<"umod">;
def SModOp : BinaryBVOp<"smod">;
def ShlOp  : BinaryBVOp<"shl">;
def LShrOp : BinaryBVOp<"lshr">;
def AShrOp : BinaryBVOp<"ashr">;
def UDivOp : BinaryBVOp<"udiv">;
def SDivOp : BinaryBVOp<"sdiv">;

// Bitwise
def BVNotOp : UnaryBVOp<"not">;
def BVAndOp : BinaryBVOp<"and">;
def BVOrOp : BinaryBVOp<"or">;
def BVXOrOp : BinaryBVOp<"xor">;
def BVNAndOp : BinaryBVOp<"nand">;
def BVNOrOp : BinaryBVOp<"nor">;
def BVXNOrOp : BinaryBVOp<"xnor">;

#endif // CIRCT_DIALECT_SMT_SMTOPS_TD
