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

def AnySMTType : AnyTypeOf<[BoolType, BitVectorType]>;

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

def ConcatOp : SMTBVOp<"concat", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let arguments = (ins BitVectorType:$lhs, BitVectorType:$rhs);
  let results = (outs BitVectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type(operands))";
}

def ExtractOp : SMTBVOp<"extract", [Pure]> {
  let arguments = (ins I32Attr:$start, BitVectorType:$input);
  let results = (outs BitVectorType:$result);
  let assemblyFormat = [{
    $input `from` $start attr-dict `:` functional-type($input, $result)
  }];
  let hasVerifier = true;
}

def RepeatOp : SMTBVOp<"repeat", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let arguments = (ins I32Attr:$count, BitVectorType:$input);
  let results = (outs BitVectorType:$result);
  let assemblyFormat = [{
    $count `times` $input attr-dict `:` qualified(type($input))
  }];
}

def PredicateSLT : I64EnumAttrCase<"slt", 0>;
def PredicateSLE : I64EnumAttrCase<"sle", 1>;
def PredicateSGT : I64EnumAttrCase<"sgt", 2>;
def PredicateSGE : I64EnumAttrCase<"sge", 3>;
def PredicateULT : I64EnumAttrCase<"ult", 4>;
def PredicateULE : I64EnumAttrCase<"ule", 5>;
def PredicateUGT : I64EnumAttrCase<"ugt", 6>;
def PredicateUGE : I64EnumAttrCase<"uge", 7>;
let cppNamespace = "circt::smt" in
def Predicate : I64EnumAttr<
    "Predicate",
    "smt comparison predicate",
    [PredicateSLT, PredicateSLE, PredicateSGT, PredicateSGE,
     PredicateULT, PredicateULE, PredicateUGT, PredicateUGE]>;

def BVCmpOp : SMTBVOp<"cmp", [Pure, SameTypeOperands]> {
  let arguments = (ins Predicate:$pred, BitVectorType:$lhs, BitVectorType:$rhs);
  let results = (outs BoolType:$result);
  let assemblyFormat = [{
    $pred $lhs `,` $rhs attr-dict `:` qualified(type($lhs))
  }];
}

def EqOp : SMTOp<"eq", [Pure, SameTypeOperands]> {
  let arguments = (ins AnySMTType:$lhs, AnySMTType:$rhs);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($lhs))";
}

def DistinctOp : SMTOp<"distinct", [Pure, SameTypeOperands]> {
  let arguments = (ins Variadic<AnySMTType>:$inputs);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$inputs attr-dict `:` qualified(type($inputs))";
}

def DeclareConstOp : SMTOp<"declare_const", [Pure]> {
  let summary = "declare a constant value";
  let arguments = (ins StrAttr:$declName);
  let results = (outs AnyTypeOf<[BoolType, BitVectorType]>:$result);
  let assemblyFormat = "$declName attr-dict `:` qualified(type($result))";
}

def SolverCreateOp : SMTOp<"solver_create", [Pure]> {
  let summary = "create a solver instance";
  let arguments = (ins StrAttr:$solverName);
  let results = (outs SolverType:$solver);
  let assemblyFormat = "$solverName attr-dict";
}

def AssertOp : SMTOp<"assert", []> {
  let summary = "assert that a boolean expression is true";
  let arguments = (ins SolverType:$solver, BoolType:$input);
  let assemblyFormat = "$solver `,` $input attr-dict";
}

def CheckSatOp : SMTOp<"check_sat", []> {
  let summary = "check if the current set of assertions is satisfiable";
  let description = [{
    Returns
      * -1 if not satisfiable
      * 0  if unknown
      * 1  if satisfiable     
  }];

  let arguments = (ins SolverType:$solver);
  let results = (outs I32:$result);
  let assemblyFormat = "$solver attr-dict";
}

#endif // CIRCT_DIALECT_SMT_SMTOPS_TD
