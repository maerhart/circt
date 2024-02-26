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
include "mlir/Interfaces/ControlFlowInterfaces.td"

class SMTOp<string mnemonic, list<Trait> traits = []> :
  Op<SMTDialect, mnemonic, traits>;

class SMTBVOp<string mnemonic, list<Trait> traits = []> :
  SMTOp<"bv." # mnemonic, traits>;

class SMTArrayOp<string mnemonic, list<Trait> traits = []> :
  SMTOp<"array." # mnemonic, traits>;

class SMTIntOp<string mnemonic, list<Trait> traits = []> :
  SMTOp<"int." # mnemonic, traits>;

def AnySMTType : AnyTypeOf<[BoolType, IntType, BitVectorType, ArrayType, SortType]>;

def BoolConstantOp : SMTOp<"constant", [
  Pure,
  ConstantLike,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = "Produce a constant boolean";

  let arguments = (ins BoolAttr:$value);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$value attr-dict";

  let hasFolder = true;
}

def IntConstantOp : SMTIntOp<"constant", [
  Pure,
  ConstantLike,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = "Produce a constant int";

  let arguments = (ins APIntAttr:$value);
  let results = (outs IntType:$result);

  let hasCustomAssemblyFormat = true;
  let hasFolder = true;
}

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

class VariadicIntOp<string mnemonic> : SMTIntOp<mnemonic, [Pure]> {
  let arguments = (ins Variadic<IntType>:$inputs);
  let results = (outs IntType:$result);
  let assemblyFormat = "$inputs attr-dict";
}

class BinaryIntOp<string mnemonic> : SMTIntOp<mnemonic, [Pure]> {
  let arguments = (ins IntType:$lhs, IntType:$rhs);
  let results = (outs IntType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def IntAddOp : VariadicIntOp<"add">;
def IntMulOp : VariadicIntOp<"mul">;
def IntSubOp : VariadicIntOp<"sub">;
def IntDivOp : BinaryIntOp<"div">;
def IntModOp : BinaryIntOp<"mod">;
def IntRemOp : BinaryIntOp<"rem">;
def IntPowOp : BinaryIntOp<"pow">;

def IntPredicateLT : I64EnumAttrCase<"lt", 0>;
def IntPredicateLE : I64EnumAttrCase<"le", 1>;
def IntPredicateGT : I64EnumAttrCase<"gt", 2>;
def IntPredicateGE : I64EnumAttrCase<"ge", 3>;
let cppNamespace = "circt::smt" in
def IntPredicate : I64EnumAttr<
    "IntPredicate",
    "smt comparison predicate for integers",
    [IntPredicateLT, IntPredicateLE, IntPredicateGT, IntPredicateGE]>;

def IntCmpOp : SMTIntOp<"cmp", [Pure]> {
  let arguments = (ins IntPredicate:$pred, IntType:$lhs, IntType:$rhs);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$pred $lhs `,` $rhs attr-dict";
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
  let results = (outs AnySMTType:$result);
  let assemblyFormat = "$declName attr-dict `:` qualified(type($result))";
}

def DeclareFuncOp : SMTOp<"declare_func", [Pure]> {
  let summary = "declare a function";
  let arguments = (ins StrAttr:$declName, UnitAttr:$fresh);
  let results = (outs SMTFunctionType:$result);
  let assemblyFormat = [{
    $declName (`fresh` $fresh^)? attr-dict `:` qualified(type($result))
  }];
}

def ApplyFuncOp : SMTOp<"apply_func", [
  Pure,
  TypesMatchWith<"summary", "func", "result", "cast<SMTFunctionType>($_self).getRangeType()">,
  RangedTypesMatchWith<"summary", "func", "args", "cast<SMTFunctionType>($_self).getDomainTypes()">
]> {
  let summary = "apply a function";
  let arguments = (ins SMTFunctionType:$func, Variadic<AnySMTType>:$args);
  let results = (outs AnySMTType:$result);
  let assemblyFormat = [{
    $func `(` $args `)` attr-dict `:` qualified(type($func))
  }];
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

def IteOp : SMTOp<"ite", [
  Pure,
  AllTypesMatch<["thenVal", "elseVal", "result"]>
]> {
  let summary = "if-then-else";
  let arguments = (ins BoolType:$cond, AnySMTType:$thenVal, AnySMTType:$elseVal);
  let results = (outs AnySMTType:$result);
  let assemblyFormat = [{
    $cond `,` $thenVal `,` $elseVal attr-dict `:` qualified(type($thenVal))
  }];
}

class UnaryBoolOp<string mnemonic> : SMTOp<mnemonic, [Pure]> {
  let arguments = (ins BoolType:$input);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$input attr-dict";
}

class BinaryBoolOp<string mnemonic> : SMTOp<mnemonic, [Pure]> {
  let arguments = (ins BoolType:$lhs, BoolType:$rhs);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

class VariadicBoolOp<string mnemonic> : SMTOp<mnemonic, [Pure]> {
  let arguments = (ins Variadic<BoolType>:$inputs);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$inputs attr-dict";
}

def NotOp : UnaryBoolOp<"not">;
def AndOp : VariadicBoolOp<"and">;
def OrOp : VariadicBoolOp<"or">;
def XOrOp : BinaryBoolOp<"xor">;
def ImpliesOp : BinaryBoolOp<"implies">;

def ArrayStoreOp : SMTArrayOp<"store", [
  Pure,
  TypesMatchWith<"summary", "array", "index", "cast<ArrayType>($_self).getDomainType()">,
  TypesMatchWith<"summary", "array", "value", "cast<ArrayType>($_self).getRangeType()">,
  AllTypesMatch<["array", "result"]>,
]> {
  let summary = "";

  let arguments = (ins ArrayType:$array, AnySMTType:$index, AnySMTType:$value);
  let results = (outs ArrayType:$result);

  let assemblyFormat = [{
    $array `[` $index `]` `,` $value attr-dict `:` qualified(type($array))
  }];
}

def ArraySelectOp : SMTArrayOp<"select", [
  Pure,
  TypesMatchWith<"summary", "array", "index", "cast<ArrayType>($_self).getDomainType()">,
  TypesMatchWith<"summary", "array", "result", "cast<ArrayType>($_self).getRangeType()">,
]> {
  let summary = "";

  let arguments = (ins ArrayType:$array, AnySMTType:$index);
  let results = (outs AnySMTType:$result);

  let assemblyFormat = [{
    $array `[` $index `]` attr-dict `:` qualified(type($array))
  }];
}

def ArrayBroadcastOp : SMTArrayOp<"broadcast", [
  Pure,
  TypesMatchWith<"summary", "result", "value", "cast<ArrayType>($_self).getRangeType()">,
]> {
  let summary = "lowers to mk_const_array";

  let arguments = (ins AnySMTType:$value);
  let results = (outs ArrayType:$result);

  let assemblyFormat = "$value attr-dict `:` qualified(type($result))";
}

def ArrayDefaultOp : SMTArrayOp<"default", [
  Pure,
  TypesMatchWith<"summary", "array", "result", "cast<ArrayType>($_self).getRangeType()">,
]> {
  let summary = "";

  let arguments = (ins ArrayType:$array);
  let results = (outs AnySMTType:$result);

  let assemblyFormat = "$array attr-dict `:` qualified(type($array))";
}

def YieldOp : SMTOp<"yield", [
  Pure,
  Terminator,
  ReturnLike,
  ParentOneOf<["smt::PatternCreateOp", "smt::ForallOp", "smt::ExistsOp"]>,
]> {
  let arguments = (ins Variadic<AnySMTType>:$values);
  let assemblyFormat = "($values^ `:` qualified(type($values)))? attr-dict";
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];
}

def PatternCreateOp : SMTOp<"pattern_create", [
  Pure,
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"smt::YieldOp">,
]> {
  let regions = (region SizedRegion<1>:$body);
  let results = (outs PatternType:$result);

  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasRegionVerifier = true;
}

class QuantifierOp<string mnemonic> : SMTOp<mnemonic, [
  Pure,
  SingleBlockImplicitTerminator<"smt::YieldOp">,
]> {
  let arguments = (ins Variadic<PatternType>:$patterns,
                       I32Attr:$weight,
                       StrArrayAttr:$boundVarNames);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs BoolType:$result);

  let assemblyFormat = [{
    $boundVarNames `patterns` `(` $patterns `)` `weight` $weight
    attr-dict-with-keyword $body
  }];

  let builders = [
    OpBuilder<(ins
      "TypeRange":$boundVarTypes,
      "ArrayRef<StringRef>":$boundVarNames,
      "function_ref<Value(OpBuilder &, Location, ValueRange)>":$bodyBuilder,
      CArg<"uint32_t", "0">:$weight,
      CArg<"ValueRange", "std::nullopt">:$patterns)>
  ];

  let skipDefaultBuilders = true;
  let hasRegionVerifier = true;
}

def ForallOp : QuantifierOp<"forall">;
def ExistsOp : QuantifierOp<"exists">;

#endif // CIRCT_DIALECT_SMT_SMTOPS_TD
