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

//===----------------------------------------------------------------------===//
// Misc Operations
//===----------------------------------------------------------------------===//

def ConstantOp : MIROp<"constant", [NoSideEffect]> {
  let summary = "A constant value";

  let arguments = (ins I32Attr:$value);
  let results = (outs MooreRValueType:$result);
  let assemblyFormat = "$value attr-dict `:` custom<RValueType>(type($result))";
}

def VariableDeclOp : MIROp<"vardecl", []> {
  let summary = "A variable declaration";

  let arguments = (ins StrAttr:$name, I32Attr:$init);
  let results = (outs MooreLValueType:$result);
  let assemblyFormat = "$name `=` $init attr-dict `:` custom<LValueType>(type($result))";
}

//===----------------------------------------------------------------------===//
// Bitwise Operations
//===----------------------------------------------------------------------===//

class UnaryBitwiseOp<string mnemonic> : MIROp<mnemonic,
    [NoSideEffect, SameOperandsAndResultType]> {
  let arguments = (ins MooreRValueTypeOf<[MooreIntegralType]>:$arg);
  let results = (outs MooreRValueTypeOf<[MooreIntegralType]>:$result);
  let assemblyFormat = "$arg attr-dict `:` custom<RValueType>(type($result))";
}

class BinaryBitwiseOp<string mnemonic> : MIROp<mnemonic,
    [NoSideEffect, Commutative, SameTypeOperands, SameOperandsAndResultType]> {
  let arguments = (ins MooreRValueTypeOf<[MooreIntegralType]>:$lhs,
                       MooreRValueTypeOf<[MooreIntegralType]>:$rhs);
  let results = (outs MooreRValueTypeOf<[MooreIntegralType]>:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` custom<RValueType>(type($result))";
}

def NotOp : UnaryBitwiseOp<"not">;
def AndOp : BinaryBitwiseOp<"and">;
def OrOp  : BinaryBitwiseOp<"or">;
def XorOp : BinaryBitwiseOp<"xor">;

//===----------------------------------------------------------------------===//
// Reduction Operations
//===----------------------------------------------------------------------===//

class UnaryBitwiseReductionOp<string mnemonic> : MIROp<mnemonic,
    [NoSideEffect, SameOperandsAndResultType]> {
  let arguments = (ins MooreRValueTypeOf<[MooreIntegralType]>:$arg);
  let results = (outs MooreBoolRValueType:$result);
  let assemblyFormat = "$arg attr-dict `:` custom<RValueType>(type($arg))";
}

def AndReduceOp : UnaryBitwiseReductionOp<"and_reduce">;
def OrReduceOp  : UnaryBitwiseReductionOp<"or_reduce">;
def XorReduceOp : UnaryBitwiseReductionOp<"xor_reduce">;

//===----------------------------------------------------------------------===//
// Integer Arithmetic Operations
//===----------------------------------------------------------------------===//

// Do these operations have their own signedness semantics? Isn't it just
// dependent on the input types? The same for the value domain?
class UnaryIntArithOp<string mnemonic> : MIROp<mnemonic, [NoSideEffect, SameOperandsAndResultType]> {
  let arguments = (ins MooreRValueTypeOf<[MooreNumericType]>:$arg);
  let results = (outs MooreRValueTypeOf<[MooreNumericType]>:$result);
  let assemblyFormat = "$arg attr-dict `:` custom<RValueType>(type($result))";

  let extraClassDeclaration = [{
    bool isSigned() { return false; }
    bool isFourValued() { return false; }
  }];
}

class BinaryIntArithOp<string mnemonic, list<OpTrait> traits>
  : MIROp<mnemonic, [NoSideEffect, SameTypeOperands, SameOperandsAndResultType] # traits> {
  let arguments = (ins MooreRValueTypeOf<[MooreNumericType]>:$lhs,
                       MooreRValueTypeOf<[MooreNumericType]>:$rhs);
  let results = (outs MooreRValueType:$result);
  // let assemblyFormat = "( `signed` $sign^ )?  ( `four_valued` $fourValued^ )? $lhs `,` $rhs attr-dict `:` type($result)";
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` custom<RValueType>(type($result))";

  let extraClassDeclaration = [{
    bool isSigned() { return false; }
    bool isFourValued() { return false; }
  }];
}

def NegOp : UnaryIntArithOp<"neg">;
def AddOp : BinaryIntArithOp<"add", [Commutative]>;
def SubOp : BinaryIntArithOp<"sub", []>;
def MulOp : BinaryIntArithOp<"mul", [Commutative]>;
def DivOp : BinaryIntArithOp<"div", []>;
def PowOp : BinaryIntArithOp<"pow", []>;

def ModOp : BinaryIntArithOp<"mod", []> {
  let arguments = (ins MooreRValueTypeOf<[MooreIntegralType]>:$lhs,
                       MooreRValueTypeOf<[MooreIntegralType]>:$rhs);
  let results = (outs MooreRValueType:$result);
}

//===----------------------------------------------------------------------===//
// Integer Comparison Operations
//===----------------------------------------------------------------------===//

class BinaryIntCompareOp<string mnemonic, list<OpTrait> traits>
  : MIROp<mnemonic, [NoSideEffect, SameTypeOperands] # traits> {
  let arguments = (ins MooreRValueTypeOf<[MooreNumericType]>:$lhs,
                       MooreRValueTypeOf<[MooreNumericType]>:$rhs);
  let results = (outs MooreBoolRValueType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` custom<RValueType>(type($lhs))";

  let extraClassDeclaration = [{
    bool isSigned() { return false; }
    bool isFourValued() { return false; }
  }];
}

// There are relational operators (numerical), case equality (any except reals),
// wildcard equality (integral), and logical equality (any).
// How are they lowered from the AST in moore?
def EqOp  : BinaryIntArithOp<"eq", [Commutative]>;
def NeqOp : BinaryIntArithOp<"neq", [Commutative]>;
def LtOp  : BinaryIntArithOp<"lt", []>;
def LeqOp : BinaryIntArithOp<"leq", []>;
def GtOp  : BinaryIntArithOp<"gt", []>;
def GeqOp : BinaryIntArithOp<"geq", []>;

//===----------------------------------------------------------------------===//
// Shift Operations
//===----------------------------------------------------------------------===//

class ShiftOp<string mnemonic> : MIROp<mnemonic, [NoSideEffect,
    TypesMatchWith<"", "value", "result", "$_self">]> {
  let arguments = (ins MooreRValueTypeOf<[MooreIntegralType]>:$value,
                       MooreRValueTypeOf<[MooreIntegralType]>:$amount);
  let results = (outs MooreRValueTypeOf<[MooreIntegralType]>:$result);
  let assemblyFormat = "$value `,` $amount attr-dict `:` custom<RValueType>(type($value)) `,` custom<RValueType>(type($amount))";
}

def ShlL : ShiftOp<"shll">;
def ShrL : ShiftOp<"shrl">;
def ShlA : ShiftOp<"shla">;
def ShrA : ShiftOp<"shra">;

//===----------------------------------------------------------------------===//
// Concat / Replication Operations
//===----------------------------------------------------------------------===//

def ConcatOp : MIROp<"concat", [NoSideEffect]> {
  // lvalues and rvalues
  let arguments = (ins Variadic<MooreRValueTypeOf<[MooreIntegralType]>>:$values);
  let results = (outs MooreRValueTypeOf<[MooreIntegralType]>:$result);
  let assemblyFormat = "$values attr-dict `:` type($values) `->` type($result)";
}

def ReplicateOp : MIROp<"replicate", [NoSideEffect]> {
  // lvalues and rvalues
  let arguments = (ins I32Attr:$count,
                       MooreRValueTypeOf<[MooreIntegralType]>:$value);
  let results = (outs MooreRValueTypeOf<[MooreIntegralType]>:$result);
  let assemblyFormat = "$count `x` $value attr-dict `:` type($value) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Ternary Operation
//===----------------------------------------------------------------------===//

def TernaryOp : MIROp<"ternary", [NoSideEffect, AllTypesMatch<["trueValue", "falseValue", "result"]>]> {
  let arguments = (ins MooreBoolRValueType:$cond,
                       MooreRValueType:$trueValue,
                       MooreRValueType:$falseValue);
  let results = (outs MooreRValueType:$result);
  let assemblyFormat = [{
    $cond `then` $trueValue `else` $falseValue attr-dict `:` custom<RValueType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// Cast Operations
//===----------------------------------------------------------------------===//

class CastOpBase<string mnemonic> : MIROp<mnemonic, [NoSideEffect]> {
  let arguments = (ins MooreRValueType:$arg);
  let results = (outs MooreRValueType:$result);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `->` type($result)";
}

def TransmuteOp : MIROp<"transmute", [NoSideEffect]> {
  // should allow lvalue and rvalue, but input and output has to match
  let arguments = (ins AnyTypeOf<[MooreLValueType, MooreRValueType]>:$arg);
  let results = (outs AnyTypeOf<[MooreLValueType, MooreRValueType]>:$result);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `->` type($result)";
}

def CastToSignedOp   : CastOpBase<"to_signed">;
def CastToUnsignedOp : CastOpBase<"to_unsigned">;
def CastToBoolOp     : CastOpBase<"to_bool">;

//===----------------------------------------------------------------------===//
// Truncate / Extension Operations
//===----------------------------------------------------------------------===//

class WidthAdjustOpBase<string mnemonic> : MIROp<mnemonic, [NoSideEffect]> {
  let arguments = (ins MooreRValueType:$arg);
  let results = (outs MooreRValueType:$result);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `->` type($result)";
}

def TruncateOp : WidthAdjustOpBase<"trunc">;
def ZExtOp     : WidthAdjustOpBase<"zext">;
def SExtOp     : WidthAdjustOpBase<"sext">;

//===----------------------------------------------------------------------===//
// Assignment Operations
//===----------------------------------------------------------------------===//

def AssignGetOldOp : MIROp<"assign_get_old", [
    TypesMatchWith<"", "src", "result", "$_self">, 
    TypesMatchWith<"", "src", "dest", "moore::LValueType::get($_self.cast<moore::RValueType>().getNestedType())">]> {
  let arguments = (ins MooreLValueType:$dest, MooreRValueType:$src);
  let results = (outs MooreRValueType:$result);
  let assemblyFormat = "$dest `=` $src attr-dict `:` custom<RValueType>(type($src))";
}

def AssignGetNewOp : MIROp<"assign_get_new", [
    TypesMatchWith<"", "src", "result", "$_self">,
    TypesMatchWith<"", "src", "dest", "moore::LValueType::get($_self.cast<moore::RValueType>().getNestedType())">]> {
  let arguments = (ins MooreLValueType:$dest, MooreRValueType:$src);
  let results = (outs MooreRValueType:$result);
  let assemblyFormat = "$dest `=` $src attr-dict `:` custom<RValueType>(type($src))";
}

// lvalue and rvalue:
// var ref
// port ref
// interface ref
// locally instantiated interface signal ref
// index
// member

// rvalue only:
// pack string
// unpack string
// string comp
// construct array
// construct struct

// lvalue only:
// destruct array
// destruct struct