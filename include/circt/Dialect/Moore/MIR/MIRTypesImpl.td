//===- MIRTypesImpl.td - Moore MIR type implementations ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type implementations for the Moore MIR dialect.
//
//===----------------------------------------------------------------------===//

class MooreMIRType<string name> : TypeDef<MIRDialect, name> { }

class SimpleMIRType<string name, string mnemo> : MooreMIRType<name> {
  let summary = "System-Verilog " # mnemo # " type";
  let mnemonic = "mir." # mnemo;
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// LValue / RValue
//===----------------------------------------------------------------------===//

def LValueTypeImpl : MooreMIRType<"LValue"> {
  let summary = "LValue";
  let mnemonic = "mir.lvalue";
  let parameters = (ins "::mlir::Type":$realType);
  let assemblyFormat = "`<` $realType `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$realType), [{
      return $_get(realType.getContext(), realType);
    }]>
  ];
}

def RValueTypeImpl : MooreMIRType<"RValue"> {
  let summary = "RValue";
  let mnemonic = "mir.rvalue";
  let parameters = (ins "::mlir::Type":$realType);
  let assemblyFormat = "`<` $realType `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$realType), [{
      return $_get(realType.getContext(), realType);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Integer atom types
//===----------------------------------------------------------------------===//

def ByteTypeImpl     : SimpleMIRType<"Byte", "byte">;
def ShortIntTypeImpl : SimpleMIRType<"ShortInt", "shortint">;
def IntTypeImpl      : SimpleMIRType<"Int", "int">;
def LongTypeImpl     : SimpleMIRType<"LongInt", "longint">;
def IntegerTypeImpl  : SimpleMIRType<"Integer", "integer">;
def TimeTypeImpl     : SimpleMIRType<"Time", "time">;

//===----------------------------------------------------------------------===//
// Integer Vector types
//===----------------------------------------------------------------------===//

def BitTypeImpl      : SimpleMIRType<"Bit", "bit">;
def LogicTypeImpl    : SimpleMIRType<"Logic", "logic">;
def RegTypeImpl      : SimpleMIRType<"Reg", "reg">;

//===----------------------------------------------------------------------===//
// Misc types
//===----------------------------------------------------------------------===//

def ErrorTypeImpl    : SimpleMIRType<"Error", "error">;
def VoidTypeImpl     : SimpleMIRType<"Void", "void">;

//===----------------------------------------------------------------------===//
// Dimensional types
//===----------------------------------------------------------------------===//

// def PackedTypeImpl : MooreMIRType<"Packed"> {
//   let summary = "System-Verilog packed type";

//   let mnemonic = "packed";
//   let parameters = (ins "::mlir::Type":$elementType,
//                         "::mlir::BoolAttr":$signed,
//                         "::mlir::ArrayAttr":$dimensions)
// }

// def UnpackedTypeImpl : MooreMIRType<"Unpacked"> {
//   let summary = "System-Verilog unpacked type";

//   let mnemonic = "unpacked";
//   let parameters = (ins "::mlir::Type":$elementType,
//                         "::mlir::BoolAttr":$signed,
//                         "::mlir::ArrayAttr":$dimensions)
// }
