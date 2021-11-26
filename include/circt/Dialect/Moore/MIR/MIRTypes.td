//===- MIRTypes.td - Moore MIR type definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types for the Moore MIR dialect. Includable in other dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MIR_MIRTYPES
#define CIRCT_DIALECT_MOORE_MIR_MIRTYPES

include "circt/Dialect/Moore/MIR/MIRDialect.td"

def MooreMIRLValueType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::LValueType>()">,
    "lvalue", "::circt::moore::mir::LValueType">;

def MooreMIRRValueType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::RValueType>()">,
    "rvalue", "::circt::moore::mir::RValueType">;

//===----------------------------------------------------------------------===//
// Integer atom types
//===----------------------------------------------------------------------===//

def MooreMIRByteType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::ByteType>()">,
    "an SystemVerilog byte", "::circt::moore::mir::ByteType">;

def MooreMIRShortIntType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::ShortIntType>()">,
    "an SystemVerilog shortint", "::circt::moore::mir::ShortIntType">;

def MooreMIRIntType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::IntType>()">,
    "an SystemVerilog int", "::circt::moore::mir::IntType">;

def MooreMIRLongIntType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::LongIntType>()">,
    "an SystemVerilog longint", "::circt::moore::mir::LongIntType">;

def MooreMIRIntegerType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::IntegerType>()">,
    "an SystemVerilog integer", "::circt::moore::mir::IntegerType">;

def MooreMIRTimeType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::TimeType>()">,
    "an SystemVerilog time", "::circt::moore::mir::TimeType">;

//===----------------------------------------------------------------------===//
// Integer Vector types
//===----------------------------------------------------------------------===//

def MooreMIRBitType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::BitType>()">,
    "an SystemVerilog bit", "::circt::moore::mir::BitType">;

def MooreMIRLogicType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::LogicType>()">,
    "an SystemVerilog logic", "::circt::moore::mir::LogicType">;

def MooreMIRRegType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::RegType>()">,
    "an SystemVerilog reg", "::circt::moore::mir::RegType">;

//===----------------------------------------------------------------------===//
// Misc types
//===----------------------------------------------------------------------===//

def MooreMIRErrorType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::ErrorType>()">,
    "an SystemVerilog error", "::circt::moore::mir::ErrorType">;

def MooreMIRVoidType : DialectType<MIRDialect,
    CPred<"$_self.isa<moore::mir::VoidType>()">,
    "an SystemVerilog void", "::circt::moore::mir::VoidType">;

#endif // CIRCT_DIALECT_MOORE_MIR_MIRTYPES
