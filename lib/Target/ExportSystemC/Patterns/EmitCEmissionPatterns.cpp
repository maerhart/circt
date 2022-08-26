//===- EmitCEmissionPatterns.cpp - EmitC Dialect Emission Patterns --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the emitc dialect.
//
//===----------------------------------------------------------------------===//

#include "EmitCEmissionPatterns.h"
#include "../EmissionPrinter.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"

using namespace mlir::emitc;
using namespace circt;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit emitc.include operations.
struct IncludeEmitter : OpEmissionPattern<IncludeOp> {
  using OpEmissionPattern::OpEmissionPattern;

  void emitStatement(IncludeOp op, EmissionPrinter &p) override {
    p << "#include " << (op.getIsStandardInclude() ? "<" : "\"")
      << op.getInclude() << (op.getIsStandardInclude() ? ">" : "\"") << "\n";
  }
};

/// Emit emitc.apply operations.
struct ApplyOpEmitter : OpEmissionPattern<ApplyOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<ApplyOp>()) {
      // We would need to check the 'applicableOperator' to select the
      // precedence to return. However, since the dereference and address_of
      // operators have the same precedence, we can omit that (for better
      // performance).
      return Precedence::ADDRESS_OF;
    }
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto applyOp = value.getDefiningOp<ApplyOp>();
    p << applyOp.getApplicableOperator();
    auto emitter = p.getInlinable(applyOp.getOperand());
    if (emitter.getPrecedence() >= Precedence::ADDRESS_OF)
      p << "(";
    emitter.emit();
    if (emitter.getPrecedence() >= Precedence::ADDRESS_OF)
      p << ")";
  }
};

/// Emit emitc.call operations.
struct CallOpEmitter : OpEmissionPattern<CallOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (auto op = value.getDefiningOp<CallOp>()) {
      // TODO: No attribute arguments and template arguments supported for now.
      if (!op.getArgs() && !op.getTemplateArgs())
        return Precedence::FUNCTION_CALL;
    }
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto callOp = value.getDefiningOp<CallOp>();
    p << callOp.getCallee();

    p << "(";
    bool first = true;

    for (Value operand : callOp.getOperands()) {
      if (!first)
        p << ", ";

      p.getInlinable(operand).emit();
      first = false;
    }

    p << ")";
  }
};

/// Emit emitc.cast operations.
struct CastOpEmitter : OpEmissionPattern<CastOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<CastOp>())
      return Precedence::CAST;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto castOp = value.getDefiningOp<CastOp>();
    p << "(";
    p.emitType(castOp.getDest().getType());
    p << ") ";
    auto emitter = p.getInlinable(castOp.getSource());
    if (emitter.getPrecedence() >= Precedence::CAST)
      p << "(";
    emitter.emit();
    if (emitter.getPrecedence() >= Precedence::CAST)
      p << ")";
  }
};

/// Emit emitc.constant operations.
struct ConstantOpEmitter : OpEmissionPattern<ConstantOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<ConstantOp>())
      return Precedence::LIT;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto constantOp = value.getDefiningOp<ConstantOp>();
    // TODO: implement attribute emission patterns and inline here.
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Type emission patterns.
//===----------------------------------------------------------------------===//

namespace {
///
struct OpaqueTypeEmitter : TypeEmissionPattern<OpaqueType> {
  void emitType(OpaqueType type, EmissionPrinter &p) override {
    p << type.getValue();
  }
};

///
struct PointerTypeEmitter : TypeEmissionPattern<PointerType> {
  void emitType(PointerType type, EmissionPrinter &p) override {
    p.emitType(type.getPointee());
    p << "*";
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateEmitCOpEmitters(
    OpEmissionPatternSet &patterns, MLIRContext *context) {
  patterns.add<IncludeEmitter, ApplyOpEmitter, CallOpEmitter, CastOpEmitter>(
      context);
}

void circt::ExportSystemC::populateEmitCTypeEmitters(
    TypeEmissionPatternSet &patterns) {
  patterns.add<OpaqueTypeEmitter, PointerTypeEmitter>();
}
