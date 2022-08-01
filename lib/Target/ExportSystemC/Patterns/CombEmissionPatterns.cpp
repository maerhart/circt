//===- CombEmissionPatterns.cpp - Comb Dialect Emission Patterns ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the comb dialect.
//
//===----------------------------------------------------------------------===//

#include "CombEmissionPatterns.h"
#include "../EmissionPattern.h"
#include "../EmissionPrinter.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::comb;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

///
static StringRef getExprString(Operation *op) {
  return TypeSwitch<Operation *, StringRef>(op)
      .Case<AddOp>([](auto op) { return " + "; })
      .Case<SubOp>([](auto op) { return " - "; })
      .Case<MulOp>([](auto op) { return " * "; })
      .Case<DivUOp>([](auto op) { return " / "; })
      .Case<ShlOp>([](auto op) { return " << "; })
      .Case<ShrUOp>([](auto op) { return " >> "; })
      .Case<ModUOp>([](auto op) { return " % "; })
      .Case<AndOp>([](auto op) { return " & "; })
      .Case<OrOp>([](auto op) { return " | "; })
      .Case<XorOp>([](auto op) { return " ^ "; });
}

///
static Precedence getExprPrecedence(Operation *op) {
  return TypeSwitch<Operation *, Precedence>(op)
      .Case<AddOp>([](auto op) { return Precedence::ADD; })
      .Case<SubOp>([](auto op) { return Precedence::SUB; })
      .Case<MulOp>([](auto op) { return Precedence::MUL; })
      .Case<DivUOp>([](auto op) { return Precedence::DIV; })
      .Case<ShlOp>([](auto op) { return Precedence::SHL; })
      .Case<ShrUOp>([](auto op) { return Precedence::SHR; })
      .Case<ModUOp>([](auto op) { return Precedence::MOD; })
      .Case<AndOp>([](auto op) { return Precedence::BITWISE_AND; })
      .Case<OrOp>([](auto op) { return Precedence::BITWISE_OR; })
      .Case<XorOp>([](auto op) { return Precedence::BITWISE_XOR; });
}

///
static void parenthesize(bool addParens, InlineEmitter emitter,
                         EmissionPrinter &p) {
  if (addParens)
    p << "(";
  emitter.emit();
  if (addParens)
    p << ")";
}

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

///
namespace {
template <typename Op>
struct VariadicExpressionEmitter : OpEmissionPattern<Op> {
  explicit VariadicExpressionEmitter(MLIRContext *context)
      : OpEmissionPattern<Op>(context) {}

  MatchResult matchInlinable(Value value, EmissionConfig &config) override {
    if (auto op = dyn_cast_or_null<Op>(value.getDefiningOp()))
      return MatchResult(getExprPrecedence(op));
    return MatchResult();
  }
  void emitInlined(Value value, EmissionConfig &config,
                   EmissionPrinter &p) override {
    Op op = value.getDefiningOp<Op>();
    bool first = true;
    for (Value value : op.getInputs()) {
      if (!first)
        p << getExprString(op);
      first = false;
      InlineEmitter operand = p.getInlinable(value);
      parenthesize(operand.getPrecedence() > getExprPrecedence(op), operand, p);
    }
  }
};

///
template <typename Op>
struct BinaryExpressionEmitter : OpEmissionPattern<Op> {
  explicit BinaryExpressionEmitter(MLIRContext *context)
      : OpEmissionPattern<Op>(context) {}

  MatchResult matchInlinable(Value value, EmissionConfig &config) override {
    if (auto op = dyn_cast_or_null<Op>(value.getDefiningOp()))
      return MatchResult(getExprPrecedence(op));
    return MatchResult();
  }
  void emitInlined(Value value, EmissionConfig &config,
                   EmissionPrinter &p) override {
    Op op = value.getDefiningOp<Op>();

    InlineEmitter lhs = p.getInlinable(op.getLhs());
    parenthesize(lhs.getPrecedence() > getExprPrecedence(op), lhs, p);

    p << getExprString(op);

    InlineEmitter rhs = p.getInlinable(op.getRhs());
    parenthesize(rhs.getPrecedence() > getExprPrecedence(op), rhs, p);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateCombEmitters(OpEmissionPatternSet &patterns,
                                                MLIRContext *context) {
  patterns
      .add<VariadicExpressionEmitter<AddOp>, VariadicExpressionEmitter<MulOp>,
           VariadicExpressionEmitter<AndOp>, VariadicExpressionEmitter<OrOp>,
           VariadicExpressionEmitter<XorOp>, BinaryExpressionEmitter<DivUOp>,
           BinaryExpressionEmitter<ModUOp>, BinaryExpressionEmitter<ShlOp>,
           BinaryExpressionEmitter<ShrUOp>, BinaryExpressionEmitter<SubOp>>(
          context);
}
