//===- HWEmissionPatterns.cpp - HW Dialect Emission Patterns --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the HW dialect.
//
//===----------------------------------------------------------------------===//

#include "HWEmissionPatterns.h"
#include "../EmissionPattern.h"
#include "../EmissionPrinter.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::hw;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// The ConstantOp always inlines its value. Examples:
/// * hw.constant 5 : i32 ==> 5
/// * hw.constant 0 : i1 ==> false
/// * hw.constant 1 : i1 ==> true
struct ConstantEmitter : OpEmissionPattern<ConstantOp> {
  using OpEmissionPattern::OpEmissionPattern;
  MatchResult matchInlinable(Value value, EmissionConfig &config) override {
    if (isa_and_nonnull<ConstantOp>(value.getDefiningOp()))
      return MatchResult(Precedence::LIT);
    return MatchResult();
  }
  void emitInlined(Value value, EmissionConfig &config,
                   EmissionPrinter &p) override {
    auto op = value.getDefiningOp<ConstantOp>();

    if (op.getValue().getBitWidth() == 1) {
      p << (op.getValue().getBoolValue() ? "true" : "false");
      return;
    }

    SmallString<32> valueString;
    op.getValue().toStringUnsigned(valueString);
    p << valueString;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Type emission patterns.
//===----------------------------------------------------------------------===//

namespace {
///
struct IntegerTypeEmitter : TypeEmissionPattern<IntegerType> {
  void emitType(IntegerType type, EmissionConfig &config,
                EmissionPrinter &p) override {
    p << "sc_dt::sc_uint<" << type.getIntOrFloatBitWidth() << ">";
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateHWEmitters(OpEmissionPatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<ConstantEmitter>(context);
}

void circt::ExportSystemC::populateHWTypeEmitters(
    TypeEmissionPatternSet &patterns) {
  patterns.add<IntegerTypeEmitter>();
}
