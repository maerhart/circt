//===- SCFEmissionPatterns.cpp - SCF Dialect Emission Patterns ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "SCFEmissionPatterns.h"
#include "../EmissionPattern.h"
#include "../EmissionPrinter.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::ExportSystemC;
using namespace mlir::scf;

namespace {
struct WhileEmitter : OpEmissionPattern<WhileOp> {
  using OpEmissionPattern::OpEmissionPattern;
  void emitStatement(WhileOp op, EmissionConfig &config,
                     EmissionPrinter &p) override {
    auto condition = p.getInlinable(op.getConditionOp().getCondition());
    p << "while (";
    condition.emit();
    p << ") ";
    p.emitRegion(op.getAfter());
  }
};
} // namespace

void circt::ExportSystemC::populateSCFEmitters(OpEmissionPatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add<OpEmissionPattern<ConditionOp>, OpEmissionPattern<YieldOp>,
               WhileEmitter>(context);
}
