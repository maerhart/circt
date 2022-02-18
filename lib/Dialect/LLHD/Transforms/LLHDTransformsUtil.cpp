//===- LLHDTransformsUtil.cpp - Helper functions for LLHD passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A collection of helper functions for various LLHD transformation passes.
//
//===----------------------------------------------------------------------===//

#include "LLHDTransformsUtil.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace circt;

Value llhd::getBranchDecisionsFromDominatorToTarget(
    OpBuilder &builder, Block *driveBlock, Block *dominator,
    DenseMap<Block *, Value> &mem) {
  Location loc = driveBlock->getTerminator()->getLoc();
  if (mem.count(driveBlock))
    return mem[driveBlock];

  SmallVector<Block *> worklist;
  worklist.push_back(driveBlock);

  while (!worklist.empty()) {
    Block *curr = worklist.back();

    if (curr == dominator || curr->getPredecessors().empty()) {
      if (!mem.count(curr))
        mem[curr] = builder.create<hw::ConstantOp>(loc, APInt(1, 1));

      worklist.pop_back();
      continue;
    }

    bool addedSomething = false;
    for (auto *predBlock : curr->getPredecessors()) {
      if (!mem.count(predBlock)) {
        worklist.push_back(predBlock);
        addedSomething = true;
      }
    }

    if (addedSomething)
      continue;

    Value runner = builder.create<hw::ConstantOp>(loc, APInt(1, 0));
    for (auto *predBlock : curr->getPredecessors()) {
      if (predBlock->getTerminator()->getNumSuccessors() != 1) {
        auto condBr = cast<mlir::cf::CondBranchOp>(predBlock->getTerminator());
        Value cond = condBr.getCondition();
        if (condBr.getFalseDest() == curr) {
          Value trueVal = builder.create<hw::ConstantOp>(loc, APInt(1, 1));
          cond = builder.create<comb::XorOp>(loc, cond, trueVal);
        }
        Value next = builder.create<comb::AndOp>(loc, mem[predBlock], cond);
        runner = builder.create<comb::OrOp>(loc, runner, next);
      } else {
        runner = builder.create<comb::OrOp>(loc, runner, mem[predBlock]);
      }
    }
    mem[curr] = runner;
    worklist.pop_back();
  }

  return mem[driveBlock];
}
