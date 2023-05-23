//===- StripSV.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <variant>

#define DEBUG_TYPE "arc-vectorize"

using namespace circt;
using namespace arc;

namespace {
struct VectorizePass : public VectorizeBase<VectorizePass> {
  void runOnOperation() override;
};
} // namespace

void VectorizePass::runOnOperation() {
  llvm::MapVector<StringAttr, SmallVector<StateOp>> grouping;

  for (auto stateOp : getOperation().getBodyBlock().getOps<StateOp>())
    if (
        // stateOp.getLatency() > 0 &&
        stateOp->getNumResults() == 1 &&
        llvm::all_of(stateOp.getInputs().getTypes(),
                     [](Type ty) { return isa<IntegerType>(ty); }))
      grouping[stateOp.getArcAttr().getAttr()].push_back(stateOp);

  OpBuilder builder(getOperation().getBodyBlock().getTerminator());
  for (auto [_, states] : grouping) {
    if (states.size() <= 1)
      continue;

    auto parallelBlock = std::make_unique<Block>();
    SmallVector<Location> locs;
    SmallVector<SmallVector<Value>> inputs(states[0].getInputs().size());
    for (auto [i, state] : llvm::enumerate(states)) {
      locs.push_back(state.getLoc());
      for (auto [j, val] : llvm::enumerate(state.getInputs()))
        inputs[j].emplace_back(val);
    }
    Location fusedLoc = builder.getFusedLoc(locs);

    SmallVector<ValueRange> inputRanges;

    for (auto &input : inputs) {
      ++numVectors;
      if (llvm::all_equal(input)) {
        inputRanges.emplace_back(input[0]);
        ++numBroadcasts;
        continue;
      }
      inputRanges.emplace_back(ValueRange{input});
    }

    states[0]->moveBefore(parallelBlock.get(), parallelBlock->end());
    SmallVector<Value> args(parallelBlock->addArguments(
        states[0].getInputs().getTypes(),
        SmallVector<Location>(states[0].getInputs().size(), fusedLoc)));
    auto parallelOp = builder.create<ParallelOp>(
        fusedLoc,
        SmallVector<Type>(states.size(), states[0]->getResult(0).getType()),
        inputRanges);
    states[0].getInputsMutable().assign(args);

    for (auto [i, state] : llvm::enumerate(states)) {
      state.getResult(0).replaceAllUsesWith(parallelOp->getResult(i));
      if (i != 0)
        state.erase();
    }

    auto ipSave = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(parallelBlock.get());
    builder.create<OutputOp>(fusedLoc, states[0]->getResults());
    parallelOp.getBody().push_back(parallelBlock.release());
    builder.restoreInsertionPoint(ipSave);
  }
}

std::unique_ptr<Pass> arc::createVectorizePass() {
  return std::make_unique<VectorizePass>();
}
