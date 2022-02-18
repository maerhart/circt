//===- BlockArgumentToMux.cpp - Implement Block Argument to Mux Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement pass to exhaustively convert block arguments to multiplexers to
// enable the canonicalization pass to remove all control flow.
//
//===----------------------------------------------------------------------===//

#include "LLHDTransformsUtil.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"

namespace circt {
#define GEN_PASS_DEF_BLOCKARGUMENTTOMUX
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static MutableOperandRange getDestOperands(Block &block, Block &succ) {
  if (auto op = dyn_cast<cf::BranchOp>(block.getTerminator()))
    return op.getDestOperandsMutable();

  if (auto op = dyn_cast<cf::CondBranchOp>(block.getTerminator()))
    return &succ == op.getTrueDest() ? op.getTrueDestOperandsMutable()
                                     : op.getFalseDestOperandsMutable();

  return cast<llhd::WaitOp>(block.getTerminator()).getDestOpsMutable();
}

static Value appendCondition(OpBuilder &builder, Block &block, Block &pred,
                             Value parentCond) {
  if (auto br = dyn_cast<cf::CondBranchOp>(pred.getTerminator())) {
    Value cond = br.getCondition();
    if (br.getFalseDest() == &block) {
      Value allset = builder.create<hw::ConstantOp>(block.front().getLoc(),
                                                    cond.getType(), -1);
      cond = builder.create<comb::XorOp>(block.front().getLoc(), cond, allset);
    }
    return builder.create<comb::AndOp>(block.front().getLoc(), parentCond,
                                       cond);
  }
  return parentCond;
}

//===----------------------------------------------------------------------===//
// Block Argument to Mux Pass
//
// This pass assumes that in earlier passes
//   * operations were moved up in the CFG as far as possible, s.t. they reside
//     in blocks that dominate blocks with arguments whenever possible (because
//     inserting a mux requires the passed block arguments to be accessible in
//     the block with the arguments)
//   * Loops were completely unrolled (to support above point)
//
// The pass still works when above criteria are not met, but likely does not
// convert all the block arguments.
//===----------------------------------------------------------------------===//

namespace {
struct BlockArgumentToMuxPass
    : public circt::impl::BlockArgumentToMuxBase<BlockArgumentToMuxPass> {
  void runOnOperation() override;
  void runOnProcess(llhd::ProcessOp proc);
};
} // anonymous namespace

void BlockArgumentToMuxPass::runOnOperation() {
  for (auto proc : getOperation().getOps<llhd::ProcessOp>())
    runOnProcess(proc);
}

void BlockArgumentToMuxPass::runOnProcess(llhd::ProcessOp proc) {
  DominanceInfo dom(proc);
  OpBuilder builder(proc);

  for (Block &block : proc.getBody().getBlocks()) {
    if (block.getNumArguments() == 0 || block.isEntryBlock())
      continue;

    // Find the nearest common dominator of all predecessors.
    // If a block dominates all predecessors of a block, it also dominates this
    // block
    Block *domBlock = &block;
    bool doesNotDominate = false;
    DenseMap<Block *, Value> mem;
    for (Block *pred : block.getPredecessors()) {
      domBlock = dom.findNearestCommonDominator(domBlock, pred);

      // Check if the predecessor has a conditional branch where both
      // destinations jump to the same block but with different block arguments,
      // then replace it with a mux for each argument and an unconditional
      // branch
      if (auto br = dyn_cast<cf::CondBranchOp>(pred->getTerminator())) {
        if (br.getTrueDest() == &block && br.getFalseDest() == &block) {
          SmallVector<Value, 4> newArgs;
          builder.setInsertionPoint(br);
          for (auto &&args :
               llvm::zip(br.getTrueDestOperands(), br.getFalseDestOperands())) {
            newArgs.push_back(builder.create<comb::MuxOp>(
                br.getLoc(), br.getCondition(), std::get<0>(args),
                std::get<1>(args)));
          }
          builder.create<cf::BranchOp>(br.getLoc(), br.getTrueDest(), newArgs);
          br->dropAllReferences();
          br->erase();
        }
      }

      // If the block arguments passed from a predecessor to this block are not
      // accessible in this block (because they are defined in a block that does
      // not dominate this block), we cannot insert a mux that needs to refer to
      // these values in this block. This is often the case when there are still
      // loops in the CFG.
      for (Value arg : (OperandRange)getDestOperands(*pred, block)) {
        if (!dom.properlyDominates(arg.getParentBlock(), &block)) {
          doesNotDominate = true;
          break;
        }
      }
    }

    if (doesNotDominate)
      continue;

    builder.setInsertionPointToStart(&block);

    // Create an array which stores the replacement values for the current block
    // arguments of this block and initialize it
    SmallVector<Value, 8> valToMux =
        (OperandRange)getDestOperands(**block.pred_begin(), block);

    // For every predecessor, find the sequence of branch decisions from the
    // nearest common dominator and add them as a sequence of instructions to
    // the TR exiting block
    for (Block *predBlock : block.getPredecessors()) {
      // Skip the first predecessor, because this will be the else-case of the
      // mux that we have already initialized above
      if (predBlock == *block.pred_begin())
        continue;

      Value tmp = llhd::getBranchDecisionsFromDominatorToTarget(
          builder, predBlock, domBlock, mem);
      Value finalValue = appendCondition(builder, block, *predBlock, tmp);

      // Create the mux operations to select the value depending on the
      // predecessor.
      for (size_t i = 0; i < valToMux.size(); i++) {
        Value arg = ((OperandRange)getDestOperands(*predBlock, block))[i];
        valToMux[i] = builder.create<comb::MuxOp>(proc.getLoc(), finalValue,
                                                  arg, valToMux[i]);
      }
    }

    // Replace the block arguments with the multiplexed values
    for (int i = valToMux.size() - 1; i >= 0; i--) {
      block.getArgument(i).replaceAllUsesWith(valToMux[i]);
      for (Block *pred : block.getPredecessors()) {
        getDestOperands(*pred, block).erase(i);
      }
      block.eraseArgument(i);
    }
  }
}

std::unique_ptr<::mlir::Pass> llhd::createBlockArgumentToMux() {
  return std::make_unique<BlockArgumentToMuxPass>();
}