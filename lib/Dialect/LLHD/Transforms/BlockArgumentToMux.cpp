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

#include "PassDetails.h"
#include "DNFUtil.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace circt;

static MutableOperandRange getDestOperands(Block *block, Block *succ) {
  if (auto op = dyn_cast<cf::BranchOp>(block->getTerminator()))
    return op.getDestOperandsMutable();

  if (auto op = dyn_cast<cf::CondBranchOp>(block->getTerminator()))
    return succ == op.getTrueDest() ? op.getTrueDestOperandsMutable()
                                 : op.getFalseDestOperandsMutable();

  return cast<llhd::WaitOp>(block->getTerminator()).destOpsMutable();
}

namespace {
struct BlockArgumentToMuxPass
    : public llhd::BlockArgumentToMuxBase<BlockArgumentToMuxPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void BlockArgumentToMuxPass::runOnOperation() {
  llhd::ProcOp proc = getOperation();
  DominanceInfo dom(proc);
  DenseMap<Block *, Value> mem;
  for (Block &block : proc.getBlocks()) {
    if (block.getNumArguments() == 0)
      continue;

    // If any of the passed block arguments is not defined in a block that dominates this block, don't convert the block argument to a select


    // Find the nearest common dominator of all predecessors.
    // If a block dominates all predecessors of a block, it also dominates this
    // block
    OpBuilder builder(proc);
    Block *domBlock = &block;
    bool doesNotDominate = false;
    for (Block *pred : block.getPredecessors()) {
      domBlock = dom.findNearestCommonDominator(domBlock, pred);
      // Check if the predecessor has a conditional branch where both destinations are the same block but with different block arguments, then replace it with a select for each argument and a unconditional branch
      if (auto br = dyn_cast<cf::CondBranchOp>(pred->getTerminator())) {
        if (br.getTrueDest() == &block && br.getFalseDest() == &block) {
          SmallVector<Value, 4> newArgs;
          builder.setInsertionPoint(br);
          for (auto &&args : llvm::zip(br.getTrueDestOperands(), br.getFalseDestOperands())) {
            newArgs.push_back(builder.create<comb::MuxOp>(br.getLoc(), br.getCondition(), std::get<0>(args), std::get<1>(args)));
          }
          builder.create<cf::BranchOp>(br.getLoc(), br.getTrueDest(), newArgs);
          br.getOperation()->dropAllReferences();
          br.getOperation()->erase();
        }
      }
      // Check for loops
      for (Value arg : (OperandRange)getDestOperands(pred, &block)) {
        if (!dom.properlyDominates(arg.getParentBlock(), &block))
          doesNotDominate = true;
        // if (arg.isa<BlockArgument>())
        //   doesNotDominate = true;
      }
      for (auto &&item : llvm::zip((OperandRange)getDestOperands(pred, &block), block.getArguments())) {
        if(std::get<0>(item) == std::get<1>(item))
        doesNotDominate = true;
      }
      // if passed argument is the block argument to be erased itself

    }
    if (doesNotDominate)
     continue;
    builder.setInsertionPointToStart(&block);

    // Create an array which stores the replacement values for the current block
    // arguments of this block and initialize it
    Value valToMux[block.getNumArguments()];
    for (unsigned i = 0; i < block.getNumArguments(); i++) {
      valToMux[i] = Value();
    }

    // For every predecessor, find the sequence of branch decisions from the
    // nearest common dominator and add them as a sequence of instructions to
    // the TR exiting block
    for (Block *predBlock : block.getPredecessors()) {
      Value finalValue = Value();
      if (predBlock != *block.pred_begin()) {
        // llhd::Dnf dnf =
        //     *llhd::getBooleanExprFromSourceToTarget(domBlock, predBlock);
        // finalValue = dnf.buildOperations(builder);
        finalValue = llhd::getBooleanExprFromSourceToTargetNonDnf(builder, domBlock, predBlock, mem);
      if (auto br = dyn_cast<cf::CondBranchOp>(predBlock->getTerminator())) {
        Value cond = br.getCondition();
        if (br.getFalseDest() == &block) {
          Value allset = builder.create<hw::ConstantOp>(proc.getLoc(), cond.getType(), -1);
          cond = builder.create<comb::XorOp>(proc.getLoc(), cond, allset);
        }
        finalValue = builder.createOrFold<comb::AndOp>(proc.getLoc(), finalValue, cond);
      }
      }


      // Create the select operations to select the value depending on the
      // predecessor. We use the condition created above to do the selection
      for (unsigned i = 0; i < block.getNumArguments(); i++) {
        Value arg = ((OperandRange)getDestOperands(predBlock, &block))[i];
        if (!valToMux[i]) {
          valToMux[i] = arg;
          continue;
        }
        if (finalValue && arg)
          valToMux[i] = builder.create<comb::MuxOp>(proc.getLoc(), finalValue, arg,
                                                 valToMux[i]);
      }
    }

    // Replace the block arguments with the multiplexed values
    std::vector<unsigned> argsToDelete;
    for (unsigned i = 0; i < block.getNumArguments(); i++) {
      if (valToMux[i] && valToMux[i] != block.getArgument(i)) {
        block.getArgument(i).replaceAllUsesWith(valToMux[i]);
        argsToDelete.push_back(i);
      }
    }

    // Delete the replaced block arguments and also delete the passed operands
    // in the predecessor blocks
    std::sort(argsToDelete.begin(), argsToDelete.end(), std::greater<>());
    for (unsigned arg : argsToDelete) {
      for (Block *pred : block.getPredecessors()) {
        getDestOperands(pred, &block).erase(arg);
      }
      block.eraseArgument(arg);
    }
  }
}

std::unique_ptr<OperationPass<llhd::ProcOp>>
circt::llhd::createBlockArgumentToMuxPass() {
  return std::make_unique<BlockArgumentToMuxPass>();
}
