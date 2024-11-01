//===- UnrollPass.cpp - Implement the Sig2Reg Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement Pass to unroll CFG loops inside processes.
//
//===----------------------------------------------------------------------===//

#include "TemporalRegions.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-unroll"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_UNROLL
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {
struct UnrollPass : public circt::llhd::impl::UnrollBase<UnrollPass> {
  using Base::Base;
  void runOnOperation() override;
  LogicalResult runOnProcess(llhd::ProcessOp procOp);
};
} // namespace

/// Takes the condition value from the cond_br as input
static int getTripCount(Value val) {
  auto icmpOp = val.getDefiningOp<comb::ICmpOp>();
  if (!icmpOp)
    return -1;

  if (!isa<BlockArgument>(icmpOp.getLhs()))
    return -1;

  Value boundVal = icmpOp.getRhs();
  if (auto wireOp = boundVal.getDefiningOp<hw::WireOp>())
    boundVal = wireOp.getInput();

  if (auto constOp = boundVal.getDefiningOp<hw::ConstantOp>())
    return constOp.getValue().getZExtValue();

  return -1;
}

LogicalResult UnrollPass::runOnProcess(llhd::ProcessOp procOp) {
  llhd::TemporalRegionAnalysis trAnalysis(procOp);

  //===--------------------------------------------------------------------===//
  // Create unique exit block per TR
  //===--------------------------------------------------------------------===//

  unsigned numTRs = trAnalysis.getNumTemporalRegions();

  // TODO: consider the case where a wait brances to itself
  for (unsigned currTR = 0; currTR < numTRs; ++currTR) {
    unsigned numTRSuccs = trAnalysis.getNumTRSuccessors(currTR);
    // NOTE: Above error checks make this impossible to trigger, but the above
    // are changed this one might have to be promoted to a proper error message.

    if (!((numTRSuccs == 1 ||
           (numTRSuccs == 2 && trAnalysis.isOwnTRSuccessor(currTR))))) {
      for (unsigned i = 0; i < numTRs; ++i) {
        llvm::outs() << "Blocks in TR " << i << "\n";
        for (auto *block : trAnalysis.getBlocksInTR(i))
          llvm::outs() << *block << "\n";
      }
      return procOp.emitError(
                 "only TRs with a single TR as possible successor are "
                 "supported for now. Num successors: ")
             << numTRSuccs;
    }

    if (trAnalysis.hasSingleExitBlock(currTR))
      continue;

    // Get entry block of successor TR
    Block *succTREntry =
        trAnalysis.getTREntryBlock(*trAnalysis.getTRSuccessors(currTR).begin());

    // Create the auxillary block as we currently don't have a single exiting
    // block and give it the same arguments as the entry block of the
    // successor TR
    Block *auxBlock = new Block();
    auxBlock->addArguments(
        succTREntry->getArgumentTypes(),
        SmallVector<Location>(succTREntry->getNumArguments(), procOp.getLoc()));

    // Insert the auxillary block after the last block of the current TR
    procOp.getBody().getBlocks().insertAfter(
        Region::iterator(trAnalysis.getExitingBlocksInTR(currTR).back()),
        auxBlock);

    // Let all current exit blocks branch to the auxillary block instead.
    for (Block *exit : trAnalysis.getExitingBlocksInTR(currTR))
      for (auto [i, succ] : llvm::enumerate(exit->getSuccessors()))
        if (trAnalysis.getBlockTR(succ) != static_cast<int>(currTR))
          exit->getTerminator()->setSuccessor(auxBlock, i);

    // Let the auxiallary block branch to the entry block of the successor
    // temporal region entry block
    OpBuilder b(procOp);
    b.setInsertionPointToEnd(auxBlock);
    b.create<cf::BranchOp>(procOp.getLoc(), succTREntry,
                           auxBlock->getArguments());
  }

  trAnalysis = llhd::TemporalRegionAnalysis(procOp);
  numTRs = trAnalysis.getNumTemporalRegions();
  DominanceInfo dom(getOperation());

  for (unsigned i = 0; i < numTRs; ++i) {
    if (trAnalysis.getExitingBlocksInTR(i).size() != 1)
      continue;

    if (trAnalysis.getBlocksInTR(i).size() <= 1)
      continue;

    for (auto *block : trAnalysis.getBlocksInTR(i)) {
      SmallVector<Block *> worklist;
      bool foundLoop = false;
      for (auto *pred : block->getPredecessors()) {
        if (dom.dominates(block, pred)) {
          foundLoop = true;
          if (pred != block)
            worklist.push_back(pred);
        }
      }

      if (!foundLoop)
        continue;

      auto condBr = dyn_cast<cf::CondBranchOp>(block->getTerminator());
      if (!condBr)
        continue;
      // return block->getTerminator()->emitError("expected cond_br");

      int tripCount = getTripCount(condBr.getCondition());
      if (tripCount == -1)
        continue;
      // return condBr->emitError("could not determine trip count");

      DenseSet<Block *> seen;
      SmallVector<Block *> blocksInLoop;
      while (!worklist.empty()) {
        auto *curr = worklist.pop_back_val();
        if (!seen.contains(curr)) {
          seen.insert(curr);
          blocksInLoop.push_back(curr);
          for (auto *pred : curr->getPredecessors())
            if (pred != block)
              worklist.push_back(pred);
        }
      }

      // TODO: handle tripCount == 0

      OpBuilder builder(condBr);
      // FIXME: don't just assume that the false branch exits the loop

      IRMapping mapping;
      // mapping.map(block, block);
      // for (auto operand : exitOperands)
      //   mapping.map(operand, operand);

      // Clone the header block once
      auto *exitBlock = builder.createBlock(
          block, block->getArgumentTypes(),
          SmallVector<Location>(block->getNumArguments(), procOp->getLoc()));
      for (auto [oldArg, newArg] :
           llvm::zip(block->getArguments(), exitBlock->getArguments()))
        mapping.map(oldArg, newArg);
      mapping.map(block, exitBlock);
      builder.setInsertionPointToStart(exitBlock);
      for (auto &op : block->getOperations())
        builder.clone(op, mapping);

      SmallVector<Block *> preds(block->getPredecessors());

      for (auto *b : blocksInLoop)
        for (auto [i, succ] :
             llvm::enumerate(b->getTerminator()->getSuccessors()))
          if (succ == block)
            b->getTerminator()->setSuccessor(exitBlock, i);

      for (auto [i, succ] :
           llvm::enumerate(block->getTerminator()->getSuccessors()))
        if (succ == block)
          block->getTerminator()->setSuccessor(exitBlock, i);

      // mapping.map(clonedBlock, block);

      builder.setInsertionPoint(condBr);
      builder.create<cf::BranchOp>(condBr.getLoc(), condBr.getTrueDest(),
                                   condBr.getTrueDestOperands());

      auto clonedCondBr = cast<cf::CondBranchOp>(mapping.lookup(condBr));
      builder.setInsertionPoint(clonedCondBr);
      builder.create<cf::BranchOp>(clonedCondBr.getLoc(),
                                   clonedCondBr.getFalseDest(),
                                   clonedCondBr.getFalseDestOperands());
      condBr->erase();
      clonedCondBr->erase();

      // mapping.clear();

      Block *prevHeader = block;
      for (int i = 1; i < tripCount; ++i) {
        // mapping.clear();
        auto *clonedBlock = builder.createBlock(
            block, block->getArgumentTypes(),
            SmallVector<Location>(block->getNumArguments(), procOp->getLoc()));
        for (auto [oldArg, newArg] :
             llvm::zip(block->getArguments(), clonedBlock->getArguments()))
          mapping.map(oldArg, newArg);
        // mapping.map(block, clonedBlock);
        mapping.map(exitBlock, prevHeader);
        prevHeader = clonedBlock;
        for (auto *b : blocksInLoop) {
          auto *clonedBlock = builder.createBlock(
              b, b->getArgumentTypes(),
              SmallVector<Location>(b->getNumArguments(), procOp->getLoc()));
          mapping.map(b, clonedBlock);
          for (auto [oldArg, newArg] :
               llvm::zip(b->getArguments(), clonedBlock->getArguments()))
            mapping.map(oldArg, newArg);
        }

        builder.setInsertionPointToStart(clonedBlock);
        for (auto &op : block->getOperations())
          builder.clone(op, mapping);

        for (auto *b : blocksInLoop) {
          builder.setInsertionPointToStart(mapping.lookup(b));
          for (auto &op : b->getOperations())
            builder.clone(op, mapping);
        }
      }

      for (auto *b : preds)
        for (auto [i, succ] :
             llvm::enumerate(b->getTerminator()->getSuccessors()))
          if (succ == block)
            b->getTerminator()->setSuccessor(prevHeader, i);
    }
  }

  return success();
}

void UnrollPass::runOnOperation() {
  hw::HWModuleOp moduleOp = getOperation();

  for (auto procOp : moduleOp.getOps<llhd::ProcessOp>())
    if (failed(runOnProcess(procOp)))
      return signalPassFailure();
}
