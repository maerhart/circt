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
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
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

  for (unsigned i = 0; i < numTRs; ++i) {
    if (trAnalysis.getExitingBlocksInTR(i).size() != 1)
      continue;

    if (trAnalysis.getBlocksInTR(i).size() <= 1)
      continue;

    Block *preExit = trAnalysis.getExitingBlocksInTR(i)[0];
    Block *entryBlock = trAnalysis.getTREntryBlock(i);

    // Collect values that are defined in this temporal region and used outside
    // of it or in the TR exit block terminator.
    SmallVector<Value> yieldValues;
    SmallVector<Block *> trBlocksVec(trAnalysis.getBlocksInTR(i));
    DenseSet<Block *> trBlocks;
    DenseMap<Block *, size_t> test;
    test[preExit]++;
    llvm::outs() << test[preExit];
    SmallVector<std::pair<Block *, size_t>> b(test.begin(), test.end());
    for (auto *bb : trBlocksVec)
      trBlocks.insert(bb);

    for (auto *bb : trBlocksVec) {
      for (auto &op : bb->getOperations()) {
        for (auto res : op.getResults()) {
          for (auto *user : res.getUsers()) {
            if (!trBlocks.contains(user->getBlock()) ||
                user == preExit->getTerminator()) {
              yieldValues.push_back(res);
              break;
            }
          }
        }
      }
    }

    Block *exitBlock = preExit->splitBlock(preExit->getTerminator());
    Block *postEntry = entryBlock->splitBlock(entryBlock->begin());

    OpBuilder builder(exitBlock->getTerminator());
    Location loc = procOp->getLoc();

    auto execRegion = builder.create<scf::ExecuteRegionOp>(
        loc, ValueRange(yieldValues).getTypes());

    preExit->moveBefore(&execRegion.getRegion(),
                        execRegion.getRegion().begin());
    for (auto *block : trBlocks) {
      if (block == exitBlock || block == entryBlock)
        continue;

      block->moveBefore(&execRegion.getRegion(),
                        execRegion.getRegion().begin());
    }
    postEntry->moveBefore(&execRegion.getRegion(),
                          execRegion.getRegion().begin());

    // Insert scf.yield at end of execute region
    builder.setInsertionPointToEnd(preExit);
    builder.create<scf::YieldOp>(loc, yieldValues);

    // Replace yielded values
    for (auto [val, res] : llvm::zip(yieldValues, execRegion->getResults()))
      val.replaceUsesWithIf(res, [&](OpOperand &operand) {
        return !execRegion->isAncestor(operand.getOwner());
      });

    // Merge entry and exit blocks.
    IRRewriter rewriter(builder);
    rewriter.mergeBlocks(exitBlock, entryBlock);

    ControlFlowToSCFTransformation transformation;
    auto &domInfo = getAnalysis<DominanceInfo>();
    FailureOr<bool> changed =
        transformCFGToSCF(execRegion.getRegion(), transformation, domInfo);
    if (failed(changed))
      return failure();

    // NOTE: we could use a combination of split block and merge block for
    // easier handling
    // TODO: redirect TR predecessors to TR exit block now
    // TODO: fixup block arguments of TR entry and exit blocks
  }

  return success();
}

void UnrollPass::runOnOperation() {
  hw::HWModuleOp moduleOp = getOperation();

  for (auto procOp : moduleOp.getOps<llhd::ProcessOp>())
    if (failed(runOnProcess(procOp)))
      return signalPassFailure();
}
