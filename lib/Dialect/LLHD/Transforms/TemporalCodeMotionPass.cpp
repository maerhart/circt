//===- TemporalCodeMotionPass.cpp - Implement Temporal Code Motion Pass ---===//
//
// Implement Pass to move all signal drives in a unique exiting block per
// temporal region and coalesce drives to the same signal.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "DNFUtil.h"
#include "TemporalRegions.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace mlir::llhd;

static void moveDrvInBlock(DrvOp drvOp, Block *dominator,
                           Operation *moveBefore) {
  OpBuilder builder(drvOp);
  builder.setInsertionPoint(moveBefore);
  Block *drvParentBlock = drvOp.getOperation()->getBlock();

  // Find sequence of branch decisions and add them as a sequence of
  // instructions to the TR exiting block
  // auto dnf = getBooleanExprFromSourceToTarget(dominator, drvParentBlock);
  // Value finalValue = dnf->buildOperations(builder);
  Value finalValue = getBooleanExprFromSourceToTargetNonDnf(builder, dominator, drvParentBlock);

  if (drvOp.getOperation()->getNumOperands() == 4) {
    finalValue =
        builder.create<llhd::AndOp>(drvOp.getLoc(), drvOp.enable(), finalValue);
    drvOp.getOperation()->setOperand(3, finalValue);
  } else {
    drvOp.getOperation()->insertOperands(3, finalValue);
  }
  drvOp.getOperation()->moveBefore(moveBefore);
} // namespace

namespace {
struct TemporalCodeMotionPass
    : public llhd::TemporalCodeMotionBase<TemporalCodeMotionPass> {
  void runOnOperation() override;
};
} // namespace

void TemporalCodeMotionPass::runOnOperation() {
  ProcOp proc = getOperation();
  TemporalRegionAnalysis trAnalysis = TemporalRegionAnalysis(proc);
  unsigned numTRs = trAnalysis.getNumTemporalRegions();

  // Only support processes with max. 2 temporal regions and one wait terminator
  // as this is enough to represent flip-flops, registers, etc.
  // NOTE: there always has to be either a wait or halt terminator in a process
  // If the wait block creates the backwards edge, we only have one TR,
  // otherwise we have 2 TRs
  // NOTE: as the wait instruction needs to be on every paths around the loop,
  // it has to be the only exiting block of its TR
  // NOTE: the other TR can either have only one exiting block, then we do not
  // need to add an auxillary block, otherwise we have to add one
  // NOTE: All drive operations have to be moved in the single exiting block of
  // its TR, to do that add the condition under which its block is reached from
  // the TR entry block as a gating condition to the drv instruction
  // NOTE: the entry blocks that are not part of the infinite loop do not count
  // as TR and have TR number -1
  // TODO: need to check that entry blocks that are note part of the loop to not
  // any instructions that have side effects that should not be allowed outside
  // of the loop (drv, prb, ...)
  // TODO: add support for more TRs and wait terminators to represent FSMs
  if (numTRs > 2) {
    // proc.emitError("More than 2 temporal regions are currently not supported!");
    // signalPassFailure();
    return;
  }

  bool seenWait = false;
  WalkResult walkResult = proc.walk([&](WaitOp op) -> WalkResult {
    if (seenWait) {
      return failure(); //op.emitError("Only one wait operation per process supported!");
    }
    // Check that the block containing the wait is the only exiting block of
    // that TR
    if (!trAnalysis.hasSingleExitBlock(
            trAnalysis.getBlockTR(op.getOperation()->getBlock()))) {
      return failure(); // op.emitError(
          // "Block with wait terinator has to be the only exiting block "
          // "of that temporal region!");
    }
    seenWait = true;
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    // signalPassFailure();
    return;
  }

  //===--------------------------------------------------------------------===//
  // Create unique exiting block per TR
  //===--------------------------------------------------------------------===//
  
  // TODO: consider the case where a wait brances to itself
  for (int currTR = 0; currTR < (int)numTRs; ++currTR) {
    unsigned numTRSuccs = trAnalysis.getNumTRSuccessors(currTR);
    assert((numTRSuccs == 1 ||
            numTRSuccs == 2 && trAnalysis.isOwnTRSuccessor(currTR)) &&
           "only TRs with a single TR as possible successor are "
           "supported for now.");

    if (trAnalysis.hasSingleExitBlock(currTR))
      continue;

    // Get entry block of successor TR
    Block *succTREntry =
        trAnalysis.getTREntryBlock(*trAnalysis.getTRSuccessors(currTR).begin());

    // Create the auxillary block as we currently don't have a single exiting
    // block and give it the same arguments as the entry block of the
    // successor TR
    Block *auxBlock = new Block();
    auxBlock->addArguments(succTREntry->getArgumentTypes());

    // Insert the auxillary block after the last block of the current TR
    proc.body().getBlocks().insertAfter(
        Region::iterator(trAnalysis.getExitingBlocksInTR(currTR).back()),
        auxBlock);

    for (Block *exit : trAnalysis.getExitingBlocksInTR(currTR)) {
      for (unsigned succ = 0; succ < exit->getTerminator()->getNumSuccessors();
           ++succ) {
        // TODO: this does not work when there can be multiple different
        // successor TRs
        if (trAnalysis.getBlockTR(exit->getTerminator()->getSuccessor(succ)) !=
            currTR) {
          exit->getTerminator()->setSuccessor(auxBlock, succ);
        }
      }
    }

    OpBuilder b(proc.getOperation());
    b.setInsertionPointToEnd(auxBlock);
    b.create<BranchOp>(proc.getLoc(), succTREntry, auxBlock->getArguments());
  }

  //===--------------------------------------------------------------------===//
  // Move drive instructions
  //===--------------------------------------------------------------------===//

  // Force a new analysis as we have changed the CFG
  trAnalysis = TemporalRegionAnalysis(proc);
  numTRs = trAnalysis.getNumTemporalRegions();
  OpBuilder builder(proc);
  for (int currTR = 0; currTR < (int)numTRs; ++currTR) {
    if (trAnalysis.getExitingBlocksInTR(currTR).size() != 1) {
      emitError(proc.getLoc(), "TR has not exactly one exiting block.");
      signalPassFailure();
      return;
    }
    Block *exitingBlock = trAnalysis.getExitingBlocksInTR(currTR)[0];
    Block *entryBlock = trAnalysis.getTREntryBlock(currTR);
    Operation *moveBefore = nullptr;

    DominanceInfo dom(proc);
    Block *dominator = exitingBlock;

    // Set insertion point before first drv op in exiting block
    exitingBlock->walk([&](DrvOp op) {
      dominator = dom.findNearestCommonDominator(dominator, op.getOperation()->getBlock());
      builder.setInsertionPoint(op);
      moveBefore = op.getOperation();
      return;
    });
    if (!dominator) {
      proc.emitError("Could not find nearest common dominator for TR exiting "
                      "block and the block containing drv");
    }

    if (trAnalysis.getBlockTR(dominator) != currTR)
      dominator = trAnalysis.getTREntryBlock(currTR);

    if (!moveBefore) {
      builder.setInsertionPointToEnd(exitingBlock);
      moveBefore = exitingBlock->getTerminator();
    }

    SmallPtrSet<Block *, 32> workQueue;
    SmallPtrSet<Block *, 32> workDone;

    if (entryBlock != exitingBlock)
      workQueue.insert(entryBlock);

    while (!workQueue.empty()) {
      auto iter = workQueue.begin();
      Block *block = *iter;
      workQueue.erase(block);
      workDone.insert(block);

      block->walk([&](DrvOp op) {
        moveDrvInBlock(op, dominator, moveBefore);
      });

      for (Block *succ : block->getSuccessors()) {
        if (succ == exitingBlock || trAnalysis.getBlockTR(succ) != currTR)
          continue;

        bool allPredDone = true;
        for (Block *pred : succ->getPredecessors()) {
          if (std::find(workDone.begin(), workDone.end(), pred) ==
              workDone.end()) {
            allPredDone = false;
            break;
          }
        }
        if (allPredDone) {
          workQueue.insert(succ);
        }
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Coalesce multiple drives to the same signal
  //===--------------------------------------------------------------------===//

  for (int currTR = 0; currTR < (int)numTRs; ++currTR) {
    if (trAnalysis.getExitingBlocksInTR(currTR).size() != 1) {
      emitError(proc.getLoc(), "TR has not exactly one exiting block.");
      signalPassFailure();
      return;
    }
    Block *exitingBlock = trAnalysis.getExitingBlocksInTR(currTR)[0];
    DenseMap<std::pair<Value, Value>, DrvOp> sigToDrv;
    exitingBlock->walk([&](DrvOp op) {
      std::pair<Value, Value> sigTimePair = std::make_pair(op.signal(), op.time());
      if (!sigToDrv.count(sigTimePair)) {
        sigToDrv.insert(std::make_pair(sigTimePair, op));
      } else {
        OpBuilder builder(op);
        if (op.enable()) {
          // Multiplex value to be driven
          if (op.value() != sigToDrv[sigTimePair].value()) {
          Value muxValue = 
              builder.create<SelectOp>(op.getLoc(), op.enable(), op.value(),
                                       sigToDrv.lookup(sigTimePair).value());
          op.valueMutable().assign(muxValue);
          }
          // Take the disjunction of the enable conditions
          if (sigToDrv[sigTimePair].enable()) {
            Value orVal = builder.create<llhd::OrOp>(
                op.getLoc(), op.enable(),
                sigToDrv[sigTimePair].enable());
            op.enableMutable().assign(orVal);
          } else {
            op.enableMutable().clear();
          }
        }
        Operation *toDelete = sigToDrv[sigTimePair].getOperation();
        toDelete->dropAllReferences();
        toDelete->erase();
        sigToDrv[sigTimePair] = op;
      }
    });
  }
}

std::unique_ptr<OperationPass<ProcOp>>
mlir::llhd::createTemporalCodeMotionPass() {
  return std::make_unique<TemporalCodeMotionPass>();
}
