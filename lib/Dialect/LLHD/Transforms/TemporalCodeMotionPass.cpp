//===- TemporalCodeMotionPass.cpp - Implement Temporal Code Motion Pass ---===//
//
// Implement Pass to move all signal drives in a unique exiting block per
// temporal region and coalesce drives to the same signal.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/LLHD/Analysis/TemporalRegions.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::llhd;

namespace {

void moveDrvInBlock(ProcOp proc, TemporalRegionAnalysis &analysis, DrvOp drvOp,
                    Block *dominator, Block *target, Operation *moveBefore) {
  OpBuilder builder(drvOp);
  builder.setInsertionPoint(moveBefore);
  Block *drvParentBlock = drvOp.getOperation()->getBlock();

  int tr = analysis.getBlockTR(drvParentBlock);

  // Find sequence of branch decisions and add them as a sequence of
  // instructions to the TR exiting block
  //
  // Mark all the blocks as not visited
  unsigned numBlocksInTR = analysis.numBlocksInTR(tr);
  DenseMap<Block *, bool> visited;

  // Create an array to store paths
  Block *path[numBlocksInTR];
  int path_index = 0; // Initialize path[] as empty

  // Initialize all blocks as not visited
  for (Block *block : analysis.getBlocksInTR(tr))
    visited.insert(std::make_pair(block, false));

  // A recursive function to get all paths from dominator
  // to drvParentBlock. visited[] keeps track of blocks in
  // current path. path[] stores actual blocks and
  // path_index is current index in path[]
  Value finalValue = Value();
  std::function<void(Block *)> getAllPathsUtil = [&](Block *curr) {
    // Mark the current node and store it in path[]
    visited[curr] = true;
    path[path_index] = curr;
    path_index++;

    // If current vertex is same as destination, then
    // print current path[]
    if (curr == drvParentBlock) {
      Value prevVal = Value();
      for (int i = 0; i < path_index; i++) {
        if (CondBranchOp br =
                dyn_cast<CondBranchOp>(path[i]->getTerminator())) {
          if (path[i] != drvParentBlock) {
            Value val;
            if (br.falseDest() == path[i + 1]) {
              val = builder
                        .create<llhd::NotOp>(target->begin()->getLoc(),
                                             br.condition())
                        .getResult();
            } else {
              val = br.condition();
            }
            if (prevVal) {
              prevVal = builder
                            .create<llhd::AndOp>(target->begin()->getLoc(),
                                                 prevVal, val)
                            .getResult();
            } else {
              prevVal = val;
            }
          }
        }
      }
      if (finalValue) {
        finalValue = builder
                         .create<llhd::OrOp>(target->begin()->getLoc(),
                                             finalValue, prevVal)
                         .getResult();
      } else {
        finalValue = prevVal;
      }
    } else {
      // Recur for all the vertices adjacent to current
      // vertex
      for (Block *succ : curr->getSuccessors())
        if (!visited[succ])
          getAllPathsUtil(succ);
    }

    // Remove current vertex from path[] and mark it as
    // unvisited
    path_index--;
    visited[curr] = false;
  };

  // Call the recursive helper function to get all paths
  if (dominator) {
    getAllPathsUtil(dominator);

    if (drvOp.getOperation()->getNumOperands() == 4 && finalValue) {
      drvOp.getOperation()->setOperand(
          3, builder
                 .create<llhd::AndOp>(target->begin()->getLoc(), drvOp.enable(),
                                      finalValue)
                 .getResult());
    } else if (finalValue) {
      drvOp.getOperation()->insertOperands(3, finalValue);
    }
  } else {
    emitRemark(drvOp.getLoc(), "Dominator is nullptr");
  }
  drvOp.getOperation()->moveBefore(moveBefore);
} // namespace

struct TemporalCodeMotionPass
    : public llhd::TemporalCodeMotionBase<TemporalCodeMotionPass> {
  void runOnOperation() override;
};

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
    proc.emitError("More than 2 temporal regions are currently not supported!");
    signalPassFailure();
    return;
  }

  bool seenWait = false;
  WalkResult walkResult = proc.walk([&](WaitOp op) -> WalkResult {
    if (seenWait) {
      return op.emitError("Only one wait operation per process supported!");
    }
    // Check that the block containing the wait is the only exiting block of
    // that TR
    if (!trAnalysis.hasSingleExitBlock(
            trAnalysis.getBlockTR(op.getOperation()->getBlock()))) {
      return op.emitError(
          "Block with wait terinator has to be the only exiting block "
          "of that temporal region!");
    }
    seenWait = true;
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    signalPassFailure();
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

    if (!moveBefore) {
      builder.setInsertionPointToEnd(exitingBlock);
      moveBefore = exitingBlock->getTerminator();
    }

    std::set<Block *> workQueue;
    std::set<Block *> workDone;

    workQueue.insert(entryBlock);

    while (!workQueue.empty()) {
      auto iter = workQueue.begin();
      Block *block = *iter;
      workQueue.erase(iter);
      workDone.insert(block);

      block->walk([&](DrvOp op) {
        moveDrvInBlock(proc, trAnalysis, op, dominator, exitingBlock, moveBefore);
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
} // namespace

std::unique_ptr<OperationPass<ProcOp>>
mlir::llhd::createTemporalCodeMotionPass() {
  return std::make_unique<TemporalCodeMotionPass>();
}
