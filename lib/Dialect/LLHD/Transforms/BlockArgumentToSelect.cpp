//===- BlockArgumentToMux.cpp - Implement Block Argument to Mux Pass ------===//
//
// Implement pass to exhaustively convert block arguments to multiplexers to
// enable the canonicalization pass to remove all control flow.
//
//===----------------------------------------------------------------------===//

#include "DNFUtil.h"
#include "PassDetails.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;

static MutableOperandRange getDestOperands(Block *block, Block *succ) {
  if (auto op = dyn_cast<BranchOp>(block->getTerminator()))
    return op.destOperandsMutable();

  if (auto op = dyn_cast<CondBranchOp>(block->getTerminator()))
    return succ == op.trueDest() ? op.trueDestOperandsMutable()
                                 : op.falseDestOperandsMutable();

  return cast<llhd::WaitOp>(block->getTerminator()).destOpsMutable();
}

namespace {
struct BlockArgumentToSelectPass
    : public llhd::BlockArgumentToSelectBase<BlockArgumentToSelectPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void BlockArgumentToSelectPass::runOnOperation() {
  llhd::ProcOp proc = getOperation();
  DominanceInfo dom(proc);
  for (Block &block : proc.getBlocks()) {
    if (block.getNumArguments() == 0)
      continue;

    // Find the nearest common dominator of all predecessors.
    // If a block dominates all predecessors of a block, it also dominates this
    // block
    Block *domBlock = &block;
    for (Block *pred : block.getPredecessors()) {
      domBlock = dom.findNearestCommonDominator(domBlock, pred);
    }

    OpBuilder builder(proc);
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
        llhd::Dnf dnf =
            *llhd::getBooleanExprFromSourceToTarget(domBlock, predBlock);
        finalValue = dnf.buildOperations(builder);
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
          valToMux[i] = builder.create<SelectOp>(proc.getLoc(), finalValue, arg,
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
mlir::llhd::createBlockArgumentToSelectPass() {
  return std::make_unique<BlockArgumentToSelectPass>();
}
