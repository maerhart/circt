//===- AllocateState.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-allocate-state"

using namespace mlir;
using namespace circt;
using namespace arc;

using llvm::SmallMapVector;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct AllocateStatePass : public AllocateStateBase<AllocateStatePass> {
  void runOnOperation() override;
  void allocateBlock(
      Block *block, unsigned &currentByte,
      SmallVector<std::pair<Operation *, unsigned>> &allocOpsWithOffset);
  void allocateOp(Operation *op, Value storage, unsigned offset);
};
} // namespace

void AllocateStatePass::runOnOperation() {
  ModelOp modelOp = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Allocating state in `" << modelOp.getName()
                          << "`\n");

  SmallVector<std::pair<Operation *, unsigned>> allocOpsWithOffset;

  // Walk the blocks from innermost to outermost and group all state allocations
  // in that block in one larger allocation.
  unsigned currentByte = 0;
  modelOp.walk([&](Block *block) {
    allocateBlock(block, currentByte, allocOpsWithOffset);
  });

  auto storage = modelOp.getBodyBlock().addArgument(
      StorageType::get(&getContext(), currentByte), modelOp.getLoc());

  for (auto [op, offset] : allocOpsWithOffset)
    allocateOp(op, storage, offset);
}

static void computeOffsets(
    Operation *op, unsigned &currentByte,
    SmallVector<std::pair<Operation *, unsigned>> &allocOpsWithOffset) {
  // Helper function to allocate storage aligned to its own size, or 8 bytes at
  // most.
  // unsigned currentByte = 0;
  auto allocBytes = [&](unsigned numBytes) {
    currentByte = llvm::alignToPowerOf2(currentByte,
                                        llvm::bit_ceil(std::min(numBytes, 8U)));
    unsigned offset = currentByte;
    currentByte += numBytes;
    return offset;
  };

  // Allocate storage for the operations.
  // OpBuilder builder(block->getParentOp());
  if (isa<AllocStateOp, RootInputOp, RootOutputOp>(op)) {
    auto result = op->getResult(0);
    // auto storage = op->getOperand(0);
    auto intType = result.getType().cast<StateType>().getType();
    unsigned numBytes = (intType.getWidth() + 7) / 8;
    // auto offset = builder.getI32IntegerAttr(allocBytes(numBytes));
    // op->setAttr("offset", offset);
    allocOpsWithOffset.push_back({op, allocBytes(numBytes)});
    // gettersToCreate.emplace_back(result, storage, offset);
    return;
  }

  if (auto memOp = dyn_cast<AllocMemoryOp>(op)) {
    auto memType = memOp.getType();
    auto intType = memType.getWordType();
    unsigned stride = (intType.getWidth() + 7) / 8;
    stride =
        llvm::alignToPowerOf2(stride, llvm::bit_ceil(std::min(stride, 8U)));
    unsigned numBytes = memType.getNumWords() * stride;
    // auto offset = builder.getI32IntegerAttr(allocBytes(numBytes));
    // op->setAttr("offset", offset);
    // op->setAttr("stride", builder.getI32IntegerAttr(stride));
    memOp.getResult().setType(MemoryType::get(memOp.getContext(),
                                              memType.getNumWords(),
                                              memType.getWordType(), stride));

    allocOpsWithOffset.push_back({op, allocBytes(numBytes)});
    // gettersToCreate.emplace_back(memOp, memOp.getStorage(), offset);
    return;
  }

  if (auto allocStorageOp = dyn_cast<AllocStorageOp>(op)) {
    // auto offset = builder.getI32IntegerAttr(
    //     allocBytes(allocStorageOp.getType().getSize()));
    // allocStorageOp.setOffsetAttr(offset);
    // gettersToCreate.emplace_back(allocStorageOp, allocStorageOp.getInput(),
    //                               offset);
    allocOpsWithOffset.push_back(
        {op, allocBytes(allocStorageOp.getType().getSize())});
    return;
  }

  assert("unsupported op for allocation" && false);
}

void AllocateStatePass::allocateBlock(
    Block *block, unsigned &currentByte,
    SmallVector<std::pair<Operation *, unsigned>> &allocOpsWithOffset) {
  // Group operations by their storage. There is generally just one storage,
  // passed into the model as a block argument.
  for (auto &op : *block) {
    if (!isa<AllocStateOp, RootInputOp, RootOutputOp, AllocMemoryOp,
             AllocStorageOp>(&op))
      continue;

    computeOffsets(&op, currentByte, allocOpsWithOffset);
  }
  LLVM_DEBUG(llvm::dbgs() << "- Visiting block in "
                          << block->getParentOp()->getName() << "\n");
}

void AllocateStatePass::allocateOp(Operation *op, Value storage,
                                   unsigned offset) {
  // For every user of the alloc op, create a local `StorageGetOp`.
  // SmallVector<StorageGetOp> getters;
  // for (auto [result, storage, offset] : gettersToCreate) {
  SmallDenseMap<Block *, StorageGetOp> getterForBlock;
  Value result = op->getResult(0);
  for (auto *user : llvm::make_early_inc_range(result.getUsers())) {
    auto &getter = getterForBlock[user->getBlock()];
    // Create a local getter in front of each user, except for
    // `AllocStorageOp`s, for which we create a block-wider accessor.
    if (!getter || !result.getDefiningOp<AllocStorageOp>()) {
      ImplicitLocOpBuilder builder(result.getLoc(), user);
      getter = builder.create<StorageGetOp>(result.getType(), storage, offset);
      // getters.push_back(getter);
    } else if (user->isBeforeInBlock(getter)) {
      // TODO: This is a very expensive operation since us inserting
      // operations makes `isBeforeInBlock` re-enumerate the entire block
      // every single time. This doesn't happen often in practice since there
      // are relatively few `AllocStorageOp`s, but we should improve this in a
      // similar fashion as we did in the `LowerStates` pass.
      getter->moveBefore(user);
    }
    user->replaceUsesOfWith(result, getter);
  }
  op->erase();
  // }

  // Create the substorage accessor at the beginning of the block.
  // Operation *storageOwner = storage.getDefiningOp();
  // if (!storageOwner)
  //   storageOwner = storage.cast<BlockArgument>().getOwner()->getParentOp();

  // if (storageOwner->isProperAncestor(block->getParentOp())) {
  //   auto substorage = builder.create<AllocStorageOp>(
  //       block->getParentOp()->getLoc(),
  //       StorageType::get(&getContext(), currentByte), storage);
  //   for (auto *op : ops)
  //     op->replaceUsesOfWith(storage, substorage);
  //   for (auto op : getters)
  //     op->replaceUsesOfWith(storage, substorage);
  // } else {
  //   storage.setType(StorageType::get(&getContext(), currentByte));
  // }
}

std::unique_ptr<Pass> arc::createAllocateStatePass() {
  return std::make_unique<AllocateStatePass>();
}
