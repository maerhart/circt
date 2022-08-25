//===- HWToSystemC.cpp - HW To SystemC Conversion Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HW to SystemC Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LowerInterop.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace systemc;

//===----------------------------------------------------------------------===//
// HW to SystemC Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerInteropPass : public LowerInteropBase<LowerInteropPass> {
  void runOnOperation() override;
};
} // namespace

/// Create a HW to SystemC dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createLowerInteropPass() {
  return std::make_unique<LowerInteropPass>();
}

/// This is the main entrypoint for the HW to SystemC conversion pass.
void LowerInteropPass::runOnOperation() {
  ModuleOp module = getOperation();

  WalkResult result =
      module->walk([](InstanceProceduralInteropOpInterface op) -> WalkResult {
        auto parent = op->getParentOfType<ModuleProceduralInteropOpInterface>();
        InteropMechanism interopType;
        bool intersect = false;
        for (auto interop : op.getInteropSupport()) {
          for (auto pi : parent.getInteropSupport()) {
            if (interop == pi) {
              interopType = interop;
              intersect = true;
            }
          }
        }

        if (!intersect)
          return WalkResult::interrupt();

        auto builder = parent.getStateBuilder(interopType);
        SmallVector<Value> state = op.buildState(builder);
        builder = parent.getStateInitBuilder(interopType);
        op.buildStateInit(builder, state);
        builder = parent.getStateUpdateBuilder(op, interopType);
        SmallVector<Value> newValues = op.buildStateUpdate(builder, state);
        builder = parent.getStateDeallocBuilder(interopType);
        op.buildStateDealloc(builder, state);

        op->replaceAllUsesWith(newValues);
        op->erase();

        return WalkResult::advance();
      });

  if (result.wasInterrupted())
    signalPassFailure();
}
