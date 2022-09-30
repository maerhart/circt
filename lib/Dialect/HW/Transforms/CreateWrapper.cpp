//===- CreateWrapper.cpp - Create Wrapper Pass ----------------------------===//
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

#include <utility>

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/InteropOpInterfaces.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Create Wrapper Pass
//===----------------------------------------------------------------------===//

namespace {
struct CreateWrapperPass : public CreateWrapperBase<CreateWrapperPass> {
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<Pass> circt::hw::createWrapperPass() {
  return std::make_unique<CreateWrapperPass>();
}

static void createWrapperModule(HWModuleOp module) {
  OpBuilder builder(module);
  auto externModule = builder.create<HWModuleExternOp>(
      module->getLoc(), module.getNameAttr(), module.getPorts());
  module.setName((module.getName() + "Wrapper").str());
  module.getBodyBlock()->clear();
  builder.setInsertionPointToStart(module.getBodyBlock());
  auto instance = builder.create<InstanceOp>(
      module.getLoc(), externModule, externModule.getName(),
      SmallVector<Value>(module.getBodyBlock()->getArguments()));
  builder.create<OutputOp>(module.getLoc(), instance.getResults());
}

/// This is the main entrypoint for the HW to SystemC conversion pass.
void CreateWrapperPass::runOnOperation() {
  ModuleOp module = getOperation();

  module->walk([this](HWModuleOp hwModule) {
    if (hwModule.getName() == wrappedModuleName)
      createWrapperModule(hwModule);
    else
      hwModule.erase();

    return WalkResult::skip();
  });

  module->walk([](Operation *op) {
    if (isa<HWGeneratorSchemaOp, HWModuleGeneratedOp>(op))
      op->erase();

    return WalkResult::skip();
  });
}
