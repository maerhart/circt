//===- FooWires.cpp - Replace all wire names with foo ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Replace all wire names with foo.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Pass/Pass.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace circt {
namespace hw {
#define GEN_PASS_DEF_FOOWIRES
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
// A test pass that simply replaces all wire names with foo_<n>
struct FooWiresPass : circt::hw::impl::FooWiresBase<FooWiresPass> {
  void runOnOperation() override;
};
} // namespace

void FooWiresPass::runOnOperation() {
  getOperation().walk(
      [&](hw::WireOp wire) { // Walk over every wire in the module
        wire.getResult().replaceAllUsesWith(wire.getInput());
        if (auto *defOp = wire.getInput().getDefiningOp())
          if (!defOp->hasAttr("sv.namehint"))
            defOp->setAttr("sv.namehint", wire.getNameAttr());
        wire->erase();
      });
  getOperation().walk([&](mlir::func::FuncOp wire) { wire->erase(); });
}

std::unique_ptr<mlir::Pass> circt::hw::createFooWiresPass() {
  return std::make_unique<FooWiresPass>();
}
