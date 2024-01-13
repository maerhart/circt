//===- LowerToBMC.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Namespace.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_LOWERTOBMC
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// Convert Lower To BMC pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerToBMCPass : public circt::impl::LowerToBMCBase<LowerToBMCPass> {
  using LowerToBMCBase::LowerToBMCBase;
  void runOnOperation() override;
};
} // namespace

static Value lookupOrCreateStringGlobal(OpBuilder &builder, ModuleOp moduleOp,
                                        StringRef str) {
  Location loc = moduleOp.getLoc();
  auto global = moduleOp.lookupSymbol<LLVM::GlobalOp>(str);
  if (!global) {
    OpBuilder b = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto arrayTy = LLVM::LLVMArrayType::get(b.getI8Type(), str.size() + 1);
    global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/true, LLVM::linkage::Linkage::Private, str,
        StringAttr::get(b.getContext(), Twine(str).concat(Twine('\00'))));
  }

  // FIXME: sanity check the fetched global: do all the attributes match what
  // we expect?

  return builder.create<LLVM::AddressOfOp>(loc, global);
}

void LowerToBMCPass::runOnOperation() {
  Namespace names;

  // Fetch the 'hw.module' operation to model check.
  Operation *expectedModule = getOperation().lookupSymbol(topModule);
  if (!expectedModule) {
    getOperation().emitError("module named '") << topModule << "' not found";
    return signalPassFailure();
  }
  auto hwModule = dyn_cast<hw::HWModuleOp>(expectedModule);
  if (!hwModule) {
    expectedModule->emitError("must be a 'hw.module'");
    return signalPassFailure();
  }

  // Create necessary function declarations and globals
  OpBuilder builder(&getContext());
  Location loc = getOperation()->getLoc();
  builder.setInsertionPointToEnd(getOperation().getBody());
  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidTy = LLVM::LLVMVoidType::get(builder.getContext());

  // Lookup or declare printf function.
  auto printfFunc =
      LLVM::lookupOrCreateFn(getOperation(), "printf", ptrTy, voidTy, true);

  // Replace the top-module with a function performing the BMC
  Type i32Ty = builder.getI32Type();
  auto entryFunc = builder.create<func::FuncOp>(
      loc, topModule, builder.getFunctionType({}, {}));
  builder.createBlock(&entryFunc.getBody());

  {
    OpBuilder::InsertionGuard guard(builder);
    auto *terminator = hwModule.getBody().front().getTerminator();
    builder.setInsertionPoint(terminator);
    builder.create<verif::YieldOp>(loc, terminator->getOperands());
    terminator->erase();
  }

  auto bmcOp = builder.create<verif::BMCOp>(loc, bound);
  bmcOp->setAttr("num_regs", hwModule->getAttr("num_regs"));
  bmcOp.getCircuit().takeBody(hwModule.getBody());
  hwModule->erase();

  auto successString =
      lookupOrCreateStringGlobal(builder, getOperation(), "Bound reached!\n");
  auto failureString = lookupOrCreateStringGlobal(
      builder, getOperation(), "Assertion can be violated!\n");
  auto formatString = builder.create<LLVM::SelectOp>(
      loc, bmcOp.getResult(), successString, failureString);
  builder.create<LLVM::CallOp>(loc, printfFunc, ValueRange{formatString});
  builder.create<func::ReturnOp>(loc);

  if (insertMainFunc) {
    builder.setInsertionPointToEnd(getOperation().getBody());
    auto mainFunc = builder.create<func::FuncOp>(
        loc, "main", builder.getFunctionType({i32Ty, ptrTy}, {i32Ty}));
    builder.createBlock(&mainFunc.getBody(), {}, {i32Ty, ptrTy}, {loc, loc});
    builder.create<func::CallOp>(loc, entryFunc, ValueRange{});
    // TODO: don't use LLVM here
    Value constZero = builder.create<LLVM::ConstantOp>(loc, i32Ty, 0);
    builder.create<func::ReturnOp>(loc, constZero);
  }
}
