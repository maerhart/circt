//===- ConstructLEC.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_CONSTRUCTLEC
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// ConstructLEC pass
//===----------------------------------------------------------------------===//

namespace {
struct ConstructLECPass
    : public circt::impl::ConstructLECBase<ConstructLECPass> {
  ConstructLECPass() : circt::impl::ConstructLECBase<ConstructLECPass>() {
    options.areTriviallyNotEquivalent = [](Operation *op1,
                                           Operation *op2) -> LogicalResult {
      auto moduleA = dyn_cast<HWModuleOp>(op1);
      auto moduleB = dyn_cast<HWModuleOp>(op2);
      if (!moduleA || !moduleB)
        return failure();

      if (moduleA.getModuleType() != moduleB.getModuleType())
        return moduleA.emitError(
                   "module's IO types don't match second modules: ")
               << moduleA.getModuleType() << " vs " << moduleB.getModuleType();

      return success();
    };
  }
  ConstructLECPass(const ConstructLECOptions &options)
      : circt::impl::ConstructLECBase<ConstructLECPass>(), options(options) {
    firstModule = options.firstModule;
    secondModule = options.secondModule;
    insertMainFunc = options.insertMainFunc;
  }

  void runOnOperation() override;
  Operation *lookupModule(StringRef name);

  ConstructLECOptions options;
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

Operation *ConstructLECPass::lookupModule(StringRef name) {
  Operation *expectedModule = SymbolTable::lookupNearestSymbolFrom(
      getOperation(), StringAttr::get(&getContext(), name));
  if (!expectedModule)
    getOperation().emitError("symbol '") << name << "' not found";

  return expectedModule;
}

void ConstructLECPass::runOnOperation() {
  // Update the options member
  options.firstModule = firstModule;
  options.secondModule = secondModule;
  options.insertMainFunc = insertMainFunc;

  // Create necessary function declarations and globals
  OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());
  Location loc = getOperation()->getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());

  // Lookup or declare printf function.
  auto printfFunc =
      LLVM::lookupOrCreateFn(getOperation(), "printf", ptrTy, voidTy, true);

  // Lookup the modules.
  auto *moduleA = lookupModule(firstModule);
  if (!moduleA)
    return signalPassFailure();
  auto *moduleB = lookupModule(secondModule);
  if (!moduleB)
    return signalPassFailure();

  if (failed(options.areTriviallyNotEquivalent(moduleA, moduleB)))
    return signalPassFailure();

  // Reuse the name of the first module for the entry function, so we don't have
  // to do any uniquing and the LEC driver also already knows this name.
  FunctionType functionType = FunctionType::get(&getContext(), {}, {});
  func::FuncOp entryFunc =
      builder.create<func::FuncOp>(loc, firstModule, functionType);

  if (insertMainFunc) {
    OpBuilder::InsertionGuard guard(builder);
    auto i32Ty = builder.getI32Type();
    auto mainFunc = builder.create<func::FuncOp>(
        loc, "main", builder.getFunctionType({i32Ty, ptrTy}, {i32Ty}));
    builder.createBlock(&mainFunc.getBody(), {}, {i32Ty, ptrTy}, {loc, loc});
    builder.create<func::CallOp>(loc, entryFunc, ValueRange{});
    // TODO: don't use LLVM here
    Value constZero = builder.create<LLVM::ConstantOp>(loc, i32Ty, 0);
    builder.create<func::ReturnOp>(loc, constZero);
  }

  builder.createBlock(&entryFunc.getBody());

  Value areEquivalent;
  if (moduleA == moduleB) {
    // Trivially equivalent
    areEquivalent =
        builder.create<LLVM::ConstantOp>(loc, builder.getI1Type(), 1);
    moduleA->erase();
  } else {
    auto lecOp = builder.create<verif::LogicEquivalenceCheckingOp>(loc);
    areEquivalent = lecOp.getAreEquivalent();
    auto *outputOpA = moduleA->getRegion(0).front().getTerminator();
    auto *outputOpB = moduleB->getRegion(0).front().getTerminator();
    lecOp.getFirstCircuit().takeBody(moduleA->getRegion(0));
    lecOp.getSecondCircuit().takeBody(moduleB->getRegion(0));

    moduleA->erase();
    moduleB->erase();

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(outputOpA);
      builder.create<verif::YieldOp>(loc, outputOpA->getOperands());
      outputOpA->erase();
      builder.setInsertionPoint(outputOpB);
      builder.create<verif::YieldOp>(loc, outputOpB->getOperands());
      outputOpB->erase();
    }

    sortTopologically(&lecOp.getFirstCircuit().front());
    sortTopologically(&lecOp.getSecondCircuit().front());
  }

  // TODO: we should find a more elegant way of reporting the result than
  // already inserting some LLVM here
  Value eqFormatString =
      lookupOrCreateStringGlobal(builder, getOperation(), "c1 == c2\n");
  Value neqFormatString =
      lookupOrCreateStringGlobal(builder, getOperation(), "c1 != c2\n");
  Value formatString = builder.create<LLVM::SelectOp>(
      loc, areEquivalent, eqFormatString, neqFormatString);
  builder.create<LLVM::CallOp>(loc, printfFunc, ValueRange{formatString});

  builder.create<func::ReturnOp>(loc, ValueRange{});
}

std::unique_ptr<Pass> circt::createConstructLEC() {
  return std::make_unique<ConstructLECPass>();
}

std::unique_ptr<Pass>
circt::createConstructLEC(const ConstructLECOptions &options) {
  return std::make_unique<ConstructLECPass>(options);
}
