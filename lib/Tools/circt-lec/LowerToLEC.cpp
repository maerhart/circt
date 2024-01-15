//===- LowerToLEC.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Conversion/CombToSMT.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_LOWERTOLEC
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a hw::ConstantOp operation to smt::ConstantOp
struct ConstantOpConversion : OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: support IntType return type of hw.constant
    rewriter.replaceOpWithNewOp<smt::ConstantOp>(
        op, smt::BitVectorAttr::get(
                getContext(), adaptor.getValue().getZExtValue(),
                smt::BitVectorType::get(getContext(),
                                        op.getType().getIntOrFloatBitWidth())));
    return success();
  }
};

/// Lower a hw::HWModuleOp operation to func::FuncOp
struct HWModuleOpConversion : OpConversionPattern<HWModuleOp> {
  using OpConversionPattern<HWModuleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HWModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: support inout ports
    FunctionType functionType = FunctionType::get(
        getContext(), op.getInputTypes(), op.getOutputTypes());
    func::FuncOp funcOp = rewriter.create<func::FuncOp>(
        op.getLoc(), adaptor.getSymName(), functionType);
    funcOp.getBody().takeBody(adaptor.getBody());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower a hw::OutputOp operation to func::ReturnOp
struct OutputOpConversion : OpConversionPattern<OutputOp> {
  using OpConversionPattern<OutputOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOutputs());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Lower To LEC pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerToLECPass : public circt::impl::LowerToLECBase<LowerToLECPass> {
  using LowerToLECBase::LowerToLECBase;
  void runOnOperation() override;
};
} // namespace

static bool hasOnlySMTTypes(FunctionType type) {
  for (Type type : type.getInputs())
    if (!isa<smt::BitVectorType, smt::BoolType>(type))
      return false;
  for (Type type : type.getResults())
    if (!isa<smt::BitVectorType, smt::BoolType>(type))
      return false;
  return true;
}

static Value buildStringRef(OpBuilder &builder, Location loc, StringRef str) {
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToEnd(
      builder.getBlock()->getParent()->getParentOfType<ModuleOp>().getBody());
  auto arrayTy = LLVM::LLVMArrayType::get(builder.getI8Type(), str.size() + 1);
  auto globalStr = builder.create<LLVM::GlobalOp>(
      loc, arrayTy, true, LLVM::linkage::Linkage::Private, str,
      StringAttr::get(builder.getContext(), Twine(str).concat(Twine('\00'))));
  builder.restoreInsertionPoint(ip);
  return builder.create<LLVM::AddressOfOp>(loc, globalStr);
}

void LowerToLECPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<HWDialect>();
  target.addIllegalDialect<comb::CombDialect>();
  target.addLegalDialect<smt::SMTDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>(
      [](func::FuncOp op) { return hasOnlySMTTypes(op.getFunctionType()); });

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  converter.addConversion([](IntegerType type) {
    return smt::BitVectorType::get(type.getContext(), type.getWidth());
  });
  patterns.add<ConstantOpConversion, HWModuleOpConversion, OutputOpConversion>(
      converter, patterns.getContext());

  populateCombToSMTConversionPatterns(converter, patterns);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();

  // Create necessary function declarations and globals
  OpBuilder builder(&getContext());
  Location loc = getOperation()->getLoc();
  builder.setInsertionPointToEnd(getOperation().getBody());
  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());

  // config functions
  auto mkConfigFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, "Z3_mk_config", LLVM::LLVMFunctionType::get(ptrTy, {}));
  auto delConfigFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, "Z3_del_config",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&getContext()),
                                  ptrTy));

  // context function
  auto mkCtxFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, "Z3_mk_context", LLVM::LLVMFunctionType::get(ptrTy, ptrTy));

  // printf
  auto printfFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, "printf",
      LLVM::LLVMFunctionType::get(builder.getI32Type(),
                                  LLVM::LLVMPointerType::get(&getContext()),
                                  true));

  // create global context
  auto ctxGlobal = builder.create<LLVM::GlobalOp>(
      loc, ptrTy, false, LLVM::linkage::Linkage::Private, "ctx", Attribute{});
  auto ip = builder.saveInsertionPoint();
  Block *initBlock = builder.createBlock(&ctxGlobal.getInitializerRegion());
  OpBuilder initBuilder = OpBuilder::atBlockBegin(initBlock);
  Value llvmZero = initBuilder.create<LLVM::ZeroOp>(loc, ptrTy);
  initBuilder.create<LLVM::ReturnOp>(loc, llvmZero);
  builder.restoreInsertionPoint(ip);

  // Insert main function
  Type i32Ty = builder.getI32Type();
  auto entryFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, "entry",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&getContext()), {}));
  Block *block = entryFunc.addEntryBlock();
  builder.setInsertionPointToStart(block);
  Value config = builder.create<LLVM::CallOp>(loc, mkConfigFunc, ValueRange{})
                     ->getResult(0);
  Value ctx =
      builder.create<LLVM::CallOp>(loc, mkCtxFunc, config)->getResult(0);
  Value addrOfCtxGlobal = builder.create<LLVM::AddressOfOp>(loc, ctxGlobal);
  builder.create<LLVM::StoreOp>(loc, ctx, addrOfCtxGlobal, 8);
  builder.create<LLVM::CallOp>(loc, delConfigFunc, config);

  func::FuncOp moduleA, moduleB;
  Operation *expectedFunc = SymbolTable::lookupNearestSymbolFrom(
      getOperation(), StringAttr::get(&getContext(), firstModule));
  if (!expectedFunc || !isa<func::FuncOp>(expectedFunc)) {
    getOperation().emitError("first module named '")
        << firstModule << "' not found";
    return signalPassFailure();
  }
  moduleA = cast<func::FuncOp>(expectedFunc);

  expectedFunc = SymbolTable::lookupNearestSymbolFrom(
      getOperation(), StringAttr::get(&getContext(), secondModule));
  if (!expectedFunc || !isa<func::FuncOp>(expectedFunc)) {
    getOperation().emitError("second module named '")
        << secondModule << "' not found";
    return signalPassFailure();
  }
  moduleB = cast<func::FuncOp>(expectedFunc);

  if (moduleA.getFunctionType() != moduleB.getFunctionType()) {
    getOperation().emitError("modules types don't match");
    return signalPassFailure();
  }

  if (moduleA.getFunctionType().getNumResults() == 0) {
    getOperation().emitError("trivially holds");
    return signalPassFailure(); // nothing to verify, trivially holds
  }

  SmallVector<Value> inputDecls;
  for (auto [i, type] : llvm::enumerate(moduleA.getFunctionType().getInputs()))
    inputDecls.push_back(builder.create<smt::DeclareConstOp>(
        loc, type, "arg" + std::to_string(i)));

  auto callA = builder.create<func::CallOp>(loc, moduleA, inputDecls);
  auto callB = builder.create<func::CallOp>(loc, moduleB, inputDecls);

  Value toAssert;
  for (auto [outA, outB] :
       llvm::zip(callA->getResults(), callB->getResults())) {
    Value dist = builder.create<smt::DistinctOp>(loc, ValueRange{outA, outB});
    if (toAssert)
      toAssert = builder.create<smt::OrOp>(loc, ValueRange{toAssert, dist});
    else
      toAssert = dist;
  }

  Value solver = builder.create<smt::SolverCreateOp>(loc, "solver");
  builder.create<smt::AssertOp>(loc, solver, toAssert);
  Value res = builder.create<smt::CheckSatOp>(loc, solver);

  Value eqFormatString = buildStringRef(builder, loc, "c1 == c2\n");
  Value neqFormatString = buildStringRef(builder, loc, "c1 != c2\n");
  Value constNeg1 =
      builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), -1);
  Value isEquivalent = builder.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::eq, res, constNeg1);
  Value formatString = builder.create<LLVM::SelectOp>(
      loc, isEquivalent, eqFormatString, neqFormatString);
  builder.create<LLVM::CallOp>(loc, printfFunc, ValueRange{formatString});

  builder.create<LLVM::ReturnOp>(loc, ValueRange{});

  builder.setInsertionPointAfter(entryFunc);
  auto mainFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, "main", LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, ptrTy}));
  Block *entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  builder.create<LLVM::CallOp>(loc, entryFunc, ValueRange{});
  Value constZero = builder.create<LLVM::ConstantOp>(loc, i32Ty, 0);
  builder.create<LLVM::ReturnOp>(loc, constZero);
}
