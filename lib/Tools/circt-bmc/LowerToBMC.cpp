//===- LowerToBMC.cpp -----------------------------------------------------===//
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
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Namespace.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_LOWERTOBMC
#include "circt/Tools/circt-bmc/Passes.h.inc"
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
    SmallVector<Type> inputTypes;
    inputTypes.push_back(smt::SolverType::get(getContext()));
    inputTypes.append(op.getInputTypes());
    FunctionType functionType =
        FunctionType::get(getContext(), inputTypes, op.getOutputTypes());
    func::FuncOp funcOp = rewriter.create<func::FuncOp>(
        op.getLoc(), adaptor.getSymName(), functionType);
    funcOp.getBody().takeBody(adaptor.getBody());
    funcOp.getBody().front().insertArgument(
        (unsigned)0, smt::SolverType::get(getContext()), op.getLoc());
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

struct VerifAssertOpConversion : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value solver = op->getParentOfType<func::FuncOp>()
                       .getFunctionBody()
                       .front()
                       .getArgument(0);
    Value constOne = rewriter.create<smt::ConstantOp>(
        op.getLoc(),
        smt::BitVectorAttr::get(getContext(), 0,
                                smt::BitVectorType::get(getContext(), 1)));
    Value cond = rewriter.create<smt::EqOp>(op.getLoc(), adaptor.getProperty(),
                                            constOne);
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, solver, cond);
    return success();
  }
};

struct FromClockOpConversion : OpConversionPattern<seq::FromClockOp> {
  using OpConversionPattern<seq::FromClockOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(seq::FromClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Lower To BMC pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerToBMCPass : public circt::impl::LowerToBMCBase<LowerToBMCPass> {
  using LowerToBMCBase::LowerToBMCBase;
  void runOnOperation() override;
};
} // namespace

static bool hasOnlySMTTypes(FunctionType type) {
  for (Type type : type.getInputs())
    if (!isa<smt::BitVectorType, smt::BoolType, smt::SolverType>(type))
      return false;
  for (Type type : type.getResults())
    if (!isa<smt::BitVectorType, smt::BoolType, smt::SolverType>(type))
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

void LowerToBMCPass::runOnOperation() {
  Namespace names;

  hw::HWModuleOp hwModule;
  Operation *expectedFunc = SymbolTable::lookupNearestSymbolFrom(
      getOperation(), StringAttr::get(&getContext(), topModule));
  if (!expectedFunc || !isa<hw::HWModuleOp>(expectedFunc)) {
    getOperation().emitError("module named '") << topModule << "' not found";
    return signalPassFailure();
  }
  hwModule = cast<hw::HWModuleOp>(expectedFunc);

  for (auto &op : llvm::make_early_inc_range(*getOperation().getBody())) {
    if (isa<hw::HWModuleOp>(&op) && &op != hwModule.getOperation())
      op.erase();
  }

  unsigned initialNumInputPorts = hwModule.getNumInputPorts();

  SmallVector<seq::CompRegOp> regs(
      hwModule.getBodyBlock()->getOps<seq::CompRegOp>());
  SmallVector<ModulePort> ports;
  SmallVector<Type> newInputs;

  BitVector clockMask(hwModule.getNumInputPorts());

  unsigned i = 0;
  for (auto port : hwModule.getPortList()) {
    ports.emplace_back(port);
    if (port.isInput()) {
      if (isa<seq::ClockType>(port.type))
        clockMask.set(i);
      ++i;
    }
  }

  for (auto reg : regs) {
    ModulePort inPort, outPort;
    inPort.dir = circt::hw::ModulePort::Input;
    outPort.dir = circt::hw::ModulePort::Output;
    inPort.type = reg.getType();
    newInputs.emplace_back(reg.getType());
    outPort.type = reg.getType();
    ports.emplace_back(inPort);
    ports.emplace_back(outPort);
  }

  hwModule.setModuleType(ModuleType::get(&getContext(), ports));
  auto newArgs = hwModule.getBodyBlock()->addArguments(
      newInputs, SmallVector<Location>(newInputs.size(), hwModule->getLoc()));

  SmallVector<Value> outputs(
      hwModule.getBodyBlock()->getTerminator()->getOperands());
  for (auto [reg, arg] : llvm::zip(regs, newArgs)) {
    outputs.emplace_back(reg.getInput());
    reg.getResult().replaceAllUsesWith(arg);
    reg->erase();
  }
  hwModule.getBodyBlock()->getTerminator()->setOperands(outputs);

  ConversionTarget target(getContext());
  target.addIllegalDialect<HWDialect>();
  target.addIllegalDialect<comb::CombDialect>();
  target.addIllegalDialect<verif::VerifDialect>();
  target.addLegalDialect<smt::SMTDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>(
      [](func::FuncOp op) { return hasOnlySMTTypes(op.getFunctionType()); });

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  converter.addConversion([](smt::SolverType type) { return type; });
  converter.addConversion([](seq::ClockType type) {
    return smt::BitVectorType::get(type.getContext(), 1);
  });
  patterns.add<ConstantOpConversion, HWModuleOpConversion, OutputOpConversion,
               VerifAssertOpConversion, FromClockOpConversion>(
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
  auto entryFunc = builder.create<func::FuncOp>(
      loc, "entry", builder.getFunctionType({}, {}));
  Block *block = entryFunc.addEntryBlock();
  builder.setInsertionPointToStart(block);
  Value config = builder.create<LLVM::CallOp>(loc, mkConfigFunc, ValueRange{})
                     ->getResult(0);
  Value ctx =
      builder.create<LLVM::CallOp>(loc, mkCtxFunc, config)->getResult(0);
  Value addrOfCtxGlobal = builder.create<LLVM::AddressOfOp>(loc, ctxGlobal);
  builder.create<LLVM::StoreOp>(loc, ctx, addrOfCtxGlobal, 8);
  builder.create<LLVM::CallOp>(loc, delConfigFunc, config);

  func::FuncOp topFunc;
  expectedFunc = SymbolTable::lookupNearestSymbolFrom(
      getOperation(), StringAttr::get(&getContext(), topModule));
  if (!expectedFunc || !isa<func::FuncOp>(expectedFunc)) {
    getOperation().emitError("func named '") << topModule << "' not found";
    return signalPassFailure();
  }
  topFunc = cast<func::FuncOp>(expectedFunc);

  SmallVector<Value> inputDecls;
  for (auto [i, type] :
       llvm::enumerate(topFunc.getFunctionType().getInputs().drop_front())) {
    if (i < initialNumInputPorts && clockMask[i])
      inputDecls.push_back(builder.create<smt::ConstantOp>(
          loc,
          smt::BitVectorAttr::get(&getContext(), 0,
                                  smt::BitVectorType::get(&getContext(), 1))));
    else
      inputDecls.push_back(builder.create<smt::DeclareConstOp>(
          loc, type, names.newName("const")));
  }

  Value solver = builder.create<smt::SolverCreateOp>(loc, "solver");
  Value lowerBound =
      builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), 0);
  Value step = builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), 1);
  Value upperBound =
      builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), bound);
  Value formatString = buildStringRef(builder, loc, "Assertion is reachable\n");
  builder.create<scf::ForOp>(
      loc, lowerBound, upperBound, step, inputDecls,
      [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
        SmallVector<Value> callArgs;
        callArgs.push_back(solver);
        callArgs.append(SmallVector<Value>(iterArgs));
        auto topFuncCall = builder.create<func::CallOp>(loc, topFunc, callArgs);
        Value res = builder.create<smt::CheckSatOp>(loc, solver);
        Value constOne =
            builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), 1);
        Value violated = builder.create<LLVM::ICmpOp>(
            loc, LLVM::ICmpPredicate::eq, res, constOne);
        builder.create<scf::IfOp>(
            loc, violated, [&](OpBuilder &builder, Location loc) {
              builder.create<LLVM::CallOp>(loc, printfFunc,
                                           ValueRange{formatString});
              builder.create<scf::YieldOp>(loc);
            });

        SmallVector<Value> newDecls;
        for (auto [i, type] : llvm::enumerate(
                 topFunc.getFunctionType().getInputs().drop_front().take_front(
                     initialNumInputPorts))) {
          if (clockMask[i])
            newDecls.push_back(builder.create<smt::ConstantOp>(
                loc, smt::BitVectorAttr::get(
                         &getContext(), 1,
                         smt::BitVectorType::get(&getContext(), 1))));
          else
            newDecls.push_back(builder.create<smt::DeclareConstOp>(
                loc, type, names.newName("arg")));
        }

        SmallVector<Value> regOutputExprs(topFuncCall->getResults());
        newDecls.append(SmallVector<Value>(
            ArrayRef<Value>(regOutputExprs).take_back(regs.size())));

        callArgs.clear();
        callArgs.push_back(solver);
        callArgs.append(SmallVector<Value>(newDecls));
        topFuncCall = builder.create<func::CallOp>(loc, topFunc, callArgs);
        res = builder.create<smt::CheckSatOp>(loc, solver);
        constOne =
            builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), 1);
        violated = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                res, constOne);
        builder.create<scf::IfOp>(
            loc, violated, [&](OpBuilder &builder, Location loc) {
              builder.create<LLVM::CallOp>(loc, printfFunc,
                                           ValueRange{formatString});
              builder.create<scf::YieldOp>(loc);
            });

        newDecls.clear();
        for (auto [i, type] : llvm::enumerate(
                 topFunc.getFunctionType().getInputs().drop_front().take_front(
                     initialNumInputPorts))) {
          if (clockMask[i])
            newDecls.push_back(builder.create<smt::ConstantOp>(
                loc, smt::BitVectorAttr::get(
                         &getContext(), 0,
                         smt::BitVectorType::get(&getContext(), 1))));
          else
            newDecls.push_back(builder.create<smt::DeclareConstOp>(
                loc, type, names.newName("arg")));
        }

        regOutputExprs = topFuncCall->getResults();
        newDecls.append(SmallVector<Value>(
            ArrayRef<Value>(regOutputExprs).take_back(regs.size())));

        builder.create<scf::YieldOp>(loc, newDecls);
      });

  formatString = buildStringRef(builder, loc, "Bound reached!\n");
  builder.create<LLVM::CallOp>(loc, printfFunc, ValueRange{formatString});
  builder.create<func::ReturnOp>(loc);

  builder.setInsertionPointAfter(entryFunc);
  auto mainFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, "main", LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, ptrTy}));
  Block *entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  builder.create<func::CallOp>(loc, entryFunc, ValueRange{});
  Value constZero = builder.create<LLVM::ConstantOp>(loc, i32Ty, 0);
  builder.create<LLVM::ReturnOp>(loc, constZero);
}
