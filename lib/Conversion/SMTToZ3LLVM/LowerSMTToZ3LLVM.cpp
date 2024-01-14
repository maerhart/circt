//===- LowerSMTToZ3LLVM.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/SMTToZ3LLVM.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-smt-to-z3-llvm"

using namespace mlir;
using namespace circt;
using namespace smt;

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

template <typename OpTy>
class SMTLoweringPattern : public OpConversionPattern<OpTy> {
public:
  SMTLoweringPattern(const TypeConverter &typeConverter, MLIRContext *context,
                     DenseMap<StringRef, LLVM::LLVMFuncOp> &funcMap,
                     DenseMap<Block *, Value> &ctxCache)
      : OpConversionPattern<OpTy>(typeConverter, context), funcMap(funcMap),
        ctxCache(ctxCache) {}

protected:
  Value buildAPICallGetPtr(OpBuilder &builder, Location loc, StringRef name,
                           ValueRange args) const {
    return *buildAPICall(builder, loc, name,
                         LLVM::LLVMFunctionType::get(
                             LLVM::LLVMPointerType::get(builder.getContext()),
                             SmallVector<Type>(args.getTypes())),
                         args);
  }

  std::optional<Value> buildAPICall(OpBuilder &builder, Location loc,
                                    StringRef name,
                                    LLVM::LLVMFunctionType funcType,
                                    ValueRange args) const {
    LLVM::CallOp callOp = buildAPICallGetOp(builder, loc, name, funcType, args);
    if (isa<LLVM::LLVMVoidType>(funcType.getReturnType()))
      return std::nullopt;
    return callOp->getResult(0);
  }

  LLVM::CallOp buildAPICallGetVoid(OpBuilder &builder, Location loc,
                                   StringRef name, ValueRange args) const {
    return buildAPICallGetOp(builder, loc, name,
                             LLVM::LLVMFunctionType::get(
                                 LLVM::LLVMVoidType::get(builder.getContext()),
                                 SmallVector<Type>(args.getTypes())),
                             args);
  }

  Value buildZ3ContextPtr(OpBuilder &builder, Location loc) const {
    Block *block = builder.getBlock();
    if (!ctxCache.contains(block)) {
      Value globalAddr = builder.create<LLVM::AddressOfOp>(
          loc, LLVM::LLVMPointerType::get(builder.getContext()), "ctx");
      ctxCache[block] = builder.create<LLVM::LoadOp>(
          loc, LLVM::LLVMPointerType::get(builder.getContext()), globalAddr);
    }

    return ctxCache[block];
  }

  Value buildStringRef(OpBuilder &builder, Location loc, StringRef str) const {
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(
        builder.getBlock()->getParent()->getParentOfType<ModuleOp>().getBody());
    auto arrayTy =
        LLVM::LLVMArrayType::get(builder.getI8Type(), str.size() + 1);
    auto globalStr = builder.create<LLVM::GlobalOp>(
        loc, arrayTy, true, LLVM::linkage::Linkage::Private, str,
        StringAttr::get(builder.getContext(), Twine(str).concat(Twine('\00'))));
    builder.restoreInsertionPoint(ip);
    return builder.create<LLVM::AddressOfOp>(loc, globalStr);
  }

private:
  LLVM::CallOp buildAPICallGetOp(OpBuilder &builder, Location loc,
                                 StringRef name,
                                 LLVM::LLVMFunctionType funcType,
                                 ValueRange args) const {
    if (!funcMap.contains(name)) {
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPointToEnd(builder.getBlock()
                                         ->getParent()
                                         ->getParentOfType<ModuleOp>()
                                         .getBody());
      funcMap[name] = builder.create<LLVM::LLVMFuncOp>(loc, name, funcType);
      builder.restoreInsertionPoint(ip);
    }
    auto funcOp = funcMap[name];
    return builder.create<LLVM::CallOp>(loc, funcOp, args);
  }

private:
  DenseMap<StringRef, LLVM::LLVMFuncOp> &funcMap;
  DenseMap<Block *, Value> &ctxCache;
};

struct DeclareConstOpLowering : public SMTLoweringPattern<DeclareConstOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(DeclareConstOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value ctx = buildZ3ContextPtr(rewriter, op.getLoc());
    Value sort = TypeSwitch<Type, Value>(op.getType())
                     .Case<BitVectorType>([&](auto ty) {
                       return buildAPICallGetPtr(
                           rewriter, op.getLoc(), "Z3_mk_bv_sort",
                           {ctx, rewriter.create<LLVM::ConstantOp>(
                                     op.getLoc(), rewriter.getI32Type(),
                                     ty.getWidth())});
                     })
                     .Case<BoolType>([&](auto ty) {
                       return buildAPICallGetPtr(rewriter, op.getLoc(),
                                                 "Z3_mk_bool_sort", ctx);
                     })
                     .Default([](auto ty) { return Value(); });

    if (!sort)
      return failure(); // TODO: error message

    Value sym = buildAPICallGetPtr(
        rewriter, op.getLoc(), "Z3_mk_string_symbol",
        {ctx, buildStringRef(rewriter, op.getLoc(), adaptor.getDeclName())});
    Value constDecl = buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_const",
                                         {ctx, sym, sort});
    rewriter.replaceOp(op, constDecl);

    return success();
  }
};

struct ConstantOpLowering : public SMTLoweringPattern<smt::ConstantOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(smt::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value ctx = buildZ3ContextPtr(rewriter, op.getLoc());
    unsigned width = op.getType().getWidth();
    Value bvWidth = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), width);
    Value constOne = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), 1);
    Type arrTy = LLVM::LLVMArrayType::get(rewriter.getI8Type(), width);
    Value alloca = rewriter.create<LLVM::AllocaOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getContext()), arrTy,
        constOne);
    Value array = rewriter.create<LLVM::UndefOp>(op.getLoc(), arrTy);
    unsigned val = adaptor.getValue().getValue();
    for (unsigned i = 0; i < width; ++i) {
      unsigned bitVal = (val >> i) & 1;
      Value bit = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI8Type(), bitVal);
      array = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), array, bit, ArrayRef<int64_t>{(int64_t)i});
    }

    rewriter.create<LLVM::StoreOp>(op.getLoc(), array, alloca);
    Value bvNumeral = buildAPICallGetPtr(
        rewriter, op.getLoc(), "Z3_mk_bv_numeral", {ctx, bvWidth, alloca});
    rewriter.replaceOp(op, bvNumeral);
    return success();
  }
};

template <typename SourceTy, typename TargetName>
class OneToOneSMTPattern : public SMTLoweringPattern<SourceTy> {
  using SMTLoweringPattern<SourceTy>::SMTLoweringPattern;
  using OpAdaptor = typename SMTLoweringPattern<SourceTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(SourceTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    SmallVector<Value> args;
    args.push_back(
        SMTLoweringPattern<SourceTy>::buildZ3ContextPtr(rewriter, op.getLoc()));
    args.append(SmallVector<Value>(adaptor.getOperands()));
    rewriter.replaceOp(op, SMTLoweringPattern<SourceTy>::buildAPICallGetPtr(
                               rewriter, op.getLoc(), TargetName::get(), args));
    return success();
  }
};

template <typename SourceTy, typename TargetName>
class VariadicSMTPattern : public SMTLoweringPattern<SourceTy> {
  using SMTLoweringPattern<SourceTy>::SMTLoweringPattern;
  using OpAdaptor = typename SMTLoweringPattern<SourceTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(SourceTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    Value numOperands = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), op->getNumOperands());
    Value constOne = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), 1);
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type arrTy = LLVM::LLVMArrayType::get(ptrTy, op->getNumOperands());
    Value storage =
        rewriter.create<LLVM::AllocaOp>(op.getLoc(), ptrTy, arrTy, constOne);
    Value array = rewriter.create<LLVM::UndefOp>(op.getLoc(), arrTy);

    for (auto [i, operand] : llvm::enumerate(adaptor.getOperands()))
      array = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), array, operand, ArrayRef<int64_t>{(int64_t)i});

    rewriter.create<LLVM::StoreOp>(op.getLoc(), array, storage);

    rewriter.replaceOp(op, SMTLoweringPattern<SourceTy>::buildAPICallGetPtr(
                               rewriter, op.getLoc(), TargetName::get(),
                               {SMTLoweringPattern<SourceTy>::buildZ3ContextPtr(
                                    rewriter, op.getLoc()),
                                numOperands, storage}));
    return success();
  }
};

struct BVCmpOpLowering : public SMTLoweringPattern<BVCmpOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(BVCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto getFuncName = [](Predicate pred) -> StringRef {
      switch (pred) {
      case Predicate::slt:
        return "Z3_mk_bvslt";
      case Predicate::sle:
        return "Z3_mk_bvsle";
      case Predicate::sgt:
        return "Z3_mk_bvsgt";
      case Predicate::sge:
        return "Z3_mk_bvsge";
      case Predicate::ult:
        return "Z3_mk_bvult";
      case Predicate::ule:
        return "Z3_mk_bvule";
      case Predicate::ugt:
        return "Z3_mk_bvugt";
      case Predicate::uge:
        return "Z3_mk_bvuge";
      }
    };
    StringRef funcName = getFuncName(op.getPred());

    rewriter.replaceOp(
        op, buildAPICallGetPtr(rewriter, op.getLoc(), funcName,
                               {buildZ3ContextPtr(rewriter, op.getLoc()),
                                adaptor.getLhs(), adaptor.getRhs()}));
    return success();
  }
};

struct SolverCreateOpLowering : public SMTLoweringPattern<SolverCreateOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(SolverCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    Value solver = buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_solver",
                                      buildZ3ContextPtr(rewriter, op.getLoc()));
    rewriter.replaceOp(op, solver);
    return success();
  }
};

struct AssertOpLowering : public SMTLoweringPattern<AssertOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    buildAPICallGetVoid(rewriter, op.getLoc(), "Z3_solver_assert",
                        {buildZ3ContextPtr(rewriter, op.getLoc()),
                         adaptor.getSolver(), adaptor.getInput()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct CheckSatOpLowering : public SMTLoweringPattern<CheckSatOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(CheckSatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    Value checkResult = *buildAPICall(
        rewriter, op.getLoc(), "Z3_solver_check",
        LLVM::LLVMFunctionType::get(
            rewriter.getI32Type(),
            {LLVM::LLVMPointerType::get(rewriter.getContext()),
             LLVM::LLVMPointerType::get(rewriter.getContext())}),
        {buildZ3ContextPtr(rewriter, op.getLoc()), adaptor.getSolver()});
    rewriter.replaceOp(op, checkResult);
    return success();
  }
};

struct RepeatOpLowering : public SMTLoweringPattern<RepeatOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(RepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    Value count = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), adaptor.getCount());
    rewriter.replaceOp(
        op, buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_repeat",
                               {buildZ3ContextPtr(rewriter, op.getLoc()), count,
                                adaptor.getInput()}));
    return success();
  }
};

struct ExtractOpLowering : public SMTLoweringPattern<ExtractOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    Value low = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), adaptor.getStart());
    Value high = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(),
        adaptor.getStart() + op.getType().getWidth() - 1);
    rewriter.replaceOp(
        op, buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_extract",
                               {buildZ3ContextPtr(rewriter, op.getLoc()), high,
                                low, adaptor.getInput()}));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerSMTToZ3LLVMPass
    : public LowerSMTToZ3LLVMBase<LowerSMTToZ3LLVMPass> {
  void runOnOperation() override;
};
} // namespace

void LowerSMTToZ3LLVMPass::runOnOperation() {
  Namespace globals;
  SymbolCache cache;
  cache.addDefinitions(getOperation());
  globals.add(cache);
  DenseMap<StringRef, LLVM::LLVMFuncOp> funcMap;
  DenseMap<Block *, Value> ctxCache;

  LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  // Setup the arc dialect type conversion.
  LLVMTypeConverter converter(&getContext());
  converter.addConversion([&](BoolType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](BitVectorType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](SolverType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  RewritePatternSet patterns(&getContext());
  populateFuncToLLVMConversionPatterns(converter, patterns);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);

#define ADD_OTO_PATTERN(OPTYPE, FUNCNAME)                                      \
  struct OPTYPE##Name {                                                        \
    static StringRef get() { return FUNCNAME; }                                \
  };                                                                           \
  patterns.add<OneToOneSMTPattern<OPTYPE, OPTYPE##Name>>(                      \
      converter, &getContext(), funcMap, ctxCache);

#define ADD_VARIADIC_PATTERN(OPTYPE, FUNCNAME)                                 \
  struct OPTYPE##Name {                                                        \
    static StringRef get() { return FUNCNAME; }                                \
  };                                                                           \
  patterns.add<VariadicSMTPattern<OPTYPE, OPTYPE##Name>>(                      \
      converter, &getContext(), funcMap, ctxCache);

  ADD_OTO_PATTERN(NegOp, "Z3_mk_bvneg");
  ADD_OTO_PATTERN(AddOp, "Z3_mk_bvadd");
  ADD_OTO_PATTERN(SubOp, "Z3_mk_bvsub");
  ADD_OTO_PATTERN(MulOp, "Z3_mk_bvmul");
  ADD_OTO_PATTERN(URemOp, "Z3_mk_bvurem");
  ADD_OTO_PATTERN(SRemOp, "Z3_mk_bvsrem");
  ADD_OTO_PATTERN(UModOp, "Z3_mk_bvumod");
  ADD_OTO_PATTERN(SModOp, "Z3_mk_bvsmod");
  ADD_OTO_PATTERN(UDivOp, "Z3_mk_bvudiv");
  ADD_OTO_PATTERN(SDivOp, "Z3_mk_bvsdiv");
  ADD_OTO_PATTERN(ShlOp, "Z3_mk_bvshl");
  ADD_OTO_PATTERN(LShrOp, "Z3_mk_bvlshr");
  ADD_OTO_PATTERN(AShrOp, "Z3_mk_bvashr");

  ADD_OTO_PATTERN(BVNotOp, "Z3_mk_bvnot");
  ADD_OTO_PATTERN(BVAndOp, "Z3_mk_bvand");
  ADD_OTO_PATTERN(BVOrOp, "Z3_mk_bvor");
  ADD_OTO_PATTERN(BVXOrOp, "Z3_mk_bvxor");
  ADD_OTO_PATTERN(BVNAndOp, "Z3_mk_bvnand");
  ADD_OTO_PATTERN(BVNOrOp, "Z3_mk_bvnor");
  ADD_OTO_PATTERN(BVXNOrOp, "Z3_mk_bvxnor");

  ADD_OTO_PATTERN(ConcatOp, "Z3_mk_concat");
  ADD_OTO_PATTERN(EqOp, "Z3_mk_eq");
  ADD_OTO_PATTERN(IteOp, "Z3_mk_ite");

  ADD_OTO_PATTERN(NotOp, "Z3_mk_not");
  ADD_OTO_PATTERN(XOrOp, "Z3_mk_xor");
  ADD_OTO_PATTERN(ImpliesOp, "Z3_mk_implies");

  ADD_VARIADIC_PATTERN(DistinctOp, "Z3_mk_distinct");
  ADD_VARIADIC_PATTERN(AndOp, "Z3_mk_and");
  ADD_VARIADIC_PATTERN(OrOp, "Z3_mk_or");

  patterns.add<ConstantOpLowering, DeclareConstOpLowering, BVCmpOpLowering,
               AssertOpLowering, CheckSatOpLowering, SolverCreateOpLowering,
               RepeatOpLowering, ExtractOpLowering>(converter, &getContext(),
                                                    funcMap, ctxCache);

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> circt::createLowerSMTToZ3LLVMPass() {
  return std::make_unique<LowerSMTToZ3LLVMPass>();
}
