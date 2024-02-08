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
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
                     DenseMap<Block *, Value> &ctxCache,
                     DenseMap<StringRef, LLVM::GlobalOp> &stringCache)
      : OpConversionPattern<OpTy>(typeConverter, context), funcMap(funcMap),
        ctxCache(ctxCache), stringCache(stringCache) {}

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
    auto &global = stringCache[str];
    if (!global) {
      builder.setInsertionPointToEnd(builder.getBlock()
                                         ->getParent()
                                         ->getParentOfType<ModuleOp>()
                                         .getBody());
      auto arrayTy =
          LLVM::LLVMArrayType::get(builder.getI8Type(), str.size() + 1);
      global = builder.create<LLVM::GlobalOp>(
          loc, arrayTy, true, LLVM::linkage::Linkage::Private, str,
          StringAttr::get(builder.getContext(),
                          Twine(str).concat(Twine('\00'))));
      builder.restoreInsertionPoint(ip);
    }
    return builder.create<LLVM::AddressOfOp>(loc, global);
  }

  Value buildSort(OpBuilder &builder, Location loc, Value smtContext,
                  Type type) const {
    return TypeSwitch<Type, Value>(type)
        .Case([&](smt::IntegerType ty) {
          return buildAPICallGetPtr(builder, loc, "Z3_mk_int_sort",
                                    {smtContext});
        })
        .Case([&](smt::BitVectorType ty) {
          Value bitwidth = builder.create<LLVM::ConstantOp>(
              loc, builder.getI32Type(), ty.getWidth());
          return buildAPICallGetPtr(builder, loc, "Z3_mk_bv_sort",
                                    {smtContext, bitwidth});
        })
        .Case([&](smt::BoolType ty) {
          return buildAPICallGetPtr(builder, loc, "Z3_mk_bool_sort",
                                    {smtContext});
        })
        .Default([](auto ty) { return Value(); });
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
  DenseMap<StringRef, LLVM::GlobalOp> &stringCache;
};

struct DeclareConstOpLowering : public SMTLoweringPattern<DeclareConstOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(DeclareConstOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value ctx = buildZ3ContextPtr(rewriter, op.getLoc());
    Value sort = buildSort(rewriter, op.getLoc(), ctx, op.getType());
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

struct ArraySelectOpLowering : public SMTLoweringPattern<smt::ArraySelectOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(smt::ArraySelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(
        op, buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_select",
                               {buildZ3ContextPtr(rewriter, op.getLoc()),
                                adaptor.getArray(), adaptor.getIndex()}));
    return success();
  }
};

struct ArrayStoreOpLowering : public SMTLoweringPattern<smt::ArrayStoreOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(smt::ArrayStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(
        op, buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_store",
                               {buildZ3ContextPtr(rewriter, op.getLoc()),
                                adaptor.getArray(), adaptor.getIndex(),
                                adaptor.getValue()}));
    return success();
  }
};

struct ArrayBroadcastOpLowering
    : public SMTLoweringPattern<smt::ArrayBroadcastOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(smt::ArrayBroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto ctx = buildZ3ContextPtr(rewriter, op.getLoc());
    auto domainSort = buildSort(
        rewriter, op.getLoc(), ctx,
        cast<smt::ArrayType>(op.getResult().getType()).getDomainType());
    if (!domainSort)
      return failure();

    rewriter.replaceOp(
        op, buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_const_array",
                               {ctx, domainSort, adaptor.getValue()}));
    return success();
  }
};

struct ArrayDefaultOpLowering : public SMTLoweringPattern<smt::ArrayDefaultOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(smt::ArrayDefaultOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(
        op, buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_array_default",
                               {buildZ3ContextPtr(rewriter, op.getLoc()),
                                adaptor.getArray()}));
    return success();
  }
};

struct BoolConstantOpLowering : public SMTLoweringPattern<smt::BoolConstantOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(smt::BoolConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getValue()) {
      rewriter.replaceOp(
          op, buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_true",
                                 {buildZ3ContextPtr(rewriter, op.getLoc())}));
      return success();
    }

    rewriter.replaceOp(
        op, buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_false",
                               {buildZ3ContextPtr(rewriter, op.getLoc())}));
    return success();
  }
};

struct IntConstantOpLowering : public SMTLoweringPattern<smt::IntConstantOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(smt::IntConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value ctx = buildZ3ContextPtr(rewriter, op.getLoc());

    SmallVector<char> numeralString;
    adaptor.getValue().toStringSigned(numeralString);

    bool isNegative = false;
    if (numeralString[0] == '-') {
      numeralString =
          SmallVector<char>(ArrayRef<char>(numeralString).drop_front());
      isNegative = true;
    }

    std::string numeralStr;
    for (auto c : numeralString)
      numeralStr.push_back(c);

    Value numeral = buildStringRef(rewriter, op.getLoc(), numeralStr);

    Value type =
        buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_int_sort", {ctx});

    Value intNumeral = buildAPICallGetPtr(
        rewriter, op.getLoc(), "Z3_mk_numeral", {ctx, numeral, type});

    if (isNegative)
      intNumeral = buildAPICallGetPtr(rewriter, op.getLoc(),
                                      "Z3_mk_unary_minus", {ctx, intNumeral});
    rewriter.replaceOp(op, intNumeral);
    return success();
  }
};

struct IntCmpOpLowering : public SMTLoweringPattern<IntCmpOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(IntCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto getFuncName = [](IntPredicate pred) -> StringRef {
      switch (pred) {
      case IntPredicate::lt:
        return "Z3_mk_lt";
      case IntPredicate::le:
        return "Z3_mk_le";
      case IntPredicate::gt:
        return "Z3_mk_gt";
      case IntPredicate::ge:
        return "Z3_mk_ge";
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

struct PatternCreateOpLowering : public SMTLoweringPattern<PatternCreateOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(PatternCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto yieldOp = cast<smt::YieldOp>(op.getBody().front().getTerminator());
    Value ctx = buildZ3ContextPtr(rewriter, op.getLoc());
    unsigned numPatterns = yieldOp.getValues().size();

    rewriter.setInsertionPoint(yieldOp);

    Value numPatternsVal = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), numPatterns);

    Value constOne = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), 1);
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type arrTy = LLVM::LLVMArrayType::get(ptrTy, numPatterns);
    Value patterns =
        rewriter.create<LLVM::AllocaOp>(op.getLoc(), ptrTy, arrTy, constOne);
    Value array = rewriter.create<LLVM::UndefOp>(op.getLoc(), arrTy);

    for (auto [i, operand] : llvm::enumerate(yieldOp.getValues())) {
      Value convOperand = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(operand.getType()),
          operand);
      array = rewriter.create<LLVM::InsertValueOp>(
          loc, array, convOperand, ArrayRef<int64_t>{(int64_t)i});
    }

    rewriter.create<LLVM::StoreOp>(op.getLoc(), array, patterns);

    Value patternExp =
        buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_pattern",
                           {ctx, numPatternsVal, patterns});

    rewriter.eraseOp(yieldOp);
    rewriter.inlineBlockBefore(&op.getBody().front(), op);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOp(op, patternExp);
    return success();
  }
};

struct ForallOpLowering : public SMTLoweringPattern<ForallOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(ForallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value ctx = buildZ3ContextPtr(rewriter, op.getLoc());

    // Patterns
    unsigned numPatterns = adaptor.getPatterns().size();
    Value weight = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                     adaptor.getWeight());
    Value numPatternsVal = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), numPatterns);

    Value constOne = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), 1);
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type patternArrTy = LLVM::LLVMArrayType::get(ptrTy, numPatterns);
    Value patterns = rewriter.create<LLVM::AllocaOp>(op.getLoc(), ptrTy,
                                                     patternArrTy, constOne);
    Value array = rewriter.create<LLVM::UndefOp>(op.getLoc(), patternArrTy);

    for (auto [i, operand] : llvm::enumerate(adaptor.getPatterns()))
      array = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), array, operand, ArrayRef<int64_t>{(int64_t)i});

    rewriter.create<LLVM::StoreOp>(op.getLoc(), array, patterns);

    // Bound variables
    unsigned numDecls = adaptor.getBoundVarNames().size();
    Value numDeclsVal =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), numDecls);

    Type declArrTy = LLVM::LLVMArrayType::get(ptrTy, numDecls);
    Value sorts = rewriter.create<LLVM::AllocaOp>(op.getLoc(), ptrTy, declArrTy,
                                                  constOne);
    Value declNames = rewriter.create<LLVM::AllocaOp>(op.getLoc(), ptrTy,
                                                      declArrTy, constOne);
    Value sortsArray = rewriter.create<LLVM::UndefOp>(op.getLoc(), declArrTy);
    Value declNameArray =
        rewriter.create<LLVM::UndefOp>(op.getLoc(), declArrTy);
    unsigned numArgs = op.getBody().front().getNumArguments();

    for (auto [i, arg, name] : llvm::enumerate(op.getBody().getArguments(),
                                               adaptor.getBoundVarNames())) {
      Type type = arg.getType();
      Value sort = buildSort(rewriter, loc, ctx, type);
      for (auto &use : arg.getUses()) {
        Operation *parent = use.getOwner();
        unsigned idx = 0;
        while (parent != op) {
          if (isa<ForallOp, ExistsOp>(parent)) {
            idx += parent->getRegion(0).getNumArguments();
          }
          parent = use.getOwner()->getParentOp();
        }
        // NOTE: de-Bruijn indices start at 1 (not 0)
        idx += numArgs - i;
        Value deBruijnIndex =
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), idx);
        Value boundVar = buildAPICallGetPtr(
            rewriter, op.getLoc(), "Z3_mk_bound", {ctx, deBruijnIndex, sort});
        rewriter.replaceUsesWithIf(
            use.get(), boundVar,
            [&](OpOperand &operand) { return operand == use; });
      }

      if (!sort)
        return failure();
      sortsArray = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), sortsArray, sort, ArrayRef<int64_t>{(int64_t)i});
      Value boundVarName =
          buildStringRef(rewriter, loc, cast<StringAttr>(name).getValue());
      Value sym = buildAPICallGetPtr(
          rewriter, op.getLoc(), "Z3_mk_string_symbol", {ctx, boundVarName});
      declNameArray = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), declNameArray, sym, ArrayRef<int64_t>{(int64_t)i});
    }
    op.getBody().front().eraseArguments(0,
                                        op.getBody().front().getNumArguments());

    rewriter.create<LLVM::StoreOp>(op.getLoc(), sortsArray, sorts);
    rewriter.create<LLVM::StoreOp>(op.getLoc(), declNameArray, declNames);

    // Body Expression
    auto yieldOp = cast<smt::YieldOp>(op.getBody().front().getTerminator());
    Value bodyExp = yieldOp.getValues()[0];
    rewriter.setInsertionPointAfterValue(bodyExp);
    bodyExp = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(bodyExp.getType()), bodyExp);
    rewriter.eraseOp(yieldOp);
    rewriter.inlineBlockBefore(&op.getBody().front(), op);

    rewriter.setInsertionPoint(op);
    Value forallExp =
        buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_forall",
                           {ctx, weight, numPatternsVal, patterns, numDeclsVal,
                            sorts, declNames, bodyExp});

    rewriter.replaceOp(op, forallExp);
    return success();
  }
};

// NOTE: just a copy-paste of the ForallOpLowering
struct ExistsOpLowering : public SMTLoweringPattern<ExistsOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(ExistsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value ctx = buildZ3ContextPtr(rewriter, op.getLoc());

    // Patterns
    unsigned numPatterns = adaptor.getPatterns().size();
    Value weight = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                     adaptor.getWeight());
    Value numPatternsVal = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), numPatterns);

    Value constOne = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), 1);
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type patternArrTy = LLVM::LLVMArrayType::get(ptrTy, numPatterns);
    Value patterns = rewriter.create<LLVM::AllocaOp>(op.getLoc(), ptrTy,
                                                     patternArrTy, constOne);
    Value array = rewriter.create<LLVM::UndefOp>(op.getLoc(), patternArrTy);

    for (auto [i, operand] : llvm::enumerate(adaptor.getPatterns()))
      array = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), array, operand, ArrayRef<int64_t>{(int64_t)i});

    rewriter.create<LLVM::StoreOp>(op.getLoc(), array, patterns);

    // Bound variables
    unsigned numDecls = adaptor.getBoundVarNames().size();
    Value numDeclsVal =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), numDecls);

    Type declArrTy = LLVM::LLVMArrayType::get(ptrTy, numDecls);
    Value sorts = rewriter.create<LLVM::AllocaOp>(op.getLoc(), ptrTy, declArrTy,
                                                  constOne);
    Value declNames = rewriter.create<LLVM::AllocaOp>(op.getLoc(), ptrTy,
                                                      declArrTy, constOne);
    Value sortsArray = rewriter.create<LLVM::UndefOp>(op.getLoc(), declArrTy);
    Value declNameArray =
        rewriter.create<LLVM::UndefOp>(op.getLoc(), declArrTy);
    unsigned numArgs = op.getBody().front().getNumArguments();

    for (auto [i, arg, name] : llvm::enumerate(op.getBody().getArguments(),
                                               adaptor.getBoundVarNames())) {
      Type type = arg.getType();
      Value sort = buildSort(rewriter, loc, ctx, type);
      for (auto &use : arg.getUses()) {
        Operation *parent = use.getOwner();
        unsigned idx = 0;
        while (parent != op) {
          if (isa<ForallOp, ExistsOp>(parent)) {
            idx += parent->getRegion(0).getNumArguments();
          }
          parent = use.getOwner()->getParentOp();
        }
        // NOTE: de-Bruijn indices start at 1 (not 0)
        idx += numArgs - i;
        Value deBruijnIndex =
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), idx);
        Value boundVar = buildAPICallGetPtr(
            rewriter, op.getLoc(), "Z3_mk_bound", {ctx, deBruijnIndex, sort});
        rewriter.replaceUsesWithIf(
            use.get(), boundVar,
            [&](OpOperand &operand) { return operand == use; });
      }

      if (!sort)
        return failure();
      sortsArray = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), sortsArray, sort, ArrayRef<int64_t>{(int64_t)i});
      Value boundVarName =
          buildStringRef(rewriter, loc, cast<StringAttr>(name).getValue());
      Value sym = buildAPICallGetPtr(
          rewriter, op.getLoc(), "Z3_mk_string_symbol", {ctx, boundVarName});
      declNameArray = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), declNameArray, sym, ArrayRef<int64_t>{(int64_t)i});
    }
    op.getBody().front().eraseArguments(0,
                                        op.getBody().front().getNumArguments());

    rewriter.create<LLVM::StoreOp>(op.getLoc(), sortsArray, sorts);
    rewriter.create<LLVM::StoreOp>(op.getLoc(), declNameArray, declNames);

    // Body Expression
    auto yieldOp = cast<smt::YieldOp>(op.getBody().front().getTerminator());
    Value bodyExp = yieldOp.getValues()[0];
    rewriter.setInsertionPointAfterValue(bodyExp);
    bodyExp = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(bodyExp.getType()), bodyExp);
    rewriter.eraseOp(yieldOp);
    rewriter.inlineBlockBefore(&op.getBody().front(), op);

    rewriter.setInsertionPoint(op);
    Value forallExp =
        buildAPICallGetPtr(rewriter, op.getLoc(), "Z3_mk_exists",
                           {ctx, weight, numPatternsVal, patterns, numDeclsVal,
                            sorts, declNames, bodyExp});

    rewriter.replaceOp(op, forallExp);
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
  DenseMap<StringRef, LLVM::GlobalOp> stringCache;

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
  converter.addConversion([&](ArrayType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](smt::IntegerType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](smt::PatternType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  RewritePatternSet patterns(&getContext());
  populateFuncToLLVMConversionPatterns(converter, patterns);
  // populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
  // populateSCFToControlFlowConversionPatterns(patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);

#define ADD_OTO_PATTERN(OPTYPE, FUNCNAME)                                      \
  struct OPTYPE##Name {                                                        \
    static StringRef get() { return FUNCNAME; }                                \
  };                                                                           \
  patterns.add<OneToOneSMTPattern<OPTYPE, OPTYPE##Name>>(                      \
      converter, &getContext(), funcMap, ctxCache, stringCache);

#define ADD_VARIADIC_PATTERN(OPTYPE, FUNCNAME)                                 \
  struct OPTYPE##Name {                                                        \
    static StringRef get() { return FUNCNAME; }                                \
  };                                                                           \
  patterns.add<VariadicSMTPattern<OPTYPE, OPTYPE##Name>>(                      \
      converter, &getContext(), funcMap, ctxCache, stringCache);

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

  ADD_VARIADIC_PATTERN(IntAddOp, "Z3_mk_add");
  ADD_VARIADIC_PATTERN(IntMulOp, "Z3_mk_mul");
  ADD_VARIADIC_PATTERN(IntSubOp, "Z3_mk_sub");

  ADD_OTO_PATTERN(IntDivOp, "Z3_mk_div");
  ADD_OTO_PATTERN(IntModOp, "Z3_mk_mod");
  ADD_OTO_PATTERN(IntRemOp, "Z3_mk_rem");
  ADD_OTO_PATTERN(IntPowOp, "Z3_mk_power");

  patterns.add<ConstantOpLowering, DeclareConstOpLowering, BVCmpOpLowering,
               AssertOpLowering, CheckSatOpLowering, SolverCreateOpLowering,
               RepeatOpLowering, ExtractOpLowering, ArraySelectOpLowering,
               ArrayStoreOpLowering, ArrayBroadcastOpLowering,
               ArrayDefaultOpLowering, BoolConstantOpLowering,
               IntConstantOpLowering, IntCmpOpLowering, PatternCreateOpLowering,
               ForallOpLowering, ExistsOpLowering>(
      converter, &getContext(), funcMap, ctxCache, stringCache);

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();

  RewritePatternSet cleanupPatterns(&getContext());
  target.addIllegalOp<UnrealizedConversionCastOp>();
  populateReconcileUnrealizedCastsPatterns(cleanupPatterns);

  if (failed(applyFullConversion(getOperation(), target,
                                 std::move(cleanupPatterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> circt::createLowerSMTToZ3LLVMPass() {
  return std::make_unique<LowerSMTToZ3LLVMPass>();
}
