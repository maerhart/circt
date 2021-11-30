//===- MooreToCore.cpp - Moore To Core Conversion Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Moore to Core Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/MooreToCore.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Moore/MIROps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// Moore to Core Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct MooreToCorePass : public ConvertMooreToCoreBase<MooreToCorePass> {
  void runOnOperation() override;
};
} // namespace

/// Create a Moore to core dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertMooreToCorePass() {
  return std::make_unique<MooreToCorePass>();
}

namespace {
/// Forward declarations
struct ConstantOpConv;
struct VariableDeclOpConv;
struct BlockingAssignOpConv;
struct DivOpConv;
struct ModOpConv;
struct PowOpConv;
struct NotOpConv;
struct NegOpConv;
struct AndReduceOpConv;
struct OrReduceOpConv;
struct XorReduceOpConv;

static Type convertMooreType(Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<moore::IntType>([](moore::IntType ty) {
        return IntegerType::get(ty.getContext(), 32);
      })
      .Case<moore::RValueType>(
          [](auto type) { return convertMooreType(type.getNestedType()); })
      .Case<moore::LValueType>([](auto type) {
        return llhd::SigType::get(convertMooreType(type.getNestedType()));
      })
      .Default([](Type type) { return type; });
}

/// Only operations with 0 or 1 result are supported
static LogicalResult oneToOneRewrite(Operation *op, StringRef targetOp,
                                     ValueRange operands,
                                     TypeConverter &typeConverter,
                                     ConversionPatternRewriter &rewriter) {
  unsigned numResults = op->getNumResults();

  Type packedType;
  if (numResults != 0) {
    packedType = typeConverter.convertType(op->getResults()[0].getType());
    if (!packedType)
      return failure();
  }

  // Create the operation through state since we don't know its C++ type.
  OperationState state(op->getLoc(), targetOp);
  state.addTypes(packedType);
  state.addOperands(operands);
  state.addAttributes(op->getAttrs());
  Operation *newOp = rewriter.createOperation(state);

  // If the operation produced 0 or 1 result, return them immediately.
  if (numResults == 0)
    return rewriter.eraseOp(op), success();
  if (numResults == 1)
    return rewriter.replaceOp(op, newOp->getResult(0)), success();

  return success();
}

template <typename SourceOp, typename TargetOp>
class OneToOneConversionPattern : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using Super = OneToOneConversionPattern<SourceOp, TargetOp>;

  /// Converts the type of the result to an LLVM type, pass operands as is,
  /// preserve attributes.
  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return oneToOneRewrite(op, TargetOp::getOperationName(),
                           adaptor.getOperands(), *this->getTypeConverter(),
                           rewriter);
  }
};

/// This is the main entrypoint for the Moore to Core conversion pass.
void MooreToCorePass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  // Mark all MIR ops as illegal such that they get rewritten.
  ConversionTarget target(context);
  target.addIllegalDialect<moore::MooreDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<llhd::LLHDDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalOp<ModuleOp>();

  TypeConverter typeConverter;
  typeConverter.addConversion(convertMooreType);
  RewritePatternSet patterns(&context);

  patterns.add<ConstantOpConv, VariableDeclOpConv, BlockingAssignOpConv>(
      typeConverter, &context);
  patterns.add<OneToOneConversionPattern<moore::AndOp, comb::AndOp>>(
      typeConverter, &context);
  patterns.add<OneToOneConversionPattern<moore::OrOp, comb::OrOp>>(
      typeConverter, &context);
  patterns.add<OneToOneConversionPattern<moore::XorOp, comb::XorOp>>(
      typeConverter, &context);
  patterns.add<OneToOneConversionPattern<moore::AddOp, comb::AddOp>>(
      typeConverter, &context);
  patterns.add<OneToOneConversionPattern<moore::SubOp, comb::SubOp>>(
      typeConverter, &context);
  patterns.add<OneToOneConversionPattern<moore::MulOp, comb::MulOp>>(
      typeConverter, &context);
  patterns.add<DivOpConv, ModOpConv, PowOpConv, NotOpConv, NegOpConv,
               AndReduceOpConv, OrReduceOpConv, XorReduceOpConv>(typeConverter,
                                                                 &context);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Operation conversion patterns
//===----------------------------------------------------------------------===//

struct ConstantOpConv : public OpConversionPattern<moore::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.valueAttr());
    return success();
  }
};

struct VariableDeclOpConv : public OpConversionPattern<moore::VariableDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::VariableDeclOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type resultType = typeConverter->convertType(op.result().getType());
    Value initVal =
        rewriter.create<hw::ConstantOp>(op->getLoc(), op.initAttr());
    rewriter.replaceOpWithNewOp<llhd::SigOp>(op, resultType, op.name(),
                                             initVal);
    return success();
  }
};

struct BlockingAssignOpConv
    : public OpConversionPattern<moore::BlockingAssignOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::BlockingAssignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value timeVal =
        rewriter.create<llhd::ConstantTimeOp>(op->getLoc(), 0, "s", 0, 1);
    Type destType = typeConverter->convertType(op.dest().getType());
    Type srcType = typeConverter->convertType(op.src().getType());
    op.dest().setType(destType);
    op.src().setType(srcType);
    rewriter.replaceOpWithNewOp<llhd::DrvOp>(op, op.dest(), op.src(), timeVal,
                                             Value());
    return success();
  }
};

struct DivOpConv : public OpConversionPattern<moore::DivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type ty = typeConverter->convertType(op.lhs().getType());
    op.lhs().setType(ty);
    op.rhs().setType(ty);
    if (op.isSigned()) {
      rewriter.replaceOpWithNewOp<comb::DivSOp>(op, op.lhs(), op.rhs());
    } else {
      rewriter.replaceOpWithNewOp<comb::DivUOp>(op, op.lhs(), op.rhs());
    }
    return success();
  }
};

struct ModOpConv : public OpConversionPattern<moore::ModOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::ModOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type ty = typeConverter->convertType(op.lhs().getType());
    op.lhs().setType(ty);
    op.rhs().setType(ty);
    if (op.isSigned()) {
      rewriter.replaceOpWithNewOp<comb::ModSOp>(op, op.lhs(), op.rhs());
    } else {
      rewriter.replaceOpWithNewOp<comb::ModUOp>(op, op.lhs(), op.rhs());
    }
    return success();
  }
};

struct PowOpConv : public OpConversionPattern<moore::PowOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::PowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type ty = typeConverter->convertType(op.lhs().getType());
    op.lhs().setType(ty);
    op.rhs().setType(ty);
    if (op.isSigned()) {
      rewriter.replaceOpWithNewOp<comb::DivSOp>(op, op.lhs(), op.rhs());
    } else {
      rewriter.replaceOpWithNewOp<comb::DivUOp>(op, op.lhs(), op.rhs());
    }
    return success();
  }
};

struct NotOpConv : public OpConversionPattern<moore::NotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type ty = typeConverter->convertType(op.arg().getType());
    op.arg().setType(ty);
    Value allsetValue = rewriter.create<hw::ConstantOp>(
        op->getLoc(), rewriter.getIntegerAttr(ty, -1));
    rewriter.replaceOpWithNewOp<comb::XorOp>(op, op.arg(), allsetValue);
    return success();
  }
};

struct NegOpConv : public OpConversionPattern<moore::NegOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type ty = typeConverter->convertType(op.arg().getType());
    op.arg().setType(ty);
    Value zeroValue = rewriter.create<hw::ConstantOp>(
        op->getLoc(), rewriter.getIntegerAttr(ty, 0));
    rewriter.replaceOpWithNewOp<comb::SubOp>(op, zeroValue, op.arg());
    return success();
  }
};

struct AndReduceOpConv : public OpConversionPattern<moore::AndReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::AndReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type ty = typeConverter->convertType(op.arg().getType());
    op.arg().setType(ty);
    Value allsetValue = rewriter.create<hw::ConstantOp>(
        op->getLoc(), rewriter.getIntegerAttr(ty, -1));
    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::eq,
                                              op.arg(), allsetValue);
    return success();
  }
};

struct OrReduceOpConv : public OpConversionPattern<moore::OrReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::OrReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type ty = typeConverter->convertType(op.arg().getType());
    op.arg().setType(ty);
    Value zeroValue = rewriter.create<hw::ConstantOp>(
        op->getLoc(), rewriter.getIntegerAttr(ty, 0));
    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                              op.arg(), zeroValue);
    return success();
  }
};

struct XorReduceOpConv : public OpConversionPattern<moore::XorReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::XorReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type ty = typeConverter->convertType(op.arg().getType());
    op.arg().setType(ty);
    rewriter.replaceOpWithNewOp<comb::ParityOp>(op, op.arg());
    return success();
  }
};

} // namespace
