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
struct AssignOpConv;

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

  patterns.add<ConstantOpConv, VariableDeclOpConv, AssignOpConv>(typeConverter,
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

struct AssignOpConv : public OpConversionPattern<moore::AssignOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(moore::AssignOp op, OpAdaptor adaptor,
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

} // namespace
