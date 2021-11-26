//===- MooreMIRToCore.cpp - Moore MIR To Core Conversion Pass -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Moore MIR to Core Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/MooreMIRToCore.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/IR/LLHDTypes.h"
#include "circt/Dialect/Moore/MIR/MIRDialect.h"
#include "circt/Dialect/Moore/MIR/MIROps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace moore;
using namespace llhd;
using namespace hw;
using namespace comb;

//===----------------------------------------------------------------------===//
// Moore MIR to Core Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct MIRToCorePass : public ConvertMooreMIRToCoreBase<MIRToCorePass> {
  void runOnOperation() override;
};
} // namespace

/// Create a Moore MIR to core dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>>
circt::createConvertMooreMIRToCorePass() {
  return std::make_unique<MIRToCorePass>();
}

namespace {
/// Forward declarations
struct ConstantOpConv;
struct VariableDeclOpConv;
struct AssignOpConv;

static Type convertMIRType(Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<mir::IntType>(
          [](mir::IntType ty) { return IntegerType::get(ty.getContext(), 32); })
      .Case<mir::RValueType>(
          [](auto type) { return convertMIRType(type.getRealType()); })
      .Case<mir::LValueType>([](auto type) {
        return llhd::SigType::get(convertMIRType(type.getRealType()));
      })
      .Default([](Type type) { return type; });
}

/// This is the main entrypoint for the MIR to Core conversion pass.
void MIRToCorePass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  // Mark all MIR ops as illegal such that they get rewritten.
  ConversionTarget target(context);
  target.addIllegalDialect<mir::MIRDialect>();
  target.addLegalDialect<HWDialect>();
  target.addLegalDialect<LLHDDialect>();
  target.addLegalDialect<CombDialect>();
  target.addLegalOp<ModuleOp>();

  TypeConverter typeConverter;
  typeConverter.addConversion(convertMIRType);
  RewritePatternSet patterns(&context);

  patterns.add<ConstantOpConv, VariableDeclOpConv, AssignOpConv>(typeConverter,
                                                                 &context);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Operation conversion patterns
//===----------------------------------------------------------------------===//

struct ConstantOpConv : public OpConversionPattern<mir::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mir::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.valueAttr());
    return success();
  }
};

struct VariableDeclOpConv : public OpConversionPattern<mir::VariableDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mir::VariableDeclOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type resultType = typeConverter->convertType(op.result().getType());
    Value initVal =
        rewriter.create<hw::ConstantOp>(op->getLoc(), op.initAttr());
    rewriter.replaceOpWithNewOp<llhd::SigOp>(op, resultType, op.name(),
                                             initVal);
    return success();
  }
};

struct AssignOpConv : public OpConversionPattern<mir::AssignOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mir::AssignOp op, OpAdaptor adaptor,
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
