//===- CombToSMT.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToSMT.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a comb::ReplicateOp operation to smt::RepeatOp
struct CombReplicateOpConversion : OpConversionPattern<ReplicateOp> {
  using OpConversionPattern<ReplicateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<smt::RepeatOp>(op, op.getMultiple(),
                                               adaptor.getInput());
    return success();
  }
};

/// Lower a comb::ICmpOp operation to a smt::BVCmpOp, smt::EqOp or
/// smt::DistinctOp
struct IcmpOpConversion : OpConversionPattern<ICmpOp> {
  using OpConversionPattern<ICmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getPredicate() == ICmpPredicate::weq ||
        adaptor.getPredicate() == ICmpPredicate::ceq ||
        adaptor.getPredicate() == ICmpPredicate::wne ||
        adaptor.getPredicate() == ICmpPredicate::cne)
      return op->emitOpError("unsupported operation");

    if (adaptor.getPredicate() == ICmpPredicate::eq) {
      rewriter.replaceOpWithNewOp<smt::EqOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
      return success();
    }

    if (adaptor.getPredicate() == ICmpPredicate::ne) {
      rewriter.replaceOpWithNewOp<smt::DistinctOp>(
          op, ValueRange{adaptor.getLhs(), adaptor.getRhs()});
      return success();
    }

    smt::Predicate pred;
    switch (adaptor.getPredicate()) {
    case ICmpPredicate::sge:
      pred = smt::Predicate::sge;
      break;
    case ICmpPredicate::sgt:
      pred = smt::Predicate::sgt;
      break;
    case ICmpPredicate::sle:
      pred = smt::Predicate::sle;
      break;
    case ICmpPredicate::slt:
      pred = smt::Predicate::slt;
      break;
    case ICmpPredicate::uge:
      pred = smt::Predicate::uge;
      break;
    case ICmpPredicate::ugt:
      pred = smt::Predicate::ugt;
      break;
    case ICmpPredicate::ule:
      pred = smt::Predicate::ule;
      break;
    case ICmpPredicate::ult:
      pred = smt::Predicate::ult;
      break;
    default:
      llvm_unreachable("other cases handled above");
    }

    rewriter.replaceOpWithNewOp<smt::BVCmpOp>(op, pred, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

/// Lower a comb::ExtractOp operation to an smt::ExtractOp
struct ExtractOpConversion : OpConversionPattern<ExtractOp> {
  using OpConversionPattern<ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<smt::ExtractOp>(
        op, typeConverter->convertType(op.getResult().getType()),
        adaptor.getLowBitAttr(), adaptor.getInput());
    return success();
  }
};

/// Lower a comb::MuxOp operation to an smt::IteOp
struct MuxOpConversion : OpConversionPattern<MuxOp> {
  using OpConversionPattern<MuxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value constOneBit = rewriter.create<smt::ConstantOp>(
        op.getLoc(), smt::BitVectorAttr::get(
                         rewriter.getContext(), 1,
                         smt::BitVectorType::get(rewriter.getContext(), 1)));
    Value condition =
        rewriter.create<smt::EqOp>(op.getLoc(), adaptor.getCond(), constOneBit);
    rewriter.replaceOpWithNewOp<smt::IteOp>(
        op, condition, adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

/// Lower the two-operand SourceOp to the two-operand TargetOp
template <typename SourceOp, typename TargetOp>
struct BinaryOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<TargetOp>(
        op,
        OpConversionPattern<SourceOp>::typeConverter->convertType(
            op.getResult().getType()),
        adaptor.getOperands());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct VariadicOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // TODO: building a tree would be better here
    ValueRange operands = adaptor.getOperands();
    Value runner = operands[0];
    for (Value operand :
         llvm::make_range(operands.begin() + 1, operands.end())) {
      runner = rewriter.create<TargetOp>(op.getLoc(), runner, operand);
    }
    rewriter.replaceOp(op, runner);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCombToSMTPass
    : public ConvertCombToSMTBase<ConvertCombToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateCombToSMTConversionPatterns(TypeConverter &converter,
                                                RewritePatternSet &patterns) {
  converter.addConversion([](IntegerType type) {
    return smt::BitVectorType::get(type.getContext(), type.getWidth());
  });
  converter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;

        if (!isa<smt::BoolType>(inputs[0].getType()) ||
            !isa<smt::BitVectorType>(resultType))
          return std::nullopt;

        MLIRContext *ctx = builder.getContext();
        Value constZero = builder.create<smt::ConstantOp>(
            loc, smt::BitVectorAttr::get(ctx, 0, resultType));
        Value constOne = builder.create<smt::ConstantOp>(
            loc, smt::BitVectorAttr::get(ctx, 1, resultType));
        return builder.create<smt::IteOp>(loc, inputs[0], constOne, constZero);
      });

  patterns.add<CombReplicateOpConversion, IcmpOpConversion, ExtractOpConversion,
               VariadicOpConversion<ConcatOp, smt::ConcatOp>,
               BinaryOpConversion<ShlOp, smt::ShlOp>,
               BinaryOpConversion<ShrUOp, smt::LShrOp>,
               BinaryOpConversion<ShrSOp, smt::AShrOp>,
               BinaryOpConversion<SubOp, smt::SubOp>,
               BinaryOpConversion<DivSOp, smt::SDivOp>,
               BinaryOpConversion<DivUOp, smt::UDivOp>,
               BinaryOpConversion<ModSOp, smt::SRemOp>,
               BinaryOpConversion<ModUOp, smt::URemOp>, MuxOpConversion,
               VariadicOpConversion<AddOp, smt::AddOp>,
               VariadicOpConversion<MulOp, smt::MulOp>,
               VariadicOpConversion<AndOp, smt::BVAndOp>,
               VariadicOpConversion<OrOp, smt::BVOrOp>,
               VariadicOpConversion<XorOp, smt::BVXOrOp>>(
      converter, patterns.getContext());
}

void ConvertCombToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<comb::CombDialect>();
  target.addLegalDialect<smt::SMTDialect>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateCombToSMTConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertCombToSMTPass() {
  return std::make_unique<ConvertCombToSMTPass>();
}
