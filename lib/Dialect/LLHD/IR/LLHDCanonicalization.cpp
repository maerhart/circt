//===- LLHDCanonicalization.cpp - Register LLHD Canonicalization Patterns -===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "circt/Dialect/LLHD/IR/LLHDCanonicalization.inc"
} // namespace

struct DrvOfPrb
    : public mlir::OpRewritePattern<llhd::DrvOp> {
  DrvOfPrb(mlir::MLIRContext *context)
      : OpRewritePattern<llhd::DrvOp>(context, /*benefit=*/999) {}

  mlir::LogicalResult
  matchAndRewrite(llhd::DrvOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // auto time = op.time()
    //                 .getDefiningOp<llhd::ConstOp>()
    //                 .valueAttr()
    //                 .cast<llhd::TimeAttr>();
    // if (time.getTime() != 0 || time.getDelta() != 0)
    //   return failure();

    // if (auto prb = op.value().getDefiningOp<llhd::PrbOp>()) {
    //   if (op.signal() == prb.signal()) {
    //     op.getOperation()->dropAllReferences();
    //     op.erase();
    //     return success();
    //   }
    //   if (prb.signal().isa<BlockArgument>()) {
    //     op.signal().replaceAllUsesWith(prb.signal());
    //     op.getOperation()->dropAllReferences();
    //     op.erase();
    //     return success();
    //   }
    //   if (auto sig = prb.signal().getDefiningOp<llhd::SigOp>()) {
    //     sig.getOperation()->moveAfter(sig.init().getDefiningOp());
    //     op.signal().replaceAllUsesWith(prb.signal());
    //     op.getOperation()->dropAllReferences();
    //     op.erase();
    //     return success();
    //   }
    // }
    return failure();
  }
};

void llhd::XorOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<XorAllBitsSet>(context);
}

void llhd::NotOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<NotOfEq, NotOfNeq>(context);
}

void llhd::EqOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<BooleanEqToXor>(context);
}

void llhd::NeqOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<BooleanNeqToXor>(context);
}

// void llhd::DrvOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
//                                               MLIRContext *context) {
//   results.insert<DrvOfPrb>(context);
// }

// void llhd::ShrOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
//                                               MLIRContext *context) {
//   results.insert<ShrOpConversion>(context);
// }

void llhd::DynExtractSliceOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<DynExtractSliceWithConstantOpStart,
                 DynExtractSliceWithLLHDConstOpStart>(context);
}

void llhd::DynExtractElementOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<DynExtractElementWithConstantOpIndex,
                 DynExtractElementWithLLHDConstOpIndex>(context);
}
