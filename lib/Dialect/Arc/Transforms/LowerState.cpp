//===- LowerState.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-state"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERSTATE
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Lowering Pattern Base
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct StateLoweringPattern : public OpConversionPattern<OpTy> {
  // StateLoweringPattern(TypeConverter converter, MLIRContext *context,
  // DenseMap<LayoutType, LayoutOp> &layoutMap, Namespace &names) :
  // OpConversionPattern<OpTy>(converter, context), layoutMap(layoutMap),
  // names(names) {}
  StateLoweringPattern(MLIRContext *context,
                       DenseMap<LayoutType, LayoutOp> &layoutMap,
                       Namespace &names)
      : OpConversionPattern<OpTy>(context), layoutMap(layoutMap), names(names) {
  }

protected:
  DenseMap<LayoutType, LayoutOp> &layoutMap;
  Namespace &names;
};

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

struct HWModuleOpLowering : public StateLoweringPattern<hw::HWModuleOp> {
  using StateLoweringPattern::StateLoweringPattern;

  LogicalResult
  matchAndRewrite(hw::HWModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    auto model = rewriter.create<ModelOp>(loc, adaptor.getSymNameAttr());
    auto modelLayout = rewriter.create<LayoutOp>(
        loc, names.newName(adaptor.getSymName() + "Layout"));
    auto layoutType = LayoutType::get(
        getContext(), FlatSymbolRefAttr::get(modelLayout.getSymNameAttr()));
    layoutMap[layoutType] = modelLayout;
    rewriter.createBlock(&model.getBody(), {}, layoutType, loc);

    rewriter.createBlock(&modelLayout.getBodyRegion());
    for (auto port : adaptor.getModuleType().getPorts()) {
      auto convertKind = [&]() {
        switch (port.dir) {
        case hw::ModulePort::Input:
          return LayoutKind::Input;
        case hw::ModulePort::InOut:
          return LayoutKind::InOut;
        case hw::ModulePort::Output:
          return LayoutKind::Output;
        }
      };
      auto type = port.type;
      if (isa<seq::ClockType>(type))
        type = rewriter.getI1Type();
      rewriter.create<EntryOp>(loc, port.name, type, convertKind());
    }

    Operation *terminator = op.getBodyBlock()->getTerminator();
    rewriter.setInsertionPointToEnd(op.getBodyBlock());
    for (auto [i, out] : llvm::enumerate(terminator->getOperands())) {
      auto name = adaptor.getModuleType().getOutputName(i);
      auto stateTy = out.getType();
      if (isa<seq::ClockType>(stateTy))
        stateTy = rewriter.getI1Type();
      if (!isa<MemoryType>(stateTy))
        stateTy = StateType::get(stateTy);
      Value newOut = rewriter.create<LayoutGetOp>(
          loc, stateTy, model.getBody().getArgument(0), name);
      Value outVal = out;
      if (isa<seq::ClockType>(out.getType()))
        outVal = rewriter.create<seq::FromClockOp>(loc, out);
      rewriter.create<StateWriteOp>(loc, newOut, outVal, Value{});
    }

    rewriter.setInsertionPointToStart(op.getBodyBlock());
    SmallVector<Value> repl;
    for (auto [i, arg] : llvm::enumerate(adaptor.getBody().getArguments())) {
      auto name = adaptor.getModuleType().getInputName(i);
      auto stateTy = arg.getType();
      if (isa<seq::ClockType>(stateTy))
        stateTy = rewriter.getI1Type();
      if (!isa<MemoryType>(stateTy))
        stateTy = StateType::get(stateTy);
      Value entry = rewriter.create<LayoutGetOp>(
          loc, stateTy, model.getBody().getArgument(0), name);
      entry = rewriter.create<StateReadOp>(loc, entry);
      if (isa<seq::ClockType>(arg.getType()))
        entry = rewriter.create<seq::ToClockOp>(loc, entry);
      repl.push_back(entry);
    }
    rewriter.inlineBlockBefore(op.getBodyBlock(), &model.getBodyBlock(),
                               model.getBodyBlock().end(), repl);

    rewriter.eraseOp(terminator);
    rewriter.eraseOp(op);
    return success();
  }
};

// struct HWOutputOpLowering : public StateLoweringPattern<hw::OutputOp> {
//   using StateLoweringPattern::StateLoweringPattern;

//   LogicalResult
//   matchAndRewrite(hw::OutputOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final {
//     Location loc = op.getLoc();
//     auto modelOp = op->getParentOfType<ModelOp>();
//     if (!modelOp)
//       return rewriter.notifyMatchFailure(op, "must be inside a model");

//     auto layoutType =
//     cast<LayoutType>(modelOp.getBody().getArgument(0).getType()); auto
//     layoutOp = layoutMap[layoutType];
//     {
//       OpBuilder::InsertionGuard g(rewriter);
//       rewriter.setInsertionPointToEnd(layoutOp.getBody());
//       rewriter.create<EntryOp>(loc);
//     }

//     rewriter.eraseOp(op);
//     return success();
//   }
// };

struct CallOpLowering : public StateLoweringPattern<CallOp> {
  using StateLoweringPattern::StateLoweringPattern;

  LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type> convertedTypes(op->getResultTypes());
    // if (failed(typeConverter->convertTypes(op->getResultTypes(),
    // convertedTypes)))
    //   return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, adaptor.getArcAttr(), convertedTypes, adaptor.getInputs());
    return success();
  }
};

struct StateOpLowering : public StateLoweringPattern<StateOp> {
  using StateLoweringPattern::StateLoweringPattern;

  LogicalResult
  matchAndRewrite(StateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto layout = op->getParentOfType<ModelOp>().getBody().getArgument(0);
    auto layoutOp = layoutMap[cast<LayoutType>(layout.getType())];

    SmallVector<Type> convertedTypes(op->getResultTypes());
    // if (failed(typeConverter->convertTypes(op->getResultTypes(),
    // convertedTypes)))
    //   return failure();

    SmallVector<StringRef> layoutNames;
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(layoutOp.getBody());
      for (Type resTy : convertedTypes) {
        StringRef name = names.newName("");
        layoutNames.push_back(name);
        rewriter.create<EntryOp>(loc, name, resTy, LayoutKind::Register);
      }
    }

    auto buildLogic = [&](bool storeZeros = false) -> LogicalResult {
      SmallVector<Value> toStore;
      if (storeZeros) {
        for (auto ty : convertedTypes) {
          if (!isa<IntegerType>(ty))
            return failure();
          toStore.push_back(rewriter.create<hw::ConstantOp>(loc, ty, 0));
        }
      } else {
        toStore = rewriter
                      .create<func::CallOp>(loc, adaptor.getArcAttr(),
                                            convertedTypes, adaptor.getInputs())
                      ->getResults();
      }
      for (auto [i, res] : llvm::enumerate(toStore)) {
        auto stateTy = res.getType();
        if (!isa<MemoryType>(stateTy))
          stateTy = StateType::get(stateTy);
        Value state =
            rewriter.create<LayoutGetOp>(loc, stateTy, layout, layoutNames[i]);
        rewriter.create<DeferredStateWriteOp>(loc, state, res, Value{});
      }
      return success();
    };
    auto buildWithEnable = [&](bool storeZeros = false) -> LogicalResult {
      if (adaptor.getEnable()) {
        Value trueVal =
            rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), 1);
        Value notEnable =
            rewriter.create<comb::XorOp>(loc, adaptor.getEnable(), trueVal);
        auto ifOp = rewriter.create<scf::IfOp>(loc, notEnable);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
      }

      if (failed(buildLogic(storeZeros)))
        return failure();

      if (adaptor.getEnable()) {
        rewriter.setInsertionPointAfter(rewriter.getBlock()->getParentOp());
      }

      return success();
    };
    auto buildWithReset = [&]() -> LogicalResult {
      if (adaptor.getReset()) {
        auto ifOp = rewriter.create<scf::IfOp>(loc, adaptor.getReset(), true);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        if (failed(buildWithEnable(true)))
          return failure();
        rewriter.setInsertionPointToStart(ifOp.elseBlock());
      }

      if (failed(buildWithEnable()))
        return failure();

      if (adaptor.getReset()) {
        rewriter.setInsertionPointAfter(rewriter.getBlock()->getParentOp());
      }

      return success();
    };

    SmallVector<Value> repl;
    for (auto [ty, name] : llvm::zip(convertedTypes, layoutNames)) {
      Type stateTy = ty;
      if (!isa<MemoryType>(stateTy))
        stateTy = StateType::get(stateTy);
      Value state = rewriter.create<LayoutGetOp>(loc, stateTy, layout, name);
      repl.push_back(rewriter.create<DeferredStateReadOp>(loc, state));
    }

    if (adaptor.getClock()) {
      Value cond = rewriter.create<seq::FromClockOp>(loc, adaptor.getClock());
      auto ifOp = rewriter.create<scf::IfOp>(loc, cond);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
    }

    if (failed(buildWithReset()))
      return failure();

    if (adaptor.getClock()) {
      rewriter.setInsertionPointAfter(rewriter.getBlock()->getParentOp());
    }

    rewriter.replaceOp(op, repl);
    return success();
  }
};

struct MemoryOpLowering : public StateLoweringPattern<MemoryOp> {
  using StateLoweringPattern::StateLoweringPattern;

  LogicalResult
  matchAndRewrite(MemoryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto layout = op->getParentOfType<ModelOp>().getBody().getArgument(0);
    auto layoutOp = layoutMap[cast<LayoutType>(layout.getType())];
    auto name = names.newName("mem");

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(layoutOp.getBody());
      rewriter.create<EntryOp>(loc, name, op.getType(), LayoutKind::Memory);
    }

    rewriter.replaceOpWithNewOp<LayoutGetOp>(op, op.getType(), layout, name);
    return success();
  }
};

struct DefineOpLowering : public StateLoweringPattern<DefineOp> {
  using StateLoweringPattern::StateLoweringPattern;

  LogicalResult
  matchAndRewrite(DefineOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto funcOp = rewriter.create<func::FuncOp>(loc, adaptor.getSymName(),
                                                adaptor.getFunctionType());
    funcOp->setAttr(
        "llvm.linkage",
        LLVM::LinkageAttr::get(getContext(), LLVM::linkage::Linkage::Internal));
    rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct OutputOpLowering : public StateLoweringPattern<OutputOp> {
  using StateLoweringPattern::StateLoweringPattern;

  LogicalResult
  matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct MemoryReadPortOpLowering
    : public StateLoweringPattern<MemoryReadPortOp> {
  using StateLoweringPattern::StateLoweringPattern;

  LogicalResult
  matchAndRewrite(MemoryReadPortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<MemoryReadOp>(op, adaptor.getMemory(),
                                              adaptor.getAddress());
    return success();
  }
};

struct MemoryWritePortOpLowering
    : public StateLoweringPattern<MemoryWritePortOp> {
  using StateLoweringPattern::StateLoweringPattern;

  LogicalResult
  matchAndRewrite(MemoryWritePortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getLatency() > 1)
      return rewriter.notifyMatchFailure(op, "latencies > 1 not supported yet");

    Location loc = op.getLoc();

    ValueRange results =
        rewriter
            .create<func::CallOp>(loc, op.getArcResultTypes(), adaptor.getArc(),
                                  adaptor.getInputs())
            ->getResults();

    auto enable = adaptor.getEnable() ? results[op.getEnableIdx()] : Value();

    // Materialize the operands for the write op within the surrounding clock
    // tree.

    // TODO: surround with clock posedge if statement
    auto address = results[op.getAddressIdx()];
    auto data = results[op.getDataIdx()];
    if (adaptor.getMask()) {
      Value mask = results[op.getMaskIdx(static_cast<bool>(enable))];
      Value oldData = rewriter.create<arc::MemoryReadOp>(
          mask.getLoc(), data.getType(), adaptor.getMemory(), address);
      Value allOnes =
          rewriter.create<hw::ConstantOp>(mask.getLoc(), oldData.getType(), -1);
      Value negatedMask =
          rewriter.create<comb::XorOp>(mask.getLoc(), mask, allOnes, true);
      Value maskedOldData = rewriter.create<comb::AndOp>(
          mask.getLoc(), negatedMask, oldData, true);
      Value maskedNewData =
          rewriter.create<comb::AndOp>(mask.getLoc(), mask, data, true);
      data = rewriter.create<comb::OrOp>(mask.getLoc(), maskedOldData,
                                         maskedNewData, true);
    }
    rewriter.create<MemoryWriteOp>(loc, adaptor.getMemory(), address, enable,
                                   data);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TapOpLowering : public StateLoweringPattern<TapOp> {
  using StateLoweringPattern::StateLoweringPattern;

  LogicalResult
  matchAndRewrite(TapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isa<IntegerType>(adaptor.getValue().getType()))
      return failure();

    Location loc = op.getLoc();
    auto layout = op->getParentOfType<ModelOp>().getBody().getArgument(0);
    auto layoutOp = layoutMap[cast<LayoutType>(layout.getType())];
    // auto name = names.newName(adaptor.getName());
    auto name = adaptor.getName();

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(layoutOp.getBody());
      rewriter.create<EntryOp>(loc, name, adaptor.getValue().getType(),
                               LayoutKind::Wire);
    }
    Type stateTy = StateType::get(adaptor.getValue().getType());
    Value state = rewriter.create<LayoutGetOp>(loc, stateTy, layout, name);
    rewriter.create<StateWriteOp>(loc, state, adaptor.getValue(), Value{});

    rewriter.eraseOp(op);
    return success();
  }
};

struct InstanceOpLowering : public StateLoweringPattern<hw::InstanceOp> {
  using StateLoweringPattern::StateLoweringPattern;

  LogicalResult
  matchAndRewrite(hw::InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallString<32> baseName(adaptor.getInstanceName());
    auto baseNameLen = baseName.size();
    Location loc = op.getLoc();
    auto layout = op->getParentOfType<ModelOp>().getBody().getArgument(0);
    auto layoutOp = layoutMap[cast<LayoutType>(layout.getType())];

    // Lower the inputs of the extmodule as state that is only written.
    for (auto [operand, name] :
         llvm::zip(adaptor.getOperands(), adaptor.getArgNames())) {
      auto intType = operand.getType().dyn_cast<IntegerType>();
      if (!intType)
        return failure();
      // return mlir::emitError(operand.getLoc(), "input ")
      //       << name << " of extern module " << adaptor.getModuleNameAttr()
      //       << " instance " << adaptor.getInstanceNameAttr()
      //       << " is of non-integer type " << operand.getType();
      baseName.resize(baseNameLen);
      baseName += '/';
      baseName += cast<StringAttr>(name).getValue();

      {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToEnd(layoutOp.getBody());
        rewriter.create<EntryOp>(loc, baseName, intType, LayoutKind::Register);
      }
      Value state = rewriter.create<LayoutGetOp>(loc, StateType::get(intType),
                                                 layout, baseName);
      rewriter.create<StateWriteOp>(loc, state, operand, Value{});
    }

    SmallVector<Value> repl;
    // Lower the outputs of the extmodule as state that is only read.
    for (auto [result, name] :
         llvm::zip(op.getResults(), adaptor.getResultNames())) {
      auto intType = result.getType().dyn_cast<IntegerType>();
      if (!intType)
        return failure();
      // return mlir::emitError(result.getLoc(), "output ")
      //       << name << " of extern module " << adaptor.getModuleNameAttr()
      //       << " instance " << adaptor.getInstanceNameAttr()
      //       << " is of non-integer type " << result.getType();
      baseName.resize(baseNameLen);
      baseName += '/';
      baseName += cast<StringAttr>(name).getValue();
      {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToEnd(layoutOp.getBody());
        rewriter.create<EntryOp>(loc, baseName, intType, LayoutKind::Register);
      }
      Value state = rewriter.create<LayoutGetOp>(loc, StateType::get(intType),
                                                 layout, baseName);
      repl.push_back(rewriter.create<StateReadOp>(loc, state));
    }

    rewriter.replaceOp(op, repl);
    return success();
  }
};

struct ClockGateOpLowering : public StateLoweringPattern<seq::ClockGateOp> {
  using StateLoweringPattern::StateLoweringPattern;

  LogicalResult
  matchAndRewrite(seq::ClockGateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerStatePass : public arc::impl::LowerStateBase<LowerStatePass> {
  LowerStatePass() = default;
  LowerStatePass(const LowerStatePass &pass) : LowerStatePass() {}

  void runOnOperation() override;
};
} // namespace

void LowerStatePass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalOp<hw::HWModuleOp>();
  target.addIllegalOp<hw::OutputOp>();
  target.addIllegalOp<hw::InstanceOp>();
  target.addIllegalOp<CallOp>();
  target.addIllegalOp<StateOp>();
  target.addIllegalOp<MemoryReadPortOp>();
  target.addIllegalOp<MemoryWritePortOp>();
  target.addIllegalOp<DefineOp>();
  target.addIllegalOp<OutputOp>();
  target.addIllegalOp<TapOp>();
  target.addIllegalOp<ClockDomainOp>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalOp<StateWriteOp>();
  target.addLegalOp<StateReadOp>();
  target.addLegalOp<DeferredStateWriteOp>();
  target.addLegalOp<DeferredStateReadOp>();
  target.addLegalOp<MemoryWriteOp>();
  target.addLegalOp<MemoryReadOp>();
  target.addLegalOp<ModelOp>();
  target.addLegalOp<LayoutOp>();
  target.addLegalOp<EntryOp>();
  target.addLegalOp<LayoutGetOp>();

  // TypeConverter converter;
  // converter.addConversion([](seq::ClockType ty) {
  //   return IntegerType::get(ty.getContext(), 1);
  // });
  // converter.addConversion([](IntegerType ty) {
  //   return StateType::get(ty);
  // });
  // converter.addArgumentMaterialization([](OpBuilder &builder, Type type,
  // ValueRange inputs, Location loc) -> Value {
  //   if (inputs.size() != 1)
  //     return Value();

  //   if (auto stateTy = dyn_cast<StateType>(inputs[0])) {
  //     if (type != stateTy.getType())
  //       return Value();

  //     return builder.create<StateReadOp>(loc, inputs[0]);
  //   }

  //   return Value();
  // });

  RewritePatternSet patterns(&getContext());
  DenseMap<LayoutType, LayoutOp> layoutMap;
  Namespace names;
  SymbolCache symCache;
  symCache.addDefinitions(getOperation());
  names.add(symCache);
  patterns.add<MemoryReadPortOpLowering, MemoryWritePortOpLowering,
               TapOpLowering, InstanceOpLowering, OutputOpLowering,
               DefineOpLowering, HWModuleOpLowering, CallOpLowering,
               StateOpLowering, MemoryOpLowering>(&getContext(), layoutMap,
                                                  names);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();

  // TODO: Remove remaining external modules
  for (auto externModule : llvm::make_early_inc_range(
           getOperation().getOps<hw::HWModuleExternOp>()))
    externModule->erase();

  for (auto model :
       llvm::make_early_inc_range(getOperation().getOps<ModelOp>()))
    sortTopologically(&model.getBodyBlock());
}

std::unique_ptr<Pass> arc::createLowerStatePass() {
  return std::make_unique<LowerStatePass>();
}
