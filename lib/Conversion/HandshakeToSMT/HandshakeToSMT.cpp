//===- HandshakeToSMT.cpp - Translate Handshake into SMT ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Handshake to SMT Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HandshakeToSMT.h"
#include "../PassDetail.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

using namespace mlir;
using namespace circt;
using namespace circt::handshake;

#define UPPER_BOUND 5

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

struct FuncOpConversion : OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    SmallVector<Type> inputs;
    SmallVector<Type> results;
    inputs.push_back(smt::SolverType::get(getContext()));
    for (auto ty : adaptor.getFunctionType().getInputs()) {
      auto convTy = typeConverter->convertType(ty);
      if (!convTy)
        return failure();
      if (auto tupleType = dyn_cast<TupleType>(convTy)) {
        for (auto r : tupleType.getTypes())
          inputs.push_back(r);
      } else {
        inputs.push_back(convTy);
      }
    }
    for (auto ty : adaptor.getFunctionType().getResults()) {
      auto convTy = typeConverter->convertType(ty);
      if (!convTy)
        return failure();
      if (auto tupleType = dyn_cast<TupleType>(convTy)) {
        for (auto r : tupleType.getTypes())
          results.push_back(r);
      } else {
        results.push_back(convTy);
      }
    }
    FunctionType functionType = FunctionType::get(getContext(), inputs, results);
    func::FuncOp funcOp = rewriter.create<func::FuncOp>(loc, op.getNameAttr(), functionType);
    funcOp.getBody().takeBody(op.getBody());

    // Adjust block arguments.
    Block &block = funcOp.getBody().front();
    rewriter.setInsertionPointToStart(&block);
    unsigned originalNumArgs = block.getNumArguments();
    SmallVector<BlockArgument> args(block.getArguments());
    block.addArgument(smt::SolverType::get(getContext()), loc);
    for (auto arg : args) {
      auto convTy = typeConverter->convertType(arg.getType());
      if (!convTy)
        return failure();
      if (auto tupleType = dyn_cast<TupleType>(convTy)) {
        SmallVector<Value> replArgs;
        for (auto r : tupleType.getTypes())
          replArgs.push_back(block.addArgument(r, arg.getLoc()));


        rewriter.replaceAllUsesWith(arg,rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{convTy}, replArgs)->getResult(0));
      } else {
        auto newArg = block.addArgument(convTy, arg.getLoc());
        rewriter.replaceAllUsesWith(arg, newArg);
      }
    }
    block.eraseArguments(0, originalNumArgs);

    rewriter.setInsertionPoint(op);
    rewriter.replaceOp(op, funcOp);

    return success();
  }
};

// struct ReturnOpConversion : OpConversionPattern<ReturnOp> {
//   using OpConversionPattern<ReturnOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     Location loc = loc;

//     SmallVector<Value> results;
//     for (auto [i, val] : llvm::enumerate(adaptor.getOperands())) {
//       if (auto tupleType = dyn_cast<TupleType>(val.getType())) {
//         results.append(SmallVector<Value>(rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), ValueRange{val})->getResults()));
//       } else {
//         if (op.getOperand(i).getDefiningOp<JoinOp>()) {
//           results.push_back(val);
//         } else {
//           // Limit the sequence to 10 elements as we do in the join pattern
//           auto constFalse = rewriter.create<smt::BoolConstantOp>(loc, false);
//           Value arr = rewriter.create<smt::ArrayBroadcastOp>(loc, val.getType(), constFalse);
//           for (int i = 0; i < UPPER_BOUND; i++) {
//             auto idx = rewriter.create<smt::IntConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), i));
//             auto select = rewriter.create<smt::ArraySelectOp>(loc, val, idx);
//             arr = rewriter.create<smt::ArrayStoreOp>(loc, arr, idx, select);
//           }
//           results.push_back(arr);
//         }
//       }
//     }
//     rewriter.replaceOpWithNewOp<func::ReturnOp>(op, results);
//     return success();
//   }
// };

struct ReturnOpConversion : OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<Value> results;
    for (auto [i, val] : llvm::enumerate(adaptor.getOperands())) {
      if (auto tupleType = dyn_cast<TupleType>(val.getType())) {
        results.append(SmallVector<Value>(rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), ValueRange{val})->getResults()));
      } else {
        // if (op->getOperand(i).getDefiningOp<JoinOp>()) {
          results.push_back(val);
        // } else if (op->getOperand(i).getDefiningOp<SourceOp>()) {
        //   Type smtIntType = smt::IntegerType::get(getContext());
        //   Value falseConst = rewriter.create<smt::BoolConstantOp>(loc, false);
        //   Value trueConst = rewriter.create<smt::BoolConstantOp>(loc, true);
        //   Value resArray = rewriter.create<smt::ArrayBroadcastOp>(loc, typeConverter->convertType(val.getType()), falseConst);
        //   for (int i = 0; i < UPPER_BOUND; ++i) {
        //     Value idx = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), i));
        //     resArray = rewriter.create<smt::ArrayStoreOp>(loc, resArray, idx, trueConst);
        //   }
        //   results.push_back(resArray);
        // } else {
        //   Type smtIntType = smt::IntegerType::get(getContext());
        //   Value falseConst = rewriter.create<smt::BoolConstantOp>(loc, false);
        //   // Value trueConst = rewriter.create<smt::BoolConstantOp>(loc, true);
        //   Value resArray = rewriter.create<smt::ArrayBroadcastOp>(loc, typeConverter->convertType(val.getType()), falseConst);
        //   Value zeroInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 0));
        //   Value oneInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 1));
        //   Value constZero = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), 0));
        //   Value constOne = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), 1));
        //   Value upperBound = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), UPPER_BOUND));
        //   Value arr = val;
        //   ValueRange arr1Results = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{zeroInt, zeroInt, resArray}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
        //     Value smtI = iterArgs[0];
        //     Value k = iterArgs[1];
        //     Value resArr = iterArgs[2];

        //     Value present = rewriter.create<smt::ArraySelectOp>(loc, arr, smtI);

        //     Value kPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{k, oneInt});
        //     Value newK = rewriter.create<smt::IteOp>(loc, present, kPlusOne, k);
        //     Value store = rewriter.create<smt::ArrayStoreOp>(loc, resArr, k, present);
        //     Value newResArr = rewriter.create<smt::IteOp>(loc, present, store, resArr);

        //     Value newSmtI = rewriter.create<smt::IntAddOp>(loc, ValueRange{smtI, oneInt});
        //     rewriter.create<scf::YieldOp>(loc, ValueRange{newSmtI, newK, newResArr});
        //   })->getResults();
        //   results.push_back(arr1Results[2]);
        // }
      }
    }
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, results);
    return success();
  }
};

struct ForkOpConversion : OpConversionPattern<ForkOp> {
  using OpConversionPattern<ForkOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ForkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, SmallVector<Value>(op->getNumResults(), adaptor.getOperand()));
    return success();
  }
};

struct SinkOpConversion : OpConversionPattern<SinkOp> {
  using OpConversionPattern<SinkOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SinkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrealizedOpConversion : OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), newTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, newTypes, adaptor.getInputs());
    return success();
  }
};

struct SourceOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value constant = rewriter.create<smt::BoolConstantOp>(loc, true); // indicate that token is present
    Type smtIntType = smt::IntegerType::get(getContext());
    Type smtBoolType = smt::BoolType::get(getContext());
    rewriter.replaceOpWithNewOp<smt::ArrayBroadcastOp>(op, smt::ArrayType::get(getContext(), smtIntType, smtBoolType), constant);
    return success();
  }
};

// struct JoinOpConversion : OpConversionPattern<JoinOp> {
//   using OpConversionPattern<JoinOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(JoinOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     if (adaptor.getData().size() != 2)
//       return op.emitOpError("only join operations with exactly two operands supported");

//     Type smtIntType = smt::IntegerType::get(getContext());
//     Value arr1 = adaptor.getData()[0];
//     Value arr2 = adaptor.getData()[1];

//     if (auto tupleType = dyn_cast<TupleType>(arr1.getType()))
//       arr1 = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{tupleType.getType(0), tupleType.getType(1)}, ValueRange{arr1})->getResult(1);

//     if (auto tupleType = dyn_cast<TupleType>(arr2.getType()))
//       arr2 = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{tupleType.getType(0), tupleType.getType(1)}, ValueRange{arr2})->getResult(1);

//     Value zeroInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 0));
//     Value oneInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 1));
//     Value falseConst = rewriter.create<smt::BoolConstantOp>(loc, false);
//     Value trueConst = rewriter.create<smt::BoolConstantOp>(loc, true);
//     Value resArray = rewriter.create<smt::ArrayBroadcastOp>(loc, typeConverter->convertType(op.getResult().getType()), falseConst);

//     Value constZero = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), 0));
//     Value constOne = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), 1));
//     Value upperBound = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), UPPER_BOUND));
//     ValueRange forLoopResults = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{zeroInt, zeroInt, zeroInt, resArray}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
//       Value smtI = iterArgs[0];
//       Value j = iterArgs[1];
//       Value k = iterArgs[2];
//       Value resArr = iterArgs[3];

//       ValueRange l1Results = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{j}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
//         Value j = iterArgs[0];
//         Value jPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{j, oneInt});
//         Value t = rewriter.create<smt::ArraySelectOp>(loc, arr1, j);
//         Value notPresent = rewriter.create<smt::EqOp>(loc, t, falseConst);
//         Value jSmallerK = rewriter.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, j, k);
//         Value cond = rewriter.create<smt::AndOp>(loc, ValueRange{notPresent, jSmallerK});
//         Value j1 = rewriter.create<smt::IteOp>(loc, cond, jPlusOne, j);
//         rewriter.create<scf::YieldOp>(loc, ValueRange{j1});
//       })->getResults();

//       j = l1Results[0];

//       ValueRange l2Results = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{k}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
//         Value k = iterArgs[0];
//         Value kPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{k, oneInt});
//         Value t = rewriter.create<smt::ArraySelectOp>(loc, arr2, k);
//         Value notPresent = rewriter.create<smt::EqOp>(loc, t, falseConst);
//         Value kSmallerJ = rewriter.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, k, j);
//         Value cond = rewriter.create<smt::AndOp>(loc, ValueRange{notPresent, kSmallerJ});
//         Value k1 = rewriter.create<smt::IteOp>(loc, cond, kPlusOne, k);
//         rewriter.create<scf::YieldOp>(loc, ValueRange{k1});
//       })->getResults();

//       k = l2Results[0];

//       Value t1 = rewriter.create<smt::ArraySelectOp>(loc, arr1, j);
//       Value t2 = rewriter.create<smt::ArraySelectOp>(loc, arr2, k);
//       Value t1Present = rewriter.create<smt::EqOp>(loc, t1, trueConst);
//       Value t2Present = rewriter.create<smt::EqOp>(loc, t2, trueConst);
//       Value res = rewriter.create<smt::AndOp>(loc, ValueRange{t1Present, t2Present});
//       Value newResArr = rewriter.create<smt::ArrayStoreOp>(loc, resArr, smtI, res);
//       Value t1Absent = rewriter.create<smt::EqOp>(loc, t1, falseConst);
//       Value t2Absent = rewriter.create<smt::EqOp>(loc, t2, falseConst);
//       Value disj2 = rewriter.create<smt::AndOp>(loc, ValueRange{t1Absent, t2Present});
//       Value jEqualsK = rewriter.create<smt::EqOp>(loc, j, k);
//       Value disj0 = rewriter.create<smt::AndOp>(loc, ValueRange{t1Absent, t2Absent, jEqualsK});
//       Value jCond = rewriter.create<smt::OrOp>(loc, ValueRange{disj0, res, disj2});
//       Value jPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{j, oneInt});
//       Value newJ = rewriter.create<smt::IteOp>(loc, jCond, jPlusOne, j);

//       disj2 = rewriter.create<smt::AndOp>(loc, ValueRange{t1Present, t2Absent});
//       Value kCond = rewriter.create<smt::OrOp>(loc, ValueRange{disj0, res, disj2});
//       Value kPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{k, oneInt});
//       Value newK = rewriter.create<smt::IteOp>(loc, kCond, kPlusOne, k);

//       Value smtIPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{smtI, oneInt});
//       rewriter.create<scf::YieldOp>(loc, ValueRange{smtIPlusOne, newJ, newK, newResArr});
//     })->getResults();

//     Value i = forLoopResults[0];
//     Value j = forLoopResults[1];
//     Value k = forLoopResults[2];
//     resArray = forLoopResults[3];

//     ValueRange l1Results = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{j}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
//       Value j = iterArgs[0];
//       Value jPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{j, oneInt});
//       Value t = rewriter.create<smt::ArraySelectOp>(loc, arr1, j);
//       Value notPresent = rewriter.create<smt::EqOp>(loc, t, falseConst);
//       Value jSmallerK = rewriter.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, j, k);
//       Value cond = rewriter.create<smt::AndOp>(loc, ValueRange{notPresent, jSmallerK});
//       Value j1 = rewriter.create<smt::IteOp>(loc, cond, jPlusOne, j);
//       rewriter.create<scf::YieldOp>(loc, ValueRange{j1});
//     })->getResults();

//     j = l1Results[0];

//     ValueRange l2Results = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{k}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
//       Value k = iterArgs[0];
//       Value kPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{k, oneInt});
//       Value t = rewriter.create<smt::ArraySelectOp>(loc, arr2, k);
//       Value notPresent = rewriter.create<smt::EqOp>(loc, t, falseConst);
//       Value kSmallerJ = rewriter.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, k, j);
//       Value cond = rewriter.create<smt::AndOp>(loc, ValueRange{notPresent, kSmallerJ});
//       Value k1 = rewriter.create<smt::IteOp>(loc, cond, kPlusOne, k);
//       rewriter.create<scf::YieldOp>(loc, ValueRange{k1});
//     })->getResults();

//     k = l2Results[0];

//     Value jEqualsI = rewriter.create<smt::EqOp>(loc, j, i);
//     Value kEqualsI = rewriter.create<smt::EqOp>(loc, k, i);
//     Value assertInput = rewriter.create<smt::AndOp>(loc, ValueRange{jEqualsI, kEqualsI});
//     Value solver = op->getParentOfType<func::FuncOp>().getArgument(0);
//     rewriter.create<smt::AssertOp>(loc, solver, assertInput);

//     rewriter.replaceOp(op, resArray);
//     return success();
//   }
// };

// NOTE: Produces way to much LLVM code, also after the LLVM is compiled, executing it and solving also takes forever.
// struct JoinOpConversion : OpConversionPattern<JoinOp> {
//   using OpConversionPattern<JoinOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(JoinOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     if (adaptor.getData().size() != 2)
//       return op.emitOpError("only join operations with exactly two operands supported");

//     Type smtIntType = smt::IntegerType::get(getContext());
//     Value arr1 = adaptor.getData()[0];
//     Value arr2 = adaptor.getData()[1];

//     if (auto tupleType = dyn_cast<TupleType>(arr1.getType()))
//       arr1 = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{tupleType.getType(0), tupleType.getType(1)}, ValueRange{arr1})->getResult(1);

//     if (auto tupleType = dyn_cast<TupleType>(arr2.getType()))
//       arr2 = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{tupleType.getType(0), tupleType.getType(1)}, ValueRange{arr2})->getResult(1);

//     Value falseConst = rewriter.create<smt::BoolConstantOp>(loc, false);
//     Value resArray = rewriter.create<smt::ArrayBroadcastOp>(loc, typeConverter->convertType(op.getResult().getType()), falseConst);
//     Value zeroInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 0));
//     Value oneInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 1));

//     Value j = zeroInt;
//     Value resArr = resArray;
//     for (int i = 0; i < UPPER_BOUND;++i) {
//       Location loc = loc;
//       Value idx = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), i));
//       Value present = rewriter.create<smt::ArraySelectOp>(loc, arr1, idx);

//       Value jPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{j, oneInt});
//       j = rewriter.create<smt::IteOp>(loc, present, jPlusOne, j);
//       Value store = rewriter.create<smt::ArrayStoreOp>(loc, resArr, j, present);
//       resArr = rewriter.create<smt::IteOp>(loc, present, store, resArr);
//     }

//     Value k = zeroInt;
//     for (int i = 0; i < UPPER_BOUND;++i) {
//       Location loc = loc;
//       Value idx = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), i));
//       Value present = rewriter.create<smt::ArraySelectOp>(loc, arr2, idx);

//       Value kPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{k, oneInt});
//       k = rewriter.create<smt::IteOp>(loc, present, kPlusOne, k);
//     }

//     auto sameNumElements = rewriter.create<smt::EqOp>(loc, j, k);
//     Value solver = op->getParentOfType<func::FuncOp>().getArgument(0);
//     rewriter.create<smt::AssertOp>(loc, solver, sameNumElements);

//     rewriter.replaceOp(op, resArr);
//     return success();
//   }
// };

struct JoinOpConversion : OpConversionPattern<JoinOp> {
  using OpConversionPattern<JoinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getData().size() != 2)
      return op.emitOpError("only join operations with exactly two operands supported");

    Location loc = op.getLoc();
    Type smtIntType = smt::IntegerType::get(getContext());
    Value arr1 = adaptor.getData()[0];
    Value arr2 = adaptor.getData()[1];

    if (auto tupleType = dyn_cast<TupleType>(arr1.getType()))
      arr1 = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{tupleType.getType(0), tupleType.getType(1)}, ValueRange{arr1})->getResult(1);

    if (auto tupleType = dyn_cast<TupleType>(arr2.getType()))
      arr2 = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{tupleType.getType(0), tupleType.getType(1)}, ValueRange{arr2})->getResult(1);

    Value falseConst = rewriter.create<smt::BoolConstantOp>(loc, false);
    Value resArray = rewriter.create<smt::ArrayBroadcastOp>(loc, typeConverter->convertType(op.getResult().getType()), falseConst);
    Value zeroInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 0));
    Value oneInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 1));
    Value constZero = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), 0));
    Value constOne = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), 1));
    Value upperBound = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), UPPER_BOUND));
    ValueRange arr1Results = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{zeroInt, zeroInt, resArray}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
      Value smtI = iterArgs[0];
      Value k = iterArgs[1];
      Value resArr = iterArgs[2];

      Value present = rewriter.create<smt::ArraySelectOp>(loc, arr1, smtI);

      Value kPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{k, oneInt});
      Value newK = rewriter.create<smt::IteOp>(loc, present, kPlusOne, k);
      Value store = rewriter.create<smt::ArrayStoreOp>(loc, resArr, k, present);
      Value newResArr = rewriter.create<smt::IteOp>(loc, present, store, resArr);

      Value newSmtI = rewriter.create<smt::IntAddOp>(loc, ValueRange{smtI, oneInt});
      rewriter.create<scf::YieldOp>(loc, ValueRange{newSmtI, newK, newResArr});
    })->getResults();

    // Value numElements;
    // if (!arr2.getDefiningOp<SourceOp>()) {
      ValueRange arr2Results = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{zeroInt, zeroInt}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
        // TODO: smtI can be optimized by adding a concrete to symbolic int conversion operation.
        Value smtI = iterArgs[0];
        Value k = iterArgs[1];

        Value present = rewriter.create<smt::ArraySelectOp>(loc, arr2, smtI);
        Value kPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{k, oneInt});
        Value newK = rewriter.create<smt::IteOp>(loc, present, kPlusOne, k);

        Value newSmtI = rewriter.create<smt::IntAddOp>(loc, ValueRange{smtI, oneInt});
        rewriter.create<scf::YieldOp>(loc, ValueRange{newSmtI, newK});
      })->getResults();
    //   numElements = arr2Results[1];
    // } else {
    //   numElements = rewriter.create<smt::IntConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), UPPER_BOUND));
    // }

      auto sameNumElements = rewriter.create<smt::EqOp>(loc, arr1Results[1], arr2Results[1]);
      Value solver = op->getParentOfType<func::FuncOp>().getArgument(0);
      rewriter.create<smt::AssertOp>(loc, solver, sameNumElements);

    rewriter.replaceOp(op, arr1Results[2]);
    return success();
  }
};

struct ConditionalBranchOpConversion : OpConversionPattern<ConditionalBranchOp> {
  using OpConversionPattern<ConditionalBranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConditionalBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type smtIntType = smt::IntegerType::get(getContext());
    Value cond = adaptor.getConditionOperand();
    Value data = adaptor.getDataOperand();

    auto condTuple = dyn_cast<TupleType>(cond.getType());
    auto dataTuple = dyn_cast<TupleType>(data.getType());
    if (!condTuple || !dataTuple)
      return failure();

    ValueRange tmp1 = rewriter.create<UnrealizedConversionCastOp>(loc, condTuple.getTypes(), ValueRange{cond})->getResults();
    ValueRange tmp2 = rewriter.create<UnrealizedConversionCastOp>(loc, dataTuple.getTypes(), ValueRange{data})->getResults();
    Value condData = tmp1[0];
    Value condIndicator = tmp1[1];
    Value dataData = tmp2[0];
    Value dataIndicator = tmp2[1];


    Value falseConst = rewriter.create<smt::BoolConstantOp>(loc, false);
    Value zeroBVDataConst = rewriter.create<smt::ConstantOp>(loc, smt::BitVectorAttr::get(getContext(), 0, cast<smt::ArrayType>(dataData.getType()).getRangeType()));
    Value zeroBVCondConst = rewriter.create<smt::ConstantOp>(loc, smt::BitVectorAttr::get(getContext(), 0, cast<smt::ArrayType>(condData.getType()).getRangeType()));
    Value constFalseArray = rewriter.create<smt::ArrayBroadcastOp>(loc, dataIndicator.getType(), falseConst);
    Value constNullDataArray = rewriter.create<smt::ArrayBroadcastOp>(loc, dataData.getType(), zeroBVDataConst);
    Value constNullCondArray = rewriter.create<smt::ArrayBroadcastOp>(loc, condData.getType(), zeroBVCondConst);
    Value zeroInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 0));
    Value oneInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 1));
    Value constZero = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), 0));
    Value constOne = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), 1));
    Value upperBound = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), UPPER_BOUND));
    ValueRange compactedData = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{zeroInt, zeroInt, constNullDataArray, constFalseArray}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
      Value smtI = iterArgs[0];
      Value k = iterArgs[1];
      Value dataArr = iterArgs[2];
      Value indicatorArr = iterArgs[3];

      Value present = rewriter.create<smt::ArraySelectOp>(loc, dataIndicator, smtI);

      Value kPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{k, oneInt});
      Value newK = rewriter.create<smt::IteOp>(loc, present, kPlusOne, k);
      Value storeInd = rewriter.create<smt::ArrayStoreOp>(loc, indicatorArr, k, present);
      Value newIndResArr = rewriter.create<smt::IteOp>(loc, present, storeInd, indicatorArr);

      Value dataField = rewriter.create<smt::ArraySelectOp>(loc, dataData, smtI);
      Value storeData = rewriter.create<smt::ArrayStoreOp>(loc, dataArr, k, dataField);
      Value newDataResArr = rewriter.create<smt::IteOp>(loc, present, storeData, dataArr);

      Value newSmtI = rewriter.create<smt::IntAddOp>(loc, ValueRange{smtI, oneInt});
      rewriter.create<scf::YieldOp>(loc, ValueRange{newSmtI, newK, newDataResArr, newIndResArr});
    })->getResults();
    ValueRange compactedCond = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{zeroInt, zeroInt, constNullCondArray, constFalseArray}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
      // TODO: smtI can be optimized by adding a concrete to symbolic int conversion operation.
      Value smtI = iterArgs[0];
      Value k = iterArgs[1];
      Value dataArr = iterArgs[2];
      Value indicatorArr = iterArgs[3];

      Value present = rewriter.create<smt::ArraySelectOp>(loc, condIndicator, smtI);
      Value kPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{k, oneInt});
      Value newK = rewriter.create<smt::IteOp>(loc, present, kPlusOne, k);
      Value storeInd = rewriter.create<smt::ArrayStoreOp>(loc, indicatorArr, k, present);
      Value newIndResArr = rewriter.create<smt::IteOp>(loc, present, storeInd, indicatorArr);

      Value dataField = rewriter.create<smt::ArraySelectOp>(loc, condData, smtI);
      Value storeData = rewriter.create<smt::ArrayStoreOp>(loc, dataArr, k, dataField);
      Value newDataResArr = rewriter.create<smt::IteOp>(loc, present, storeData, dataArr);

      Value newSmtI = rewriter.create<smt::IntAddOp>(loc, ValueRange{smtI, oneInt});
      rewriter.create<scf::YieldOp>(loc, ValueRange{newSmtI, newK, newDataResArr, newIndResArr});
    })->getResults();

    auto sameNumElements = rewriter.create<smt::EqOp>(loc, compactedData[1], compactedCond[1]);
    Value solver = op->getParentOfType<func::FuncOp>().getArgument(0);
    rewriter.create<smt::AssertOp>(loc, solver, sameNumElements);

    ValueRange result = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{zeroInt, zeroInt, zeroInt, constNullDataArray, constNullDataArray}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
      // TODO: smtI can be optimized by adding a concrete to symbolic int conversion operation.
      Value smtI = iterArgs[0];
      Value trueInd = iterArgs[1];
      Value falseInd = iterArgs[2];
      Value trueArr = iterArgs[3];
      Value falseArr = iterArgs[4];

      Value cond = rewriter.create<smt::ArraySelectOp>(loc, compactedCond[2], smtI);
      Value data = rewriter.create<smt::ArraySelectOp>(loc, compactedData[2], smtI);

      Value bvOne = rewriter.create<smt::ConstantOp>(loc, smt::BitVectorAttr::get(getContext(), 1, smt::BitVectorType::get(getContext(), 1)));
      Value boolCond = rewriter.create<smt::EqOp>(loc, cond, bvOne);

      Value trueIndPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{trueInd, oneInt});
      Value falseIndPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{falseInd, oneInt});
      Value newTrueInd = rewriter.create<smt::IteOp>(loc, boolCond, trueIndPlusOne, trueInd);
      Value newFalseInd = rewriter.create<smt::IteOp>(loc, boolCond, falseInd, falseIndPlusOne);

      Value newTrueArr = rewriter.create<smt::ArrayStoreOp>(loc, trueArr, trueInd, data);
      Value newFalseArr = rewriter.create<smt::ArrayStoreOp>(loc, falseArr, falseInd, data);
      trueArr = rewriter.create<smt::IteOp>(loc, boolCond, newTrueArr, trueArr);
      falseArr = rewriter.create<smt::IteOp>(loc, boolCond, falseArr, newFalseArr);

      Value newSmtI = rewriter.create<smt::IntAddOp>(loc, ValueRange{smtI, oneInt});
      rewriter.create<scf::YieldOp>(loc, ValueRange{newSmtI, newTrueInd, newFalseInd, trueArr, falseArr});
    })->getResults();

    Value trueRes = rewriter.create<UnrealizedConversionCastOp>(loc, typeConverter->convertType(op.getTrueResult().getType()), ValueRange{result[3], compactedData[3]})->getResult(0);
    Value falseRes = rewriter.create<UnrealizedConversionCastOp>(loc, typeConverter->convertType(op.getTrueResult().getType()), ValueRange{result[4], compactedData[3]})->getResult(0);

    // rewriter.replaceAllUsesWith(op.getTrueResult(), trueRes);
    // rewriter.replaceAllUsesWith(op.getFalseResult(), falseRes);
    rewriter.replaceOp(op, ValueRange{trueRes, falseRes});
    return success();
  }
};

struct ConstantOpConversion : OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (auto intAttr = dyn_cast<IntegerAttr>(adaptor.getValue())) {
      Type bvType = smt::BitVectorType::get(getContext(), cast<IntegerType>(intAttr.getType()).getWidth());
      Value constant = rewriter.create<smt::ConstantOp>(loc, smt::BitVectorAttr::get(getContext(), intAttr.getInt(), bvType));
      Type intType = smt::IntegerType::get(getContext());
      Value dataArray = rewriter.create<smt::ArrayBroadcastOp>(loc, smt::ArrayType::get(getContext(), intType, bvType), constant);
      rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, TypeRange{TupleType::get(getContext(), {dataArray.getType(), typeConverter->convertType(adaptor.getCtrl().getType())})}, ValueRange{dataArray, adaptor.getCtrl()});
      return success();
    }
    return failure();
  }
};

struct BufferOpConversion : OpConversionPattern<BufferOp> {
  using OpConversionPattern<BufferOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperand());
    return success();
  }
};

// struct CombAndOpConversion : OpConversionPattern<comb::AndOp> {
//   using OpConversionPattern<comb::AndOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(comb::AndOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     Location loc = loc;
//     TupleType tupleType = cast<TupleType>(adaptor.getInputs()[0].getType());
//     ValueRange inputs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getInputs()[0])->getResults();

//     rewriter.replaceOp(op, adaptor.getOperand());
//     return success();
//   }
// };

// struct CombXorOpConversion : OpConversionPattern<comb::XorOp> {
//   using OpConversionPattern<comb::XorOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(comb::XorOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     if (adaptor.getInputs().size() != 2)
//       return failure();

//     TupleType tupleType = cast<TupleType>(adaptor.getInputs()[0].getType());
//     ValueRange lhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getInputs()[0])->getResults();
//     ValueRange rhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getInputs()[1])->getResults();
//     Value lhsData = lhs[0];
//     Value lhsIndicator = lhs[1];
//     Value rhsData = rhs[0];
//     Value rhsIndicator = rhs[1];


//     Value solver = op->getParentOfType<func::FuncOp>().getArgument(0);
//     Value constFalse = rewriter.create<smt::ConstantOp>(loc, smt::BitVectorAttr::get(getContext(), 0, cast<smt::ArrayType>(lhsData.getType()).getRangeType()));
//     Value resArr = rewriter.create<smt::ArrayBroadcastOp>(loc, lhsData.getType(), constFalse);
//     for (int i = 0; i < UPPER_BOUND; ++i) {
//       Value idx = rewriter.create<smt::IntConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), i));
//       Value lhsDataElement = rewriter.create<smt::ArraySelectOp>(loc, lhsData, idx);
//       Value rhsDataElement = rewriter.create<smt::ArraySelectOp>(loc, rhsData, idx);
//       Value lhsIndicatorElement = rewriter.create<smt::ArraySelectOp>(loc, lhsIndicator, idx);
//       Value rhsIndicatorElement = rewriter.create<smt::ArraySelectOp>(loc, rhsIndicator, idx);
//       Value sameIndicators = rewriter.create<smt::EqOp>(loc, lhsIndicatorElement, rhsIndicatorElement);
//       rewriter.create<smt::AssertOp>(loc, solver, sameIndicators);
//       Value resVal = rewriter.create<smt::BVXOrOp>(loc, lhsDataElement, rhsDataElement);
//       resArr = rewriter.create<smt::ArrayStoreOp>(loc, resArr, idx, resVal);
//     }

//     Value result = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType, ValueRange{resArr, lhsIndicator})->getResult(0);

//     rewriter.replaceOp(op, result);
//     return success();
//   }
// };

struct CombXorOpConversion : OpConversionPattern<comb::XorOp> {
  using OpConversionPattern<comb::XorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::XorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (adaptor.getInputs().size() != 2)
      return failure();

    TupleType tupleType = cast<TupleType>(adaptor.getInputs()[0].getType());
    ValueRange lhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getInputs()[0])->getResults();
    ValueRange rhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getInputs()[1])->getResults();
    Value lhsData = lhs[0];
    Value lhsIndicator = lhs[1];
    Value rhsData = rhs[0];
    Value rhsIndicator = rhs[1];

    Type smtIntType = smt::IntegerType::get(getContext());

    Value falseConst = rewriter.create<smt::BoolConstantOp>(loc, false);
    // Value zeroBVDataConst = rewriter.create<smt::ConstantOp>(loc, smt::BitVectorAttr::get(getContext(), 0, cast<smt::ArrayType>(dataData.getType()).getRangeType()));
    // Value zeroBVCondConst = rewriter.create<smt::ConstantOp>(loc, smt::BitVectorAttr::get(getContext(), 0, cast<smt::ArrayType>(condData.getType()).getRangeType()));
    // Value constFalseArray = rewriter.create<smt::ArrayBroadcastOp>(loc, dataIndicator.getType(), falseConst);
    // Value constNullDataArray = rewriter.create<smt::ArrayBroadcastOp>(loc, dataData.getType(), zeroBVDataConst);
    // Value constNullCondArray = rewriter.create<smt::ArrayBroadcastOp>(loc, condData.getType(), zeroBVCondConst);
    Value zeroInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 0));
    Value oneInt = rewriter.create<smt::IntConstantOp>(loc, smtIntType, IntegerAttr::get(rewriter.getI32Type(), 1));
    Value constZero = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), 0));
    Value constOne = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), 1));
    Value upperBound = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), UPPER_BOUND));

    Value solver = op->getParentOfType<func::FuncOp>().getArgument(0);
    Value constFalse = rewriter.create<smt::ConstantOp>(loc, smt::BitVectorAttr::get(getContext(), 0, cast<smt::ArrayType>(lhsData.getType()).getRangeType()));
    Value resArr = rewriter.create<smt::ArrayBroadcastOp>(loc, lhsData.getType(), constFalse);
    Value resIndArr = rewriter.create<smt::ArrayBroadcastOp>(loc, lhsIndicator.getType(), falseConst);
    ValueRange loopOuts = rewriter.create<scf::ForOp>(loc, constZero, upperBound, constOne, ValueRange{zeroInt, resArr, resIndArr}, [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
      Value idx = iterArgs[0];
      Value resArr = iterArgs[1];
      Value resIndArr = iterArgs[2];
      // Value idx = rewriter.create<smt::IntConstantOp>(loc, IntegerAttr::get(rewriter.getI32Type(), i));
      Value idxPlusOne = rewriter.create<smt::IntAddOp>(loc, ValueRange{idx, oneInt});
      Value lhsDataElement = rewriter.create<smt::ArraySelectOp>(loc, lhsData, idx);
      Value rhsDataElement = rewriter.create<smt::ArraySelectOp>(loc, rhsData, idx);
      Value lhsIndicatorElement = rewriter.create<smt::ArraySelectOp>(loc, lhsIndicator, idx);
      Value rhsIndicatorElement = rewriter.create<smt::ArraySelectOp>(loc, rhsIndicator, idx);
      Value sameIndicators = rewriter.create<smt::AndOp>(loc, ValueRange{lhsIndicatorElement, rhsIndicatorElement});
      resIndArr = rewriter.create<smt::ArrayStoreOp>(loc, resIndArr, idx, sameIndicators);

      // rewriter.create<smt::AssertOp>(loc, solver, sameIndicators);
      Value resVal = rewriter.create<smt::BVXOrOp>(loc, lhsDataElement, rhsDataElement);
      resArr = rewriter.create<smt::ArrayStoreOp>(loc, resArr, idx, resVal);
      rewriter.create<scf::YieldOp>(loc, ValueRange{idxPlusOne, resArr, resIndArr});
    }).getResults();

    Value result = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType, ValueRange{loopOuts[1], loopOuts[2]})->getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CombXorOpQuantConversion : OpConversionPattern<comb::XorOp> {
  using OpConversionPattern<comb::XorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::XorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (adaptor.getInputs().size() != 2)
      return failure();

    TupleType tupleType = cast<TupleType>(adaptor.getInputs()[0].getType());
    ValueRange lhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getInputs()[0])->getResults();
    ValueRange rhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getInputs()[1])->getResults();
    Value lhsData = lhs[0];
    Value lhsIndicator = lhs[1];
    Value rhsData = rhs[0];
    Value rhsIndicator = rhs[1];

    Type smtIntType = smt::IntegerType::get(getContext());

    Value resData = rewriter.create<smt::DeclareConstOp>(loc, lhsData.getType(), "xor_data");
    Value resIndicator = rewriter.create<smt::DeclareConstOp>(loc, lhsIndicator.getType(), "xor_indicator");
    Value solver = op->getParentOfType<func::FuncOp>().getArgument(0);

    Value dataConstraint = rewriter.create<smt::ForallOp>(loc, smtIntType, SmallVector<StringRef>{"i123"}, [&](OpBuilder &builder, Location loc, ValueRange boundVars) -> Value {
      Value lhsElement = builder.create<smt::ArraySelectOp>(loc, lhsData, boundVars[0]);
      Value rhsElement = builder.create<smt::ArraySelectOp>(loc, rhsData, boundVars[0]);
      Value resElement = builder.create<smt::ArraySelectOp>(loc, resData, boundVars[0]);
      Value xord = rewriter.create<smt::BVXOrOp>(loc, lhsElement, rhsElement);
      return rewriter.create<smt::EqOp>(loc, xord, resElement);
    });
    rewriter.create<smt::AssertOp>(loc, solver, dataConstraint);
    Value indicatorConstraint = rewriter.create<smt::ForallOp>(loc, smtIntType, SmallVector<StringRef>{"i124"}, [&](OpBuilder &builder, Location loc, ValueRange boundVars) -> Value {
      Value lhsElement = builder.create<smt::ArraySelectOp>(loc, lhsIndicator, boundVars[0]);
      Value rhsElement = builder.create<smt::ArraySelectOp>(loc, rhsIndicator, boundVars[0]);
      Value resElement = builder.create<smt::ArraySelectOp>(loc, resIndicator, boundVars[0]);
      Value xord = rewriter.create<smt::AndOp>(loc, ValueRange{lhsElement, rhsElement});
      return rewriter.create<smt::EqOp>(loc, xord, resElement);
    });
    rewriter.create<smt::AssertOp>(loc, solver, indicatorConstraint);

    Value result = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType, ValueRange{resData, resIndicator})->getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CombShrUOpQuantConversion : OpConversionPattern<comb::ShrUOp> {
  using OpConversionPattern<comb::ShrUOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::ShrUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    TupleType tupleType = cast<TupleType>(adaptor.getLhs().getType());
    ValueRange lhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getLhs())->getResults();
    ValueRange rhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getRhs())->getResults();
    Value lhsData = lhs[0];
    Value lhsIndicator = lhs[1];
    Value rhsData = rhs[0];
    Value rhsIndicator = rhs[1];

    Type smtIntType = smt::IntegerType::get(getContext());

    Value resData = rewriter.create<smt::DeclareConstOp>(loc, lhsData.getType(), "shru_data");
    Value resIndicator = rewriter.create<smt::DeclareConstOp>(loc, lhsIndicator.getType(), "shru_indicator");
    Value solver = op->getParentOfType<func::FuncOp>().getArgument(0);

    Value dataConstraint = rewriter.create<smt::ForallOp>(loc, smtIntType, SmallVector<StringRef>{"i223"}, [&](OpBuilder &builder, Location loc, ValueRange boundVars) -> Value {
      Value lhsElement = builder.create<smt::ArraySelectOp>(loc, lhsData, boundVars[0]);
      Value rhsElement = builder.create<smt::ArraySelectOp>(loc, rhsData, boundVars[0]);
      Value resElement = builder.create<smt::ArraySelectOp>(loc, resData, boundVars[0]);
      Value xord = rewriter.create<smt::LShrOp>(loc, lhsElement, rhsElement);
      return rewriter.create<smt::EqOp>(loc, xord, resElement);
    });
    rewriter.create<smt::AssertOp>(loc, solver, dataConstraint);
    Value indicatorConstraint = rewriter.create<smt::ForallOp>(loc, smtIntType, SmallVector<StringRef>{"i224"}, [&](OpBuilder &builder, Location loc, ValueRange boundVars) -> Value {
      Value lhsElement = builder.create<smt::ArraySelectOp>(loc, lhsIndicator, boundVars[0]);
      Value rhsElement = builder.create<smt::ArraySelectOp>(loc, rhsIndicator, boundVars[0]);
      Value resElement = builder.create<smt::ArraySelectOp>(loc, resIndicator, boundVars[0]);
      Value xord = rewriter.create<smt::AndOp>(loc, ValueRange{lhsElement, rhsElement});
      return rewriter.create<smt::EqOp>(loc, xord, resElement);
    });
    rewriter.create<smt::AssertOp>(loc, solver, indicatorConstraint);

    Value result = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType, ValueRange{resData, resIndicator})->getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CombShrSOpQuantConversion : OpConversionPattern<comb::ShrSOp> {
  using OpConversionPattern<comb::ShrSOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::ShrSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    TupleType tupleType = cast<TupleType>(adaptor.getLhs().getType());
    ValueRange lhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getLhs())->getResults();
    ValueRange rhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getRhs())->getResults();
    Value lhsData = lhs[0];
    Value lhsIndicator = lhs[1];
    Value rhsData = rhs[0];
    Value rhsIndicator = rhs[1];

    Type smtIntType = smt::IntegerType::get(getContext());

    Value resData = rewriter.create<smt::DeclareConstOp>(loc, lhsData.getType(), "shru_data");
    Value resIndicator = rewriter.create<smt::DeclareConstOp>(loc, lhsIndicator.getType(), "shru_indicator");
    Value solver = op->getParentOfType<func::FuncOp>().getArgument(0);

    Value dataConstraint = rewriter.create<smt::ForallOp>(loc, smtIntType, SmallVector<StringRef>{"i223"}, [&](OpBuilder &builder, Location loc, ValueRange boundVars) -> Value {
      Value lhsElement = builder.create<smt::ArraySelectOp>(loc, lhsData, boundVars[0]);
      Value rhsElement = builder.create<smt::ArraySelectOp>(loc, rhsData, boundVars[0]);
      Value resElement = builder.create<smt::ArraySelectOp>(loc, resData, boundVars[0]);
      Value xord = rewriter.create<smt::AShrOp>(loc, lhsElement, rhsElement);
      return rewriter.create<smt::EqOp>(loc, xord, resElement);
    });
    rewriter.create<smt::AssertOp>(loc, solver, dataConstraint);
    Value indicatorConstraint = rewriter.create<smt::ForallOp>(loc, smtIntType, SmallVector<StringRef>{"i224"}, [&](OpBuilder &builder, Location loc, ValueRange boundVars) -> Value {
      Value lhsElement = builder.create<smt::ArraySelectOp>(loc, lhsIndicator, boundVars[0]);
      Value rhsElement = builder.create<smt::ArraySelectOp>(loc, rhsIndicator, boundVars[0]);
      Value resElement = builder.create<smt::ArraySelectOp>(loc, resIndicator, boundVars[0]);
      Value xord = rewriter.create<smt::AndOp>(loc, ValueRange{lhsElement, rhsElement});
      return rewriter.create<smt::EqOp>(loc, xord, resElement);
    });
    rewriter.create<smt::AssertOp>(loc, solver, indicatorConstraint);

    Value result = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType, ValueRange{resData, resIndicator})->getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CombMulOpQuantConversion : OpConversionPattern<comb::MulOp> {
  using OpConversionPattern<comb::MulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (adaptor.getInputs().size() != 2)
      return failure();

    TupleType tupleType = cast<TupleType>(adaptor.getInputs()[0].getType());
    ValueRange lhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getInputs()[0])->getResults();
    ValueRange rhs = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType.getTypes(), adaptor.getInputs()[1])->getResults();
    Value lhsData = lhs[0];
    Value lhsIndicator = lhs[1];
    Value rhsData = rhs[0];
    Value rhsIndicator = rhs[1];

    Type smtIntType = smt::IntegerType::get(getContext());

    Value resData = rewriter.create<smt::DeclareConstOp>(loc, lhsData.getType(), "xor_data");
    Value resIndicator = rewriter.create<smt::DeclareConstOp>(loc, lhsIndicator.getType(), "xor_indicator");
    Value solver = op->getParentOfType<func::FuncOp>().getArgument(0);

    Value dataConstraint = rewriter.create<smt::ForallOp>(loc, smtIntType, SmallVector<StringRef>{"i323"}, [&](OpBuilder &builder, Location loc, ValueRange boundVars) -> Value {
      Value lhsElement = builder.create<smt::ArraySelectOp>(loc, lhsData, boundVars[0]);
      Value rhsElement = builder.create<smt::ArraySelectOp>(loc, rhsData, boundVars[0]);
      Value resElement = builder.create<smt::ArraySelectOp>(loc, resData, boundVars[0]);
      Value xord = rewriter.create<smt::MulOp>(loc, lhsElement, rhsElement);
      return rewriter.create<smt::EqOp>(loc, xord, resElement);
    });
    rewriter.create<smt::AssertOp>(loc, solver, dataConstraint);
    Value indicatorConstraint = rewriter.create<smt::ForallOp>(loc, smtIntType, SmallVector<StringRef>{"i324"}, [&](OpBuilder &builder, Location loc, ValueRange boundVars) -> Value {
      Value lhsElement = builder.create<smt::ArraySelectOp>(loc, lhsIndicator, boundVars[0]);
      Value rhsElement = builder.create<smt::ArraySelectOp>(loc, rhsIndicator, boundVars[0]);
      Value resElement = builder.create<smt::ArraySelectOp>(loc, resIndicator, boundVars[0]);
      Value xord = rewriter.create<smt::AndOp>(loc, ValueRange{lhsElement, rhsElement});
      return rewriter.create<smt::EqOp>(loc, xord, resElement);
    });
    rewriter.create<smt::AssertOp>(loc, solver, indicatorConstraint);

    Value result = rewriter.create<UnrealizedConversionCastOp>(loc, tupleType, ValueRange{resData, resIndicator})->getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct JoinOpQuantConversion : OpConversionPattern<JoinOp> {
  using OpConversionPattern<JoinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getData().size() != 2)
      return op.emitOpError("only join operations with exactly two operands supported");

    Location loc = op.getLoc();
    Type smtIntType = smt::IntegerType::get(getContext());
    Value arr1 = adaptor.getData()[0];
    Value arr2 = adaptor.getData()[1];
    Value solver = op->getParentOfType<func::FuncOp>().getArgument(0);

    if (auto tupleType = dyn_cast<TupleType>(arr1.getType()))
      arr1 = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{tupleType.getType(0), tupleType.getType(1)}, ValueRange{arr1})->getResult(1);

    if (auto tupleType = dyn_cast<TupleType>(arr2.getType()))
      arr2 = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{tupleType.getType(0), tupleType.getType(1)}, ValueRange{arr2})->getResult(1);

    Value resIndicator = rewriter.create<smt::DeclareConstOp>(loc, arr1.getType(), "join_indicator");

    Value indicatorConstraint = rewriter.create<smt::ForallOp>(loc, smtIntType, SmallVector<StringRef>{"i125"}, [&](OpBuilder &builder, Location loc, ValueRange boundVars) -> Value {
      Value lhsElement = builder.create<smt::ArraySelectOp>(loc, arr1, boundVars[0]);
      Value rhsElement = builder.create<smt::ArraySelectOp>(loc, arr2, boundVars[0]);
      Value resElement = builder.create<smt::ArraySelectOp>(loc, resIndicator, boundVars[0]);
      Value xord = rewriter.create<smt::AndOp>(loc, ValueRange{lhsElement, rhsElement});
      return rewriter.create<smt::EqOp>(loc, xord, resElement);
    });
    rewriter.create<smt::AssertOp>(loc, solver, indicatorConstraint);

    rewriter.replaceOp(op, resIndicator);
    return success();
  }
};

} // namespace 

//===----------------------------------------------------------------------===//
// Convert Handshake to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertHandshakeToSMTPass
    : public HandshakeToSMTBase<ConvertHandshakeToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertHandshakeToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<comb::CombDialect>();
  target.addIllegalDialect<HandshakeDialect>();
  target.addLegalDialect<smt::SMTDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  converter.addConversion([](IntegerType type) {
    Type bvType = smt::BitVectorType::get(type.getContext(), type.getWidth());
    Type smtIntType = smt::IntegerType::get(type.getContext());
    Type smtBoolType = smt::BoolType::get(type.getContext());
    Type dataArrayType = smt::ArrayType::get(type.getContext(), smtIntType, bvType);
    // TODO: Maybe an enumeration or bitvector type is easier to handle for the
    // SMT solver than an int type for the range here?
    Type indicatorArrayType = smt::ArrayType::get(type.getContext(), smtIntType, smtBoolType);
    return TupleType::get(type.getContext(), {dataArrayType, indicatorArrayType});
  });
  converter.addConversion([](NoneType type) {
    Type smtIntType = smt::IntegerType::get(type.getContext());
    Type smtBoolType = smt::BoolType::get(type.getContext());
    return smt::ArrayType::get(type.getContext(), smtIntType, smtBoolType);
  });
  // Convert SMT types to themselves
  converter.addConversion([](Type type) -> std::optional<Type> {
    std::function<bool(Type)> isSMTType = [&](Type ty) {
      if (isa<smt::BoolType, smt::IntegerType, smt::SolverType, smt::BitVectorType>(ty))
        return true;
      if (auto arrTy = dyn_cast_or_null<smt::ArrayType>(ty)) {
        return isSMTType(arrTy.getDomainType()) && isSMTType(arrTy.getRangeType());
      }
      if (auto tupleTy = dyn_cast_or_null<TupleType>(ty)) {
        for (auto ty : tupleTy.getTypes()) {
          if (!isSMTType(ty))
            return false;
        }
        return true;
      }
      return false;
    };
    if (isSMTType(type))
      return type;
    return std::nullopt;
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
  converter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)->getResult(0);
      });

  patterns.add<ForkOpConversion, SinkOpConversion, SourceOpConversion, FuncOpConversion,
   ReturnOpConversion,
  // UnrealizedOpConversion,
  CombXorOpQuantConversion, CombShrUOpQuantConversion, CombMulOpQuantConversion, CombShrSOpQuantConversion,
   JoinOpQuantConversion, ConditionalBranchOpConversion, ConstantOpConversion, BufferOpConversion>(converter, patterns.getContext());
  // populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
  // populateReconcileUnrealizedCastsPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();

  // ConversionTarget cleanupTarget(getContext());
  // cleanupTarget.addIllegalOp<UnrealizedConversionCastOp>();

  // RewritePatternSet cleanupPatterns(&getContext());
  // populateReconcileUnrealizedCastsPatterns(cleanupPatterns);
  // if (failed(mlir::applyPartialConversion(getOperation(), cleanupTarget, std::move(cleanupPatterns))))
  //   return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::createHandshakeToSMTPass() {
  return std::make_unique<ConvertHandshakeToSMTPass>();
}
