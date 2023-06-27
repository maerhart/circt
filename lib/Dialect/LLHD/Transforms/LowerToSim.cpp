//===- LLHDToLLVM.cpp - LLHD to LLVM Conversion Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main LLHD to LLVM Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace circt::llhd;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Get an existing global string.
static Value getGlobalString(Location loc, OpBuilder &builder,
                             TypeConverter *typeConverter,
                             LLVM::GlobalOp &str) {
  auto voidPtrTy = LLVM::LLVMPointerType::get(builder.getContext());
  auto i32Ty = IntegerType::get(builder.getContext(), 32);

  auto addr = builder.create<LLVM::AddressOfOp>(
      loc, LLVM::LLVMPointerType::get(str.getType()), str.getName());
  auto idx = builder.create<LLVM::ConstantOp>(loc, i32Ty,
                                              builder.getI32IntegerAttr(0));
  std::array<Value, 2> idxs({idx, idx});
  return builder.create<LLVM::GEPOp>(loc, voidPtrTy, addr, idxs);
}

/// Looks up a symbol and inserts a new functino at the beginning of the
/// module's region in case the function does not exists. If
/// insertBodyAndTerminator is set, also adds the entry block and return
/// terminator.
static func::FuncOp getOrInsertFunction(ModuleOp &module,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc, std::string name,
                                        Type signature,
                                        bool insertBodyAndTerminator = false) {
  auto func = module.lookupSymbol<func::FuncOp>(name);
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<func::FuncOp>(loc, name, signature);
    if (insertBodyAndTerminator) {
      func.addEntryBlock();
      OpBuilder b(func.getBody());
      b.create<func::ReturnOp>(loc);
    }
  }
  return func;
}

/// Return a struct type of arrays containing one entry for each RegOp condition
/// that require more than one state of the trigger to infer it (i.e. `both`,
/// `rise` and `fall`).
static LLVM::LLVMStructType getRegStateTy(LLVM::LLVMDialect *dialect,
                                          Operation *entity) {
  SmallVector<Type, 4> types;
  entity->walk([&](llhd::RegOp op) {
    size_t count = 0;
    for (size_t i = 0; i < op.getModes().size(); ++i) {
      auto mode = op.getRegModeAt(i);
      if (mode == llhd::RegMode::fall || mode == llhd::RegMode::rise ||
          mode == llhd::RegMode::both)
        ++count;
    }
    if (count > 0)
      types.push_back(LLVM::LLVMArrayType::get(
          IntegerType::get(dialect->getContext(), 1), count));
  });
  return LLVM::LLVMStructType::getLiteral(dialect->getContext(), types);
}

/// Returns true if the given value is passed as an argument to the destination
/// block of the given WaitOp.
static bool isWaitDestArg(WaitOp op, Value val) {
  for (auto arg : op.getDestOps()) {
    if (arg == val)
      return true;
  }
  return false;
}

// Returns true if the given operation is used as a destination argument in a
// WaitOp.
static bool isWaitDestArg(Operation *op) {
  for (auto user : op->getUsers()) {
    if (auto wait = dyn_cast<WaitOp>(user))
      return isWaitDestArg(wait, op->getResult(0));
  }
  return false;
}

/// Gather the types of values that are used outside of the block they're
/// defined in. An LLVMType structure containing those types, in order of
/// appearance, is returned.
static Type getProcPersistenceTy(LLVM::LLVMDialect *dialect,
                                 TypeConverter *converter, ProcOp &proc) {
  SmallVector<Type, 3> types = SmallVector<Type, 3>();
  proc.walk([&](Operation *op) -> void {
    if (op->isUsedOutsideOfBlock(op->getBlock()) || isWaitDestArg(op)) {
      auto ty = op->getResult(0).getType();
      auto convertedTy = converter->convertType(ty);
      if (ty.isa<PtrType, hw::InOutType>()) {
        // Persist the unwrapped value.
        types.push_back(
            convertedTy.cast<LLVM::LLVMPointerType>().getElementType());
      } else {
        // Persist the value as is.
        types.push_back(convertedTy);
      }
    }
  });

  // Also persist block arguments escaping their defining block.
  for (auto &block : proc.getBlocks()) {
    // Skip entry block (contains the function signature in its args).
    if (block.isEntryBlock())
      continue;

    for (auto arg : block.getArguments()) {
      if (arg.isUsedOutsideOfBlock(&block))
        types.push_back(converter->convertType(arg.getType()));
    }
  }

  return LLVM::LLVMStructType::getLiteral(dialect->getContext(), types);
}

/// Insert a comparison block that either jumps to the trueDest block, if the
/// resume index mathces the current index, or to falseDest otherwise. If no
/// falseDest is provided, the next block is taken insead.
static void insertComparisonBlock(ConversionPatternRewriter &rewriter,
                                  LLVM::LLVMDialect *dialect, Location loc,
                                  Region *body, Value resumeIdx, int currIdx,
                                  Block *trueDest, ValueRange trueDestArgs,
                                  Block *falseDest = nullptr) {
  auto i32Ty = IntegerType::get(dialect->getContext(), 32);
  auto secondBlock = ++body->begin();
  auto newBlock = rewriter.createBlock(body, secondBlock);
  auto cmpIdx = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(currIdx));
  auto cmpRes = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                              resumeIdx, cmpIdx);

  // Default to jumping to the next block for the false case, if no explicit
  // block is provided.
  if (!falseDest)
    falseDest = &*secondBlock;

  rewriter.create<LLVM::CondBrOp>(loc, cmpRes, trueDest, trueDestArgs,
                                  falseDest, ValueRange());

  // Redirect the entry block terminator to the new comparison block.
  auto entryTer = body->front().getTerminator();
  entryTer->setSuccessor(newBlock, 0);
}

/// Insert a GEP operation to the pointer of the i-th value in the process
/// persistence table.
static Value gepPersistenceState(LLVM::LLVMDialect *dialect, Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 Type elementTy, int index, Value state) {
  auto i32Ty = IntegerType::get(dialect->getContext(), 32);
  auto zeroC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  auto threeC = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(3));
  auto indC = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(index));
  return rewriter.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(elementTy), state,
      ArrayRef<Value>({zeroC, threeC, indC}));
}

/// Persist a `Value` by storing it into the process persistence table, and
/// substituting the uses that escape the block the operation is defined in with
/// a load from the persistence table.
static void persistValue(LLVM::LLVMDialect *dialect, Location loc,
                         TypeConverter *converter,
                         ConversionPatternRewriter &rewriter, Type stateTy,
                         int &i, Value state, Value persist) {
  auto elemTy = stateTy.cast<LLVM::LLVMStructType>()
                    .getBody()[3]
                    .cast<LLVM::LLVMStructType>()
                    .getBody()[i];

  if (auto arg = persist.dyn_cast<BlockArgument>()) {
    rewriter.setInsertionPointToStart(arg.getParentBlock());
  } else {
    rewriter.setInsertionPointAfter(persist.getDefiningOp());
  }

  Value convPersist = converter->materializeTargetConversion(
      rewriter, loc, converter->convertType(persist.getType()), {persist});

  auto gep0 = gepPersistenceState(dialect, loc, rewriter, elemTy, i, state);

  Value toStore;
  if (auto ptr = persist.getType().dyn_cast<PtrType>()) {
    // Unwrap the pointer and store it's value.
    auto elemTy = converter->convertType(ptr.getUnderlyingType());
    toStore = rewriter.create<LLVM::LoadOp>(loc, elemTy, convPersist);
  } else if (persist.getType().isa<hw::InOutType>()) {
    // Unwrap and store the signal struct.
    toStore = rewriter.create<LLVM::LoadOp>(loc, getLLVMSigType(dialect),
                                            convPersist);
  } else {
    // Store the value directly.
    toStore = convPersist;
  }

  rewriter.create<LLVM::StoreOp>(loc, toStore, gep0);

  // Load the value from the persistence table and substitute the original
  // use with it, whenever it is in a different block.
  for (auto &use : llvm::make_early_inc_range(persist.getUses())) {
    auto user = use.getOwner();
    if (persist.getType().isa<PtrType>() && user != toStore.getDefiningOp() &&
        user != convPersist.getDefiningOp() &&
        persist.getParentBlock() == user->getBlock()) {
      // Redirect uses of the pointer in the same block to the pointer in the
      // persistence state. This ensures that stores and loads all operate on
      // the same value.
      use.set(gep0);
    } else if (persist.getParentBlock() != user->getBlock() ||
               (isa<WaitOp>(user) &&
                isWaitDestArg(cast<WaitOp>(user), persist))) {
      // The destination args of a wait op have to be loaded in the entry block
      // of the function, before jumping to the resume destination, so they can
      // be passed as block arguments by the comparison block.
      if (isa<WaitOp>(user) && isWaitDestArg(cast<WaitOp>(user), persist))
        rewriter.setInsertionPoint(
            user->getParentRegion()->front().getTerminator());
      else
        rewriter.setInsertionPointToStart(user->getBlock());

      auto gep1 = gepPersistenceState(dialect, loc, rewriter, elemTy, i, state);
      // Use the pointer in the state struct directly for pointer and signal
      // types.
      if (persist.getType().isa<PtrType, hw::InOutType>()) {
        use.set(gep1);
      } else {
        auto load1 = rewriter.create<LLVM::LoadOp>(loc, elemTy, gep1);
        // Load the value otherwise.
        use.set(load1);
      }
    }
  }
  i++;
}

/// Insert the blocks and operations needed to persist values across suspension,
/// as well as ones needed to resume execution at the right spot.
static void insertPersistence(TypeConverter *converter,
                              ConversionPatternRewriter &rewriter,
                              LLVM::LLVMDialect *dialect, Location loc,
                              ProcOp &proc, Type &stateTy,
                              LLVM::LLVMFuncOp &converted,
                              Operation *splitEntryBefore) {
  auto i32Ty = IntegerType::get(dialect->getContext(), 32);

  auto &firstBB = converted.getBody().front();

  // Split entry block such that all the operations contained in it in the
  // original process appear after the comparison blocks.
  auto splitFirst =
      rewriter.splitBlock(&firstBB, splitEntryBefore->getIterator());

  // Insert dummy branch terminator at the new end of the function's entry
  // block.
  rewriter.setInsertionPointToEnd(&firstBB);
  rewriter.create<LLVM::BrOp>(loc, ValueRange(), splitFirst);

  // Load the resume index from the process state argument.
  rewriter.setInsertionPoint(firstBB.getTerminator());
  auto zeroC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  auto oneC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                rewriter.getI32IntegerAttr(1));
  auto gep = rewriter.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(i32Ty), converted.getArgument(1),
      ArrayRef<Value>({zeroC, oneC}));

  auto larg = rewriter.create<LLVM::LoadOp>(loc, i32Ty, gep);

  auto body = &converted.getBody();

  // Insert an abort block as the last block.
  auto abortBlock = rewriter.createBlock(body, body->end());
  rewriter.create<LLVM::ReturnOp>(loc, ValueRange());

  // Redirect the entry block to a first comparison block. If on a first
  // execution, jump to the new (splitted) entry block, else the process is in
  // an illegal state and jump to the abort block.
  insertComparisonBlock(rewriter, dialect, loc, body, larg, 0, splitFirst,
                        ValueRange(), abortBlock);

  // Keep track of the index in the presistence table of the operation we
  // are currently processing.
  int i = 0;
  // Keep track of the current resume index for comparison blocks.
  int waitInd = 0;

  // Insert operations required to persist values across process suspension.
  converted.walk([&](Operation *op) -> void {
    if ((op->isUsedOutsideOfBlock(op->getBlock()) || isWaitDestArg(op)) &&
        op->getResult(0) != larg.getResult()) {
      persistValue(dialect, loc, converter, rewriter, stateTy, i,
                   converted.getArgument(1), op->getResult(0));
    }

    // Insert a comparison block for wait operations.
    if (auto wait = dyn_cast<WaitOp>(op)) {
      insertComparisonBlock(rewriter, dialect, loc, body, larg, ++waitInd,
                            wait.getDest(), wait.getDestOps());

      // Insert the resume index update at the wait operation location.
      rewriter.setInsertionPoint(op);
      auto procState = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(1);
      auto resumeIdxC = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(waitInd));
      auto resumeIdxPtr = rewriter.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(i32Ty), procState,
          ArrayRef<Value>({zeroC, oneC}));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), resumeIdxC, resumeIdxPtr);
    }
  });

  // Also persist argument blocks escaping their defining block.
  for (auto &block : converted.getBlocks()) {
    // Skip entry block as it contains the function signature.
    if (block.isEntryBlock())
      continue;

    for (auto arg : block.getArguments()) {
      if (arg.isUsedOutsideOfBlock(&block)) {
        persistValue(dialect, loc, converter, rewriter, stateTy, i,
                     converted.getArgument(1), arg);
      }
    }
  }
}

/// Create a zext operation by one bit on the given value. This is useful when
/// passing unsigned indexes to a GEP instruction, which treats indexes as
/// signed values, to avoid unexpected "sign overflows".
static Value zextByOne(Location loc, ConversionPatternRewriter &rewriter,
                       Value value) {
  auto valueTy = value.getType();
  auto zextTy = IntegerType::get(valueTy.getContext(),
                                 valueTy.getIntOrFloatBitWidth() + 1);
  return rewriter.create<LLVM::ZExtOp>(loc, zextTy, value);
}

/// Adjust the bithwidth of value to be the same as targetTy's bitwidth.
static Value adjustBitWidth(Location loc, ConversionPatternRewriter &rewriter,
                            Type targetTy, Value value) {
  auto valueWidth = value.getType().getIntOrFloatBitWidth();
  auto targetWidth = targetTy.getIntOrFloatBitWidth();

  if (valueWidth < targetWidth)
    return rewriter.create<LLVM::ZExtOp>(loc, targetTy, value);

  if (valueWidth > targetWidth)
    return rewriter.create<LLVM::TruncOp>(loc, targetTy, value);

  return value;
}

static unsigned getIndexOfOperandResult(Operation *op, Value result) {
  for (unsigned j = 0, e = op->getNumResults(); j < e; ++j) {
    if (result == result.getDefiningOp()->getResult(j))
      return j;
  }
  llvm_unreachable(
      "no way to recurse to an operation that does not return any value");
}

/// Recursively clone the init origin of a sig operation into the init function,
/// up to the initial constant value(s). This is required to clone the
/// initialization of array and struct signals, where the init operand cannot
/// originate from a constant operation.
static Value recursiveCloneInit(OpBuilder &initBuilder, IRMapping &mapping,
                                Value init) {
  SmallVector<Value> clonedOperands;
  Operation *initOp = init.getDefiningOp();

  // If we end up at a value that we get via BlockArgument or as a result of a
  // llhd.prb op, return a nullptr to signal that something went wrong, because
  // these cases are not supported.
  if (!initOp || isa<llhd::PrbOp>(initOp))
    return nullptr;

  for (size_t i = 0, e = initOp->getNumOperands(); i < e; ++i) {
    Value operand = initOp->getOperand(i);

    // If we have some value that is used multiple times (e.g., broadcasted to
    // an array) then don't emit the ops to create this value several times,
    // but instead remember the cloned value and use it again.
    if (auto memorizedOperand = mapping.lookupOrNull(operand)) {
      clonedOperands.push_back(memorizedOperand);
      continue;
    }

    // Recursively follow operands.
    Value clonedOperand = recursiveCloneInit(initBuilder, mapping, operand);
    if (!clonedOperand)
      return nullptr;

    mapping.map(operand, clonedOperand);
    clonedOperands.push_back(clonedOperand);
  }

  Operation *clone = initOp->clone();
  clone->setOperands(clonedOperands);

  // If we have cloned an operation that returns several values, we have to
  // find the result value of the cloned operation we want to return.
  unsigned index = getIndexOfOperandResult(initOp, init);
  return initBuilder.insert(clone)->getResult(index);
}

//===----------------------------------------------------------------------===//
// Unit conversions
//===----------------------------------------------------------------------===//

namespace {
/// Convert an `llhd.proc` operation to LLVM dialect. This inserts the required
/// logic to resume execution after an `llhd.wait` operation, as well as state
/// keeping for values that need to persist across suspension.
struct ProcOpConversion : public ConvertOpToLLVMPattern<llhd::ProcOp> {
  using ConvertOpToLLVMPattern<llhd::ProcOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(llhd::ProcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Collect used llvm types.
    auto voidTy = getVoidType();
    auto voidPtrTy = getVoidPtrType();
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto senseTableTy = LLVM::LLVMPointerType::get(
        LLVM::LLVMArrayType::get(i1Ty, procOp.getNumArguments()));
    auto stateTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(),
        {/* current instance  */ i32Ty, /* resume index */ i32Ty,
         /* sense flags */ senseTableTy, /* persistent types */
         getProcPersistenceTy(&getDialect(), typeConverter, procOp)});
    auto sigTy = getLLVMSigType(&getDialect());

    // Keep track of the original first operation of the process, to know where
    // to split the first block to insert comparison blocks.
    auto &firstOp = op->getRegion(0).front().front();

    // Have an intermediate signature conversion to add the arguments for the
    // state, process-specific state and signal table.
    LLVMTypeConverter::SignatureConversion intermediate(
        procOp.getNumArguments());
    // Add state, process state table and signal table arguments.
    std::array<Type, 3> procArgTys({voidPtrTy,
                                    LLVM::LLVMPointerType::get(stateTy),
                                    LLVM::LLVMPointerType::get(sigTy)});
    intermediate.addInputs(procArgTys);
    for (size_t i = 0, e = procOp.getNumArguments(); i < e; ++i)
      intermediate.addInputs(i, voidTy);
    rewriter.applySignatureConversion(&procOp.getBody(), intermediate,
                                      typeConverter);

    // Get the final signature conversion.
    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&procOp.getBlocks().front());
    LLVMTypeConverter::SignatureConversion final(
        intermediate.getConvertedTypes().size());
    final.addInputs(0, voidPtrTy);
    final.addInputs(1, LLVM::LLVMPointerType::get(stateTy));
    final.addInputs(2, LLVM::LLVMPointerType::get(sigTy));

    for (size_t i = 0, e = procOp.getNumArguments(); i < e; ++i) {
      // Create gep operations from the signal table for each original argument.
      auto index = bodyBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto gep = bodyBuilder.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(sigTy),
          procOp.getArgument(2), ArrayRef<Value>({index}));

      // Remap the i-th original argument to the gep'd value.
      final.remapInput(i + 3, gep.getResult());
    }

    // Get the converted process signature.
    auto funcTy = rewriter.getFunctionType(
        {}, {voidPtrTy, LLVM::LLVMPointerType::get(stateTy),
             LLVM::LLVMPointerType::get(sigTy)});
    // Create a new llvm function to house the lowered process.
    auto func =
        rewriter.create<func::FuncOp>(op->getLoc(), procOp.getName(), funcTy);

    // Inline the process region in the new llvm function.
    rewriter.inlineRegionBefore(procOp.getBody(), func.getBody(), func.end());

    insertPersistence(typeConverter, rewriter, &getDialect(), op->getLoc(),
                      procOp, stateTy, llvmFunc, &firstOp);

    // Convert the block argument types after inserting the persistence, as this
    // would otherwise interfere with the persistence generation.
    if (failed(rewriter.convertRegionTypes(&llvmFunc.getBody(), *typeConverter,
                                           &final)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
/// Convert an `llhd.halt` operation to LLVM dialect. This zeroes out all the
/// senses and returns, effectively making the process unable to be invoked
/// again.
struct HaltOpConversion : public ConvertToLLVMPattern {
  explicit HaltOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(HaltOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);

    auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto procState = llvmFunc.getArgument(1);
    auto senseTableTy = procState.getType()
                            .cast<LLVM::LLVMPointerType>()
                            .getElementType()
                            .cast<LLVM::LLVMStructType>()
                            .getBody()[2]
                            .cast<LLVM::LLVMPointerType>()
                            .getElementType()
                            .cast<LLVM::LLVMArrayType>();

    // Get senses ptr from the process state argument.
    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
    auto twoC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(2));
    auto sensePtrGep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(),
        LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(senseTableTy)),
        procState, ArrayRef<Value>({zeroC, twoC}));
    auto sensePtr = rewriter.create<LLVM::LoadOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(senseTableTy), sensePtrGep);

    // Zero out all the senses flags.
    for (size_t i = 0, e = senseTableTy.getNumElements(); i < e; ++i) {
      auto indC = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto zeroB = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getI32IntegerAttr(0));
      auto senseElemPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), sensePtr,
          ArrayRef<Value>({zeroC, indC}));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), zeroB, senseElemPtr);
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange());
    return success();
  }
};
} // namespace

namespace {
/// Convert an `llhd.wait` operation to LLVM dialect. This sets the current
/// resume point, sets the observed senses (if present) and schedules the timed
/// wake up (if present).
struct WaitOpConversion : public ConvertToLLVMPattern {
  explicit WaitOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(WaitOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto waitOp = cast<WaitOp>(op);
    WaitOpAdaptor transformed(operands, op->getAttrDictionary());
    auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>();

    auto voidTy = getVoidType();
    auto voidPtrTy = getVoidPtrType();
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

    // Get the llhdSuspend runtime function.
    auto llhdSuspendTy = LLVM::LLVMFunctionType::get(
        voidTy, {voidPtrTy, voidPtrTy, i64Ty, i64Ty, i64Ty});
    auto module = op->getParentOfType<ModuleOp>();
    auto llhdSuspendFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                               "llhdSuspend", llhdSuspendTy);

    auto statePtr = llvmFunc.getArgument(0);
    auto procState = llvmFunc.getArgument(1);
    auto procStateTy = procState.getType();
    auto senseTableTy = procStateTy.cast<LLVM::LLVMPointerType>()
                            .getElementType()
                            .cast<LLVM::LLVMStructType>()
                            .getBody()[2]
                            .cast<LLVM::LLVMPointerType>()
                            .getElementType();

    // Get senses ptr.
    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
    auto twoC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(2));
    auto sensePtrGep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(),
        LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(senseTableTy)),
        procState, ArrayRef<Value>({zeroC, twoC}));
    auto sensePtr = rewriter.create<LLVM::LoadOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(senseTableTy), sensePtrGep);

    // Reset sense table, if not all signals are observed.
    if (waitOp.getObs().size() <
        senseTableTy.cast<LLVM::LLVMArrayType>().getNumElements()) {
      auto zeroB = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getBoolAttr(false));
      for (size_t i = 0,
                  e = senseTableTy.cast<LLVM::LLVMArrayType>().getNumElements();
           i < e; ++i) {
        auto indC = rewriter.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
        auto senseElemPtr = rewriter.create<LLVM::GEPOp>(
            op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), sensePtr,
            ArrayRef<Value>({zeroC, indC}));
        rewriter.create<LLVM::StoreOp>(op->getLoc(), zeroB, senseElemPtr);
      }
    }

    // Set sense flags for observed signals.
    for (auto observed : transformed.getObs()) {
      auto instIndexPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(i64Ty), observed,
          ArrayRef<Value>({zeroC, twoC}));
      auto instIndex =
          rewriter.create<LLVM::LoadOp>(op->getLoc(), i64Ty, instIndexPtr);
      auto oneB = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty,
                                                    rewriter.getBoolAttr(true));
      auto senseElementPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), sensePtr,
          ArrayRef<Value>({zeroC, instIndex}));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), oneB, senseElementPtr);
    }

    // Update and store the new resume index in the process state.
    auto procStateBC =
        rewriter.create<LLVM::BitcastOp>(op->getLoc(), voidPtrTy, procState);

    // Spawn scheduled event, if present.
    if (waitOp.getTime()) {
      auto realTime = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), transformed.getTime(), 0);
      auto delta = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), transformed.getTime(), 1);
      auto eps = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), transformed.getTime(), 2);

      std::array<Value, 5> args({statePtr, procStateBC, realTime, delta, eps});
      rewriter.create<LLVM::CallOp>(op->getLoc(), std::nullopt,
                                    SymbolRefAttr::get(llhdSuspendFunc), args);
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange());
    return success();
  }
};
} // namespace
namespace {
/// Lower an llhd.inst operation to LLVM dialect. This generates malloc calls
/// and allocSignal calls (to store the pointer into the state) for each signal
/// in the instantiated entity.
struct HWInstOpConversion : public ConvertOpToLLVMPattern<hw::InstanceOp> {
  using ConvertOpToLLVMPattern<hw::InstanceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the parent module.
    auto module = op->getParentOfType<ModuleOp>();
    auto entity = op->getParentOfType<hw::HWModuleOp>();

    auto voidTy = getVoidType();
    auto voidPtrTy = getVoidPtrType();
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

    // Init function signature: (i8* %state) -> void.
    auto initFuncTy = rewriter.getFunctionType(voidTy, {voidPtrTy});
    auto initFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(), "llhd_init",
                            initFuncTy, /*insertBodyAndTerminator=*/true);

    // Get or insert the malloc function definition.
    // Malloc function signature: (i64 %size) -> i8* %pointer.
    auto mallocSigFuncTy = rewriter.getFunctionType(voidPtrTy, {i64Ty});
    auto mallFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                        "malloc", mallocSigFuncTy);

    // Get or insert the allocSignal library call definition.
    // allocSignal function signature: (i8* %state, i8* %sig_name, i8*
    // %sig_owner, i32 %value) -> i32 %sig_index.
    auto allocSigFuncTy = LLVM::LLVMFunctionType::get(
        i32Ty, {voidPtrTy, i32Ty, voidPtrTy, voidPtrTy, i64Ty});
    auto sigFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                       "allocSignal", allocSigFuncTy);

    // Add information about the elements of an array signal to the state.
    // Signature: (i8* state, i32 signalIndex, i32 size, i32 numElements) ->
    // void
    auto addSigArrElemFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, i32Ty, i32Ty, i32Ty});
    auto addSigElemFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(),
                            "addSigArrayElements", addSigArrElemFuncTy);

    // Add information about one element of a struct signal to the state.
    // Signature: (i8* state, i32 signalIndex, i32 offset, i32 size) -> void
    auto addSigStructElemFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, i32Ty, i32Ty, i32Ty});
    auto addSigStructFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(),
                            "addSigStructElement", addSigStructElemFuncTy);

    // Get or insert allocProc library call definition.
    auto allocProcFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, voidPtrTy, voidPtrTy});
    auto allocProcFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                             "allocProc", allocProcFuncTy);

    // Get or insert allocEntity library call definition.
    auto allocEntityFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, voidPtrTy, voidPtrTy});
    auto allocEntityFunc = getOrInsertFunction(
        module, rewriter, op->getLoc(), "allocEntity", allocEntityFuncTy);

    Value initStatePtr = initFunc.getArgument(0);

    // Get a builder for the init function.
    OpBuilder initBuilder =
        OpBuilder::atBlockTerminator(&initFunc.getBody().getBlocks().front());

    // Use the instance name to retrieve the instance from the state.
    auto ownerName = entity.getName().str() + "." + instOp.getName().str();

    // Get or create owner name string
    Value owner;
    auto parentSym =
        module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName);
    if (!parentSym) {
      owner = LLVM::createGlobalString(
          op->getLoc(), initBuilder, "instance." + ownerName, ownerName + '\0',
          LLVM::Linkage::Internal, /*useOpaquePointers=*/true);
      parentSym = module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName);
    } else {
      owner =
          getGlobalString(op->getLoc(), initBuilder, typeConverter, parentSym);
    }

    // Handle entity instantiation.
    if (auto child = module.lookupSymbol<hw::HWModuleOp>(instOp.getCallee())) {
      auto regStateTy = getRegStateTy(&getDialect(), child.getOperation());
      auto regStatePtrTy = LLVM::LLVMPointerType::get(regStateTy);

      // Get reg state size.
      auto oneC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
      auto regNull =
          initBuilder.create<LLVM::NullOp>(op->getLoc(), regStatePtrTy);
      auto regGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), regStatePtrTy, regNull, ArrayRef<Value>({oneC}));
      auto regSize =
          initBuilder.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, regGep);

      // Malloc reg state.
      auto regMall = initBuilder
                         .create<LLVM::CallOp>(op->getLoc(), voidPtrTy,
                                               SymbolRefAttr::get(mallFunc),
                                               ArrayRef<Value>({regSize}))
                         .getResult();
      auto regMallBC = initBuilder.create<LLVM::BitcastOp>(
          op->getLoc(), regStatePtrTy, regMall);
      auto zeroB = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getBoolAttr(false));

      // Zero-initialize reg state entries.
      for (size_t i = 0,
                  e = regStateTy.cast<LLVM::LLVMStructType>().getBody().size();
           i < e; ++i) {
        size_t f = regStateTy.cast<LLVM::LLVMStructType>()
                       .getBody()[i]
                       .cast<LLVM::LLVMArrayType>()
                       .getNumElements();
        for (size_t j = 0; j < f; ++j) {
          auto regIndexC = initBuilder.create<LLVM::ConstantOp>(
              op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
          auto triggerIndexC = initBuilder.create<LLVM::ConstantOp>(
              op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(j));
          auto regGep = initBuilder.create<LLVM::GEPOp>(
              op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), regMallBC,
              ArrayRef<Value>({zeroB, regIndexC, triggerIndexC}));
          initBuilder.create<LLVM::StoreOp>(op->getLoc(), zeroB, regGep);
        }
      }

      // Add reg state pointer to global state.
      initBuilder.create<LLVM::CallOp>(
          op->getLoc(), std::nullopt, SymbolRefAttr::get(allocEntityFunc),
          ArrayRef<Value>({initStatePtr, owner, regMall}));

      // Index of the signal in the entity's signal table.
      int initCounter = 0;
      // Walk over the entity and generate mallocs for each one of its signals.
      WalkResult sigWalkResult = child.walk([&](SigOp op) -> WalkResult {
        // if (auto sigOp = dyn_cast<SigOp>(op)) {
        auto underlyingTy = typeConverter->convertType(op.getInit().getType());
        // Get index constant of the signal in the entity's signal table.
        auto indexConst = initBuilder.create<LLVM::ConstantOp>(
            op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(initCounter));
        initCounter++;

        // Clone and insert the operation that defines the signal's init
        // operand (assmued to be a constant/array op)
        IRMapping mapping;
        Value initDef = recursiveCloneInit(initBuilder, mapping, op.getInit());

        if (!initDef)
          return WalkResult::interrupt();

        Value initDefCast = typeConverter->materializeTargetConversion(
            initBuilder, initDef.getLoc(),
            typeConverter->convertType(initDef.getType()), initDef);

        // Compute the required space to malloc.
        auto oneC = initBuilder.create<LLVM::ConstantOp>(
            op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
        auto twoC = initBuilder.create<LLVM::ConstantOp>(
            op.getLoc(), i64Ty, rewriter.getI32IntegerAttr(2));
        auto nullPtr = initBuilder.create<LLVM::NullOp>(
            op.getLoc(), LLVM::LLVMPointerType::get(underlyingTy));
        auto sizeGep = initBuilder.create<LLVM::GEPOp>(
            op.getLoc(), LLVM::LLVMPointerType::get(underlyingTy), nullPtr,
            ArrayRef<Value>(oneC));
        auto size =
            initBuilder.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, sizeGep);
        // Malloc double the required space to make sure signal
        // shifts do not segfault.
        auto mallocSize =
            initBuilder.create<LLVM::MulOp>(op.getLoc(), i64Ty, size, twoC);
        std::array<Value, 1> margs({mallocSize});
        auto mall =
            initBuilder
                .create<LLVM::CallOp>(op.getLoc(), voidPtrTy,
                                      SymbolRefAttr::get(mallFunc), margs)
                .getResult();

        // Store the initial value.
        auto bitcast = initBuilder.create<LLVM::BitcastOp>(
            op.getLoc(), LLVM::LLVMPointerType::get(underlyingTy), mall);

        initBuilder.create<LLVM::StoreOp>(op.getLoc(), initDefCast, bitcast);

        // Get the amount of bytes required to represent an integer underlying
        // type. Use the whole size of the type if not an integer.
        Value passSize;
        if (auto intTy = underlyingTy.dyn_cast<IntegerType>()) {
          auto byteWidth = llvm::divideCeil(intTy.getWidth(), 8);
          passSize = initBuilder.create<LLVM::ConstantOp>(
              op.getLoc(), i64Ty, rewriter.getI64IntegerAttr(byteWidth));
        } else {
          passSize = size;
        }

        std::array<Value, 5> args(
            {initStatePtr, indexConst, owner, mall, passSize});
        auto sigIndex =
            initBuilder
                .create<LLVM::CallOp>(op.getLoc(), i32Ty,
                                      SymbolRefAttr::get(sigFunc), args)
                .getResult();

        // Add structured underlying type information.
        if (auto arrayTy = underlyingTy.dyn_cast<LLVM::LLVMArrayType>()) {
          auto zeroC = initBuilder.create<LLVM::ConstantOp>(
              op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));

          auto numElements = initBuilder.create<LLVM::ConstantOp>(
              op.getLoc(), i32Ty,
              rewriter.getI32IntegerAttr(arrayTy.getNumElements()));

          // Get element size.
          auto null = initBuilder.create<LLVM::NullOp>(
              op.getLoc(), LLVM::LLVMPointerType::get(arrayTy));
          auto gepFirst = initBuilder.create<LLVM::GEPOp>(
              op.getLoc(), LLVM::LLVMPointerType::get(arrayTy.getElementType()),
              null, ArrayRef<Value>({zeroC, oneC}));
          auto toInt = initBuilder.create<LLVM::PtrToIntOp>(op.getLoc(), i32Ty,
                                                            gepFirst);

          // Add information to the state.
          initBuilder.create<LLVM::CallOp>(
              op.getLoc(), std::nullopt, SymbolRefAttr::get(addSigElemFunc),
              ArrayRef<Value>({initStatePtr, sigIndex, toInt, numElements}));
        } else if (auto structTy =
                       underlyingTy.dyn_cast<LLVM::LLVMStructType>()) {
          auto zeroC = initBuilder.create<LLVM::ConstantOp>(
              op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));

          auto null = initBuilder.create<LLVM::NullOp>(
              op.getLoc(), LLVM::LLVMPointerType::get(structTy));
          for (size_t i = 0, e = structTy.getBody().size(); i < e; ++i) {
            auto oneC = initBuilder.create<LLVM::ConstantOp>(
                op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
            auto indexC = initBuilder.create<LLVM::ConstantOp>(
                op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));

            // Get pointer offset.
            auto gepElem = initBuilder.create<LLVM::GEPOp>(
                op.getLoc(), LLVM::LLVMPointerType::get(structTy.getBody()[i]),
                null, ArrayRef<Value>({zeroC, indexC}));
            auto elemToInt = initBuilder.create<LLVM::PtrToIntOp>(
                op.getLoc(), i32Ty, gepElem);

            // Get element size.
            auto elemNull = initBuilder.create<LLVM::NullOp>(
                op.getLoc(), LLVM::LLVMPointerType::get(structTy.getBody()[i]));
            auto gepElemSize = initBuilder.create<LLVM::GEPOp>(
                op.getLoc(), LLVM::LLVMPointerType::get(structTy.getBody()[i]),
                elemNull, ArrayRef<Value>({oneC}));
            auto elemSizeToInt = initBuilder.create<LLVM::PtrToIntOp>(
                op.getLoc(), i32Ty, gepElemSize);

            // Add information to the state.
            initBuilder.create<LLVM::CallOp>(
                op.getLoc(), std::nullopt, SymbolRefAttr::get(addSigStructFunc),
                ArrayRef<Value>(
                    {initStatePtr, sigIndex, elemToInt, elemSizeToInt}));
          }
        }
        return WalkResult::advance();
      });

      if (sigWalkResult.wasInterrupted())
        return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
/// Lower an llhd.inst operation to LLVM dialect. This generates malloc calls
/// and allocSignal calls (to store the pointer into the state) for each signal
/// in the instantiated entity.
struct LLHDInstOpConversion : public ConvertToLLVMPattern {
  explicit LLHDInstOpConversion(MLIRContext *ctx,
                                LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(InstOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the inst operation.
    auto instOp = cast<InstOp>(op);
    // Get the parent module.
    auto module = op->getParentOfType<ModuleOp>();
    auto entity = op->getParentOfType<hw::HWModuleOp>();

    auto voidTy = getVoidType();
    auto voidPtrTy = getVoidPtrType();
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

    // Init function signature: (i8* %state) -> void.
    auto initFuncTy = rewriter.getFunctionType(voidTy, {voidPtrTy});
    auto initFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(), "llhd_init",
                            initFuncTy, /*insertBodyAndTerminator=*/true);

    // Get or insert the malloc function definition.
    // Malloc function signature: (i64 %size) -> i8* %pointer.
    auto mallocSigFuncTy = rewriter.getFunctionType(voidPtrTy, {i64Ty});
    auto mallFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                        "malloc", mallocSigFuncTy);

    // Get or insert the allocSignal library call definition.
    // allocSignal function signature: (i8* %state, i8* %sig_name, i8*
    // %sig_owner, i32 %value) -> i32 %sig_index.
    auto allocSigFuncTy = LLVM::LLVMFunctionType::get(
        i32Ty, {voidPtrTy, i32Ty, voidPtrTy, voidPtrTy, i64Ty});
    auto sigFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                       "allocSignal", allocSigFuncTy);

    // Add information about the elements of an array signal to the state.
    // Signature: (i8* state, i32 signalIndex, i32 size, i32 numElements) ->
    // void
    auto addSigArrElemFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, i32Ty, i32Ty, i32Ty});
    auto addSigElemFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(),
                            "addSigArrayElements", addSigArrElemFuncTy);

    // Add information about one element of a struct signal to the state.
    // Signature: (i8* state, i32 signalIndex, i32 offset, i32 size) -> void
    auto addSigStructElemFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, i32Ty, i32Ty, i32Ty});
    auto addSigStructFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(),
                            "addSigStructElement", addSigStructElemFuncTy);

    // Get or insert allocProc library call definition.
    auto allocProcFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, voidPtrTy, voidPtrTy});
    auto allocProcFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                             "allocProc", allocProcFuncTy);

    // Get or insert allocEntity library call definition.
    auto allocEntityFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, voidPtrTy, voidPtrTy});
    auto allocEntityFunc = getOrInsertFunction(
        module, rewriter, op->getLoc(), "allocEntity", allocEntityFuncTy);

    Value initStatePtr = initFunc.getArgument(0);

    // Get a builder for the init function.
    OpBuilder initBuilder =
        OpBuilder::atBlockTerminator(&initFunc.getBody().getBlocks().front());

    // Use the instance name to retrieve the instance from the state.
    auto ownerName = entity.getName().str() + "." + instOp.getName().str();

    // Get or create owner name string
    Value owner;
    auto parentSym =
        module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName);
    if (!parentSym) {
      owner = LLVM::createGlobalString(
          op->getLoc(), initBuilder, "instance." + ownerName, ownerName + '\0',
          LLVM::Linkage::Internal, /*useOpaquePointers=*/true);
      parentSym = module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName);
    } else {
      owner =
          getGlobalString(op->getLoc(), initBuilder, typeConverter, parentSym);
    }

    // Handle entity instantiation.
    if (auto proc = module.lookupSymbol<ProcOp>(instOp.getCallee())) {
      // Handle process instantiation.
      auto sensesPtrTy = LLVM::LLVMPointerType::get(
          LLVM::LLVMArrayType::get(i1Ty, proc.getNumArguments()));
      auto procStatePtrTy =
          LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getLiteral(
              rewriter.getContext(),
              {i32Ty, i32Ty, sensesPtrTy,
               getProcPersistenceTy(&getDialect(), typeConverter, proc)}));

      auto zeroC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
      auto oneC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
      auto twoC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(2));

      // Malloc space for the process state.
      auto procStateNullPtr =
          initBuilder.create<LLVM::NullOp>(op->getLoc(), procStatePtrTy);
      auto procStateGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), procStatePtrTy, procStateNullPtr,
          ArrayRef<Value>({oneC}));
      auto procStateSize = initBuilder.create<LLVM::PtrToIntOp>(
          op->getLoc(), i64Ty, procStateGep);
      std::array<Value, 1> procStateMArgs({procStateSize});
      auto procStateMall = initBuilder
                               .create<LLVM::CallOp>(
                                   op->getLoc(), voidPtrTy,
                                   SymbolRefAttr::get(mallFunc), procStateMArgs)
                               .getResult();

      auto procStateBC = initBuilder.create<LLVM::BitcastOp>(
          op->getLoc(), procStatePtrTy, procStateMall);

      // Store the initial resume index.
      auto resumeGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(i32Ty), procStateBC,
          ArrayRef<Value>({zeroC, oneC}));
      initBuilder.create<LLVM::StoreOp>(op->getLoc(), zeroC, resumeGep);

      // Malloc space for the senses table.
      auto sensesNullPtr =
          initBuilder.create<LLVM::NullOp>(op->getLoc(), sensesPtrTy);
      auto sensesGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), sensesPtrTy, sensesNullPtr, ArrayRef<Value>({oneC}));
      auto sensesSize =
          initBuilder.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, sensesGep);
      std::array<Value, 1> senseMArgs({sensesSize});
      auto sensesMall =
          initBuilder
              .create<LLVM::CallOp>(op->getLoc(), voidPtrTy,
                                    SymbolRefAttr::get(mallFunc), senseMArgs)
              .getResult();

      auto sensesBC = initBuilder.create<LLVM::BitcastOp>(
          op->getLoc(), sensesPtrTy, sensesMall);

      // Set all initial senses to 1.
      for (size_t i = 0, e = sensesPtrTy.cast<LLVM::LLVMPointerType>()
                                 .getElementType()
                                 .cast<LLVM::LLVMArrayType>()
                                 .getNumElements();
           i < e; ++i) {
        auto oneB = initBuilder.create<LLVM::ConstantOp>(
            op->getLoc(), i1Ty, rewriter.getBoolAttr(true));
        auto gepInd = initBuilder.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
        auto senseGep = initBuilder.create<LLVM::GEPOp>(
            op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), sensesBC,
            ArrayRef<Value>({zeroC, gepInd}));
        initBuilder.create<LLVM::StoreOp>(op->getLoc(), oneB, senseGep);
      }

      // Store the senses pointer in the process state.
      auto procStateSensesPtr = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(sensesPtrTy), procStateBC,
          ArrayRef<Value>({zeroC, twoC}));
      initBuilder.create<LLVM::StoreOp>(op->getLoc(), sensesBC,
                                        procStateSensesPtr);

      std::array<Value, 3> allocProcArgs({initStatePtr, owner, procStateMall});
      initBuilder.create<LLVM::CallOp>(op->getLoc(), std::nullopt,
                                       SymbolRefAttr::get(allocProcFunc),
                                       allocProcArgs);
    }

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Signal conversions
//===----------------------------------------------------------------------===//

namespace {
/// Convert an `llhd.sig` operation to LLVM dialect. The i-th signal of an
/// entity get's lowered to a load of the i-th element of the signal table,
/// passed as an argument.
struct SigOpConversion : public ConvertToLLVMPattern {
  explicit SigOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter,
                           size_t &sigCounter)
      : ConvertToLLVMPattern(llhd::SigOp::getOperationName(), ctx,
                             typeConverter),
        sigCounter(sigCounter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the adapted opreands.
    SigOpAdaptor transformed(operands);

    // Collect the used llvm types.
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto sigTy = getLLVMSigType(&getDialect());

    // Get the signal table pointer from the arguments.
    Value sigTablePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(2);

    // Get the index in the signal table and increase counter.
    auto indexConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(sigCounter));
    ++sigCounter;

    // Insert a gep to the signal index in the signal table argument.
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, LLVM::LLVMPointerType::get(sigTy), sigTablePtr,
        ArrayRef<Value>(indexConst));

    return success();
  }

private:
  size_t &sigCounter;
};
} // namespace

namespace {
/// Convert an `llhd.reg` operation to LLVM dialect. This generates a series of
/// comparisons (blocks) that end up driving the signal with the arguments of
/// the first matching trigger from the trigger list.
struct RegOpConversion : public ConvertToLLVMPattern {
  explicit RegOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter,
                           size_t &regCounter)
      : ConvertToLLVMPattern(RegOp::getOperationName(), ctx, typeConverter),
        regCounter(regCounter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto regOp = cast<RegOp>(op);
    RegOpAdaptor transformed(operands, op->getAttrDictionary());

    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);

    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();

    // Retrieve and update previous trigger values for rising/falling edge
    // detection.
    size_t triggerIndex = 0;
    SmallVector<Value, 4> prevTriggers;
    for (int i = 0, e = regOp.getValues().size(); i < e; ++i) {
      auto mode = regOp.getRegModeAt(i);
      if (mode == RegMode::both || mode == RegMode::fall ||
          mode == RegMode::rise) {
        auto zeroC = rewriter.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
        auto regIndexC = rewriter.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(regCounter));
        auto triggerIndexC = rewriter.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(triggerIndex++));
        auto gep = rewriter.create<LLVM::GEPOp>(
            op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), func.getArgument(1),
            ArrayRef<Value>({zeroC, regIndexC, triggerIndexC}));
        prevTriggers.push_back(
            rewriter.create<LLVM::LoadOp>(op->getLoc(), gep));
        rewriter.create<LLVM::StoreOp>(op->getLoc(),
                                       transformed.getTriggers()[i], gep);
      }
    }

    // Create blocks for drive and continue.
    auto block = op->getBlock();
    auto continueBlock = block->splitBlock(op);

    auto drvBlock = rewriter.createBlock(continueBlock);
    auto valArg = drvBlock->addArgument(transformed.getValues()[0].getType(),
                                        transformed.getValues()[0].getLoc());
    auto delayArg = drvBlock->addArgument(transformed.getDelays()[0].getType(),
                                          transformed.getDelays()[0].getLoc());
    auto gateArg = drvBlock->addArgument(i1Ty, rewriter.getUnknownLoc());

    // Create a drive with the block arguments.
    rewriter.setInsertionPointToStart(drvBlock);
    rewriter.create<DrvOp>(op->getLoc(), regOp.getSignal(), valArg, delayArg,
                           gateArg);
    rewriter.create<LLVM::BrOp>(op->getLoc(), ValueRange(), continueBlock);

    int j = prevTriggers.size() - 1;
    // Create a comparison block for each of the reg tuples.
    for (int i = regOp.getValues().size() - 1, e = i; i >= 0; --i) {
      auto cmpBlock = rewriter.createBlock(block->getNextNode());
      rewriter.setInsertionPointToStart(cmpBlock);

      Value gate;
      if (regOp.hasGate(i)) {
        gate = regOp.getGateAt(i);
      } else {
        gate = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty,
                                                 rewriter.getBoolAttr(true));
      }

      auto drvArgs = std::array<Value, 3>(
          {transformed.getValues()[i], transformed.getDelays()[i], gate});

      RegMode mode = regOp.getRegModeAt(i);

      // Create comparison constants for all modes other than both.
      Value rhs;
      if (mode == RegMode::low || mode == RegMode::fall) {
        rhs = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty,
                                                rewriter.getBoolAttr(false));
      } else if (mode == RegMode::high || mode == RegMode::rise) {
        rhs = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty,
                                                rewriter.getBoolAttr(true));
      }

      // Create comparison for non-both modes.
      Value comp;
      if (rhs)
        comp =
            rewriter.create<LLVM::ICmpOp>(op->getLoc(), LLVM::ICmpPredicate::eq,
                                          transformed.getTriggers()[i], rhs);

      // Create comparison for modes needing more than one state of the trigger.
      Value brCond;
      if (mode == RegMode::rise || mode == RegMode::fall ||
          mode == RegMode::both) {

        auto cmpPrev = rewriter.create<LLVM::ICmpOp>(
            op->getLoc(), LLVM::ICmpPredicate::ne, transformed.getTriggers()[i],
            prevTriggers[j--]);
        if (mode == RegMode::both)
          brCond = cmpPrev;
        else
          brCond =
              rewriter.create<LLVM::AndOp>(op->getLoc(), i1Ty, comp, cmpPrev);
      } else {
        brCond = comp;
      }

      Block *nextBlock;
      nextBlock = cmpBlock->getNextNode();
      // Don't go to next block for last comparison's false branch (skip the
      // drive block).
      if (i == e)
        nextBlock = continueBlock;

      rewriter.create<LLVM::CondBrOp>(op->getLoc(), brCond, drvBlock, drvArgs,
                                      nextBlock, ValueRange());
    }

    rewriter.setInsertionPointToEnd(block);
    rewriter.create<LLVM::BrOp>(op->getLoc(), ArrayRef<Value>(),
                                block->getNextNode());

    rewriter.eraseOp(op);

    ++regCounter;

    return success();
  }

private:
  size_t &regCounter;
}; // namespace
} // namespace

//===----------------------------------------------------------------------===//
// Memory operations
//===----------------------------------------------------------------------===//

namespace {
/// Convert an `hw.module` entity to LLVM dialect. The result is an
/// `llvm.func` which takes a pointer to the global simulation state, a pointer
/// to the entity's local state, and a pointer to the instance's signal table as
/// arguments.
struct HWModuleOpConversion : public ConvertOpToLLVMPattern<hw::HWModuleOp> {
  using ConvertOpToLLVMPattern<hw::HWModuleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::HWModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto sigTy = getInOutType(rewriter.getContext());
    auto entityStatePtrTy =
        LLVM::LLVMPointerType::get(getRegStateTy(&getDialect(), op));

    // Use an intermediate signature conversion to add the arguments for the
    // state and signal table pointer arguments.
    LLVMTypeConverter::SignatureConversion intermediate(op.getNumArguments());
    // Add state and signal table arguments.
    intermediate.addInputs(
        std::array<Type, 3>({getVoidPtrType(), entityStatePtrTy,
                             LLVM::LLVMPointerType::get(sigTy)}));
    for (size_t i = 0, e = op.getNumArguments(); i < e; ++i) {
      intermediate.addInputs(
          i, typeConverter->convertType(op.getArgumentTypes()[i]));
    }

    // rewriter.applySignatureConversion(&op.getBody(), intermediate,
    //                                   typeConverter);

    // OpBuilder bodyBuilder =
    //     OpBuilder::atBlockBegin(&op.getBlocks().front());
    // LLVMTypeConverter::SignatureConversion final(
    //     intermediate.getConvertedTypes().size());
    // final.addInputs(0, voidPtrTy);
    // final.addInputs(1, entityStatePtrTy);
    // final.addInputs(2, LLVM::LLVMPointerType::get(sigTy));

    // // The first n elements of the signal table represent the entity
    // arguments,
    // // while the remaining elements represent the entity's owned signals.
    // for (size_t i = 0, e = op.getNumArguments(); i < e; ++i) {
    //   // Create gep operations from the signal table for each original
    //   argument. auto index = bodyBuilder.create<LLVM::ConstantOp>(
    //       op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
    //   auto gep = bodyBuilder.create<LLVM::GEPOp>(
    //       op->getLoc(), LLVM::LLVMPointerType::get(sigTy),
    //       op.getArgument(2), ArrayRef<Value>(index));
    //   // Remap i-th original argument to the gep'd signal pointer.
    //   final.remapInput(i + 3, gep.getResult());
    // }

    rewriter.applySignatureConversion(&op.getBody(), intermediate,
                                      typeConverter);
    Type returnType = getVoidType();
    if (op.getNumOutputs() == 1) {
      returnType = typeConverter->convertType(op.getOutputPort(0).type);
    } else if (op.getNumOutputs() > 1) {
      SmallVector<Type> structFieldTypes;
      for (unsigned i = 0; i < op.getNumOutputs(); ++i)
        structFieldTypes.push_back(
            typeConverter->convertType(op.getOutputPort(i).type));
      returnType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                    structFieldTypes);
    }

    // Get the converted entity signature.
    auto funcTy = LLVM::LLVMFunctionType::get(returnType,
                                              intermediate.getConvertedTypes());

    // Create the a new llvm function to house the lowered entity.
    auto llvmFunc =
        rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), op.getName(), funcTy);

    // Inline the entity region in the new llvm function.
    rewriter.inlineRegionBefore(op.getBody(), llvmFunc.getBody(),
                                llvmFunc.end());

    // Erase the original operation.
    rewriter.eraseOp(op);
    return success();
  }
};

struct OutputOpConversion : public ConvertOpToLLVMPattern<hw::OutputOp> {
  using ConvertOpToLLVMPattern<hw::OutputOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOutputs());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

namespace {
struct LowerToSimPass : public LowerToSimBase<LowerToSimPass> {
  void runOnOperation() override;
};
} // namespace

void LowerToSimPass::runOnOperation() {
  // Namespace globals;
  // SymbolCache cache;
  // cache.addDefinitions(getOperation());
  // globals.add(cache);

  // Keep a counter to infer a signal's index in his entity's signal table.
  size_t sigCounter = 0;
  // Keep a counter to infer a reg's index in his entity.
  size_t regCounter = 0;

  RewritePatternSet patterns(&getContext());

  // Apply a partial conversion first, lowering only the instances, to generate
  // the init function.
  patterns.add<InstOpConversion>(&getContext(), converter);

  LLVMConversionTarget target(getContext());
  target.addIllegalOp<InstOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Apply the partial conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
  patterns.clear();
  patterns.add<ProcOpConversion, WaitOpConversion, HaltOpConversion>(
      &getContext());
  patterns.add<SigOpConversion, RegOpConversion>(&getContext(), sigCounter);
}

/// Create an LLHD to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::llhd::createLowerToSimPass() {
  return std::make_unique<LowerToSimPass>();
}
