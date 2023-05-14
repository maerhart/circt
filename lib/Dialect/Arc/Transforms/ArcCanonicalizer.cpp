//===- ArcCanonicalizer.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Simulation centric canonicalizations for non-arc operations and
// canonicalizations that require efficient symbol lookups.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "arc-canonicalizer"

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Datastructures
//===----------------------------------------------------------------------===//

/// A combination of SymbolCache and SymbolUserMap that also allows to add users
/// and remove symbols on-demand.
class SymbolHandler : public SymbolCache {
public:
  /// Return the users of the provided symbol operation.
  ArrayRef<Operation *> getUsers(Operation *symbol) const {
    auto it = userMap.find(symbol);
    return it != userMap.end() ? it->second.getArrayRef() : std::nullopt;
  }

  /// Return true if the given symbol has no uses.
  bool useEmpty(Operation *symbol) {
    return !userMap.count(symbol) || userMap[symbol].empty();
  }

  void addUser(Operation *def, Operation *user) {
    assert(isa<mlir::SymbolOpInterface>(def));
    if (!symbolCache.contains(cast<mlir::SymbolOpInterface>(def).getNameAttr()))
      symbolCache.insert(
          {cast<mlir::SymbolOpInterface>(def).getNameAttr(), def});
    userMap[def].insert(user);
  }

  void removeUser(Operation *def, Operation *user) {
    assert(isa<mlir::SymbolOpInterface>(def));
    if (symbolCache.contains(cast<mlir::SymbolOpInterface>(def).getNameAttr()))
      userMap[def].remove(user);
    if (userMap[def].empty())
      userMap.erase(def);
  }

  void removeDefinitionAndAllUsers(Operation *def) {
    assert(isa<mlir::SymbolOpInterface>(def));
    symbolCache.erase(cast<mlir::SymbolOpInterface>(def).getNameAttr());
    userMap.erase(def);
  }

  void collectAllSymbolUses(Operation *symbolTableOp,
                            SymbolTableCollection &symbolTable) {
    // NOTE: the following is almost 1-1 taken from the SymbolUserMap
    // constructor. They made it difficult to extend the implementation by
    // having a lot of members private and non-virtual methods.
    SmallVector<Operation *> symbols;
    auto walkFn = [&](Operation *symbolTableOp, bool allUsesVisible) {
      for (Operation &nestedOp : symbolTableOp->getRegion(0).getOps()) {
        auto symbolUses = SymbolTable::getSymbolUses(&nestedOp);
        assert(symbolUses && "expected uses to be valid");

        for (const SymbolTable::SymbolUse &use : *symbolUses) {
          symbols.clear();
          (void)symbolTable.lookupSymbolIn(symbolTableOp, use.getSymbolRef(),
                                           symbols);
          for (Operation *symbolOp : symbols)
            userMap[symbolOp].insert(use.getUser());
        }
      }
    };
    // We just set `allSymUsesVisible` to false here because it isn't necessary
    // for building the user map.
    SymbolTable::walkSymbolTables(symbolTableOp, /*allSymUsesVisible=*/false,
                                  walkFn);
  }

private:
  DenseMap<Operation *, SetVector<Operation *>> userMap;
};

struct PatternStatistics {
  unsigned removeUnusedArcArgumentsPatternNumArgsRemoved = 0;
};

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
/// A rewrite pattern that has access to a symbol cache to access and modify the
/// symbol-defining op and symbol users as well as a namespace to query new
/// names. Each pattern has to make sure that the symbol handler is kept
/// up-to-date no matter whether the pattern succeeds of fails.
template <typename SourceOp>
class SymOpRewritePattern : public OpRewritePattern<SourceOp> {
public:
  SymOpRewritePattern(MLIRContext *ctxt, SymbolHandler &symbolCache,
                      Namespace &names, PatternStatistics &stats,
                      mlir::PatternBenefit benefit = 1,
                      ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern<SourceOp>(ctxt, benefit, generatedNames), names(names),
        symbolCache(symbolCache), statistics(stats) {}

protected:
  Namespace &names;
  SymbolHandler &symbolCache;
  PatternStatistics &statistics;
};

class MemWritePortEnableAndMaskCanonicalizer
    : public SymOpRewritePattern<MemoryWritePortOp> {
public:
  MemWritePortEnableAndMaskCanonicalizer(
      MLIRContext *ctxt, SymbolHandler &symbolCache, Namespace &names,
      PatternStatistics &stats, DenseMap<StringAttr, StringAttr> &arcMapping)
      : SymOpRewritePattern<MemoryWritePortOp>(ctxt, symbolCache, names, stats),
        arcMapping(arcMapping) {}
  LogicalResult matchAndRewrite(MemoryWritePortOp op,
                                PatternRewriter &rewriter) const final;

private:
  DenseMap<StringAttr, StringAttr> &arcMapping;
};

struct CallPassthroughArc : public SymOpRewritePattern<CallOp> {
  using SymOpRewritePattern::SymOpRewritePattern;
  LogicalResult matchAndRewrite(CallOp op,
                                PatternRewriter &rewriter) const final;
};

struct StatePassthroughArc : public SymOpRewritePattern<StateOp> {
  using SymOpRewritePattern::SymOpRewritePattern;
  LogicalResult matchAndRewrite(StateOp op,
                                PatternRewriter &rewriter) const final;
};

struct RemoveUnusedArcs : public SymOpRewritePattern<DefineOp> {
  using SymOpRewritePattern::SymOpRewritePattern;
  LogicalResult matchAndRewrite(DefineOp op,
                                PatternRewriter &rewriter) const final;
};

struct ICMPCanonicalizer : public OpRewritePattern<comb::ICmpOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(comb::ICmpOp op,
                                PatternRewriter &rewriter) const final;
};

struct RemoveUnusedArcArgumentsPattern : public SymOpRewritePattern<DefineOp> {
  using SymOpRewritePattern::SymOpRewritePattern;
  LogicalResult matchAndRewrite(DefineOp op,
                                PatternRewriter &rewriter) const final;
};

struct SinkArcInputsPattern : public SymOpRewritePattern<DefineOp> {
  using SymOpRewritePattern::SymOpRewritePattern;
  LogicalResult matchAndRewrite(DefineOp op,
                                PatternRewriter &rewriter) const final;
};

class ZeroCountRaising : public OpRewritePattern<comb::MuxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(comb::MuxOp op,
                                PatternRewriter &rewriter) const final;

private:
  using DeltaFunc = std::function<uint32_t(uint32_t, bool)>;
  LogicalResult handleSequenceInitializer(OpBuilder &rewriter, Location loc,
                                          const DeltaFunc &deltaFunc,
                                          bool isLeading, Value falseValue,
                                          Value extractedFrom,
                                          SmallVectorImpl<Value> &arrayElements,
                                          uint32_t &currIndex) const;
};

struct IndexingConstArray : public OpRewritePattern<hw::ArrayGetOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(hw::ArrayGetOp op,
                                PatternRewriter &rewriter) const final;
};

} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

LogicalResult canonicalizePassthoughCall(mlir::CallOpInterface callOp,
                                         SymbolHandler &symbolCache,
                                         PatternRewriter &rewriter) {
  auto defOp = cast<DefineOp>(symbolCache.getDefinition(
      callOp.getCallableForCallee().get<SymbolRefAttr>().getLeafReference()));
  if (defOp.isPassthrough()) {
    symbolCache.removeUser(defOp, callOp);
    rewriter.replaceOp(callOp, callOp.getArgOperands());
    return success();
  }
  return failure();
}

static Value zextUsingConcatOp(OpBuilder &builder, Location loc, Value toZext,
                               uint32_t targetWidth) {
  assert(toZext.getType().isSignlessInteger() &&
         "Can only concatenate integers");

  uint32_t bitWidth = toZext.getType().getIntOrFloatBitWidth();
  assert(bitWidth <= targetWidth && "cannot zext to a smaller bitwidth");

  if (bitWidth == targetWidth)
    return toZext;

  Value zero =
      builder.create<hw::ConstantOp>(loc, APInt(targetWidth - bitWidth, 0));
  return builder.create<comb::ConcatOp>(loc, zero, toZext);
}

//===----------------------------------------------------------------------===//
// Canonicalization pattern implementations
//===----------------------------------------------------------------------===//

LogicalResult MemWritePortEnableAndMaskCanonicalizer::matchAndRewrite(
    MemoryWritePortOp op, PatternRewriter &rewriter) const {
  auto defOp = cast<DefineOp>(symbolCache.getDefinition(op.getArcAttr()));
  APInt enable;

  if (op.getEnable() &&
      mlir::matchPattern(
          defOp.getBodyBlock().getTerminator()->getOperand(op.getEnableIdx()),
          mlir::m_ConstantInt(&enable))) {
    if (enable.isZero()) {
      symbolCache.removeUser(defOp, op);
      rewriter.eraseOp(op);
      if (symbolCache.useEmpty(defOp)) {
        symbolCache.removeDefinitionAndAllUsers(defOp);
        rewriter.eraseOp(defOp);
      }
      return success();
    }
    if (enable.isAllOnes()) {
      if (arcMapping.count(defOp.getNameAttr())) {
        auto arcWithoutEnable = arcMapping[defOp.getNameAttr()];
        // Remove the enable attribute
        rewriter.updateRootInPlace(op, [&]() {
          op.setEnable(false);
          op.setArc(arcWithoutEnable.getValue());
        });
        symbolCache.removeUser(defOp, op);
        symbolCache.addUser(symbolCache.getDefinition(arcWithoutEnable), op);
        return success();
      }

      auto newName = names.newName(defOp.getName());
      auto users = SmallVector<Operation *>(symbolCache.getUsers(defOp));
      symbolCache.removeDefinitionAndAllUsers(defOp);

      // Remove the enable attribute
      rewriter.updateRootInPlace(op, [&]() {
        op.setEnable(false);
        op.setArc(newName);
      });

      auto newResultTypes = op.getArcResultTypes();

      // Create a new arc that acts as replacement for other users
      rewriter.setInsertionPoint(defOp);
      auto newDefOp = rewriter.cloneWithoutRegions(defOp);
      auto *block = rewriter.createBlock(
          &newDefOp.getBody(), newDefOp.getBody().end(),
          newDefOp.getArgumentTypes(),
          SmallVector<Location>(newDefOp.getNumArguments(), defOp.getLoc()));
      auto callOp = rewriter.create<CallOp>(newDefOp.getLoc(), newResultTypes,
                                            newName, block->getArguments());
      SmallVector<Value> results(callOp->getResults());
      Value constTrue = rewriter.create<hw::ConstantOp>(
          newDefOp.getLoc(), rewriter.getI1Type(), 1);
      results.insert(results.begin() + op.getEnableIdx(), constTrue);
      rewriter.create<OutputOp>(newDefOp.getLoc(), results);

      // Remove the enable output from the current arc
      auto *terminator = defOp.getBodyBlock().getTerminator();
      rewriter.updateRootInPlace(
          terminator, [&]() { terminator->eraseOperand(op.getEnableIdx()); });
      rewriter.updateRootInPlace(defOp, [&]() {
        defOp.setName(newName);
        defOp.setFunctionType(
            rewriter.getFunctionType(defOp.getArgumentTypes(), newResultTypes));
      });

      // Update symbol cache
      symbolCache.addDefinition(defOp.getNameAttr(), defOp);
      symbolCache.addDefinition(newDefOp.getNameAttr(), newDefOp);
      symbolCache.addUser(defOp, callOp);
      for (auto *user : users)
        symbolCache.addUser(user == op ? defOp : newDefOp, user);

      arcMapping[newDefOp.getNameAttr()] = defOp.getNameAttr();
      return success();
    }
  }
  return failure();
}

LogicalResult
CallPassthroughArc::matchAndRewrite(CallOp op,
                                    PatternRewriter &rewriter) const {
  return canonicalizePassthoughCall(op, symbolCache, rewriter);
}

LogicalResult
StatePassthroughArc::matchAndRewrite(StateOp op,
                                     PatternRewriter &rewriter) const {
  if (op.getLatency() == 0)
    return canonicalizePassthoughCall(op, symbolCache, rewriter);
  return failure();
}

LogicalResult
RemoveUnusedArcs::matchAndRewrite(DefineOp op,
                                  PatternRewriter &rewriter) const {
  if (symbolCache.useEmpty(op)) {
    op.getBody().walk([&](mlir::CallOpInterface user) {
      if (auto symbol = user.getCallableForCallee().dyn_cast<SymbolRefAttr>())
        if (auto *defOp = symbolCache.getDefinition(symbol.getLeafReference()))
          symbolCache.removeUser(defOp, user);
    });
    symbolCache.removeDefinitionAndAllUsers(op);
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

LogicalResult
ICMPCanonicalizer::matchAndRewrite(comb::ICmpOp op,
                                   PatternRewriter &rewriter) const {
  auto getConstant = [&](const APInt &constant) -> Value {
    return rewriter.create<hw::ConstantOp>(op.getLoc(), constant);
  };
  auto sameWidthIntegers = [](TypeRange types) -> std::optional<unsigned> {
    if (llvm::all_equal(types) && !types.empty())
      if (auto intType = dyn_cast<IntegerType>(*types.begin()))
        return intType.getWidth();
    return std::nullopt;
  };
  auto negate = [&](Value input) -> Value {
    auto constTrue = rewriter.create<hw::ConstantOp>(op.getLoc(), APInt(1, 1));
    return rewriter.create<comb::XorOp>(op.getLoc(), input, constTrue,
                                        op.getTwoState());
  };

  APInt rhs;
  if (matchPattern(op.getRhs(), mlir::m_ConstantInt(&rhs))) {
    if (auto concatOp = op.getLhs().getDefiningOp<comb::ConcatOp>()) {
      if (auto optionalWidth =
              sameWidthIntegers(concatOp->getOperands().getTypes())) {
        if ((op.getPredicate() == comb::ICmpPredicate::eq ||
             op.getPredicate() == comb::ICmpPredicate::ne) &&
            rhs.isAllOnes()) {
          Value andOp = rewriter.create<comb::AndOp>(
              op.getLoc(), concatOp.getInputs(), op.getTwoState());
          if (*optionalWidth == 1) {
            if (op.getPredicate() == comb::ICmpPredicate::ne)
              andOp = negate(andOp);
            rewriter.replaceOp(op, andOp);
            return success();
          }
          rewriter.replaceOpWithNewOp<comb::ICmpOp>(
              op, op.getPredicate(), andOp,
              getConstant(APInt(*optionalWidth, rhs.getZExtValue())),
              op.getTwoState());
          return success();
        }

        if ((op.getPredicate() == comb::ICmpPredicate::ne ||
             op.getPredicate() == comb::ICmpPredicate::eq) &&
            rhs.isZero()) {
          Value orOp = rewriter.create<comb::OrOp>(
              op.getLoc(), concatOp.getInputs(), op.getTwoState());
          if (*optionalWidth == 1) {
            if (op.getPredicate() == comb::ICmpPredicate::eq)
              orOp = negate(orOp);
            rewriter.replaceOp(op, orOp);
            return success();
          }
          rewriter.replaceOpWithNewOp<comb::ICmpOp>(
              op, op.getPredicate(), orOp,
              getConstant(APInt(*optionalWidth, rhs.getZExtValue())),
              op.getTwoState());
          return success();
        }
      }
    }
  }
  return failure();
}

LogicalResult RemoveUnusedArcArgumentsPattern::matchAndRewrite(
    DefineOp op, PatternRewriter &rewriter) const {
  BitVector toDelete(op.getNumArguments());
  for (auto [i, arg] : llvm::enumerate(op.getArguments()))
    if (arg.use_empty())
      toDelete.set(i);

  if (toDelete.none())
    return failure();

  // Collect the mutable callers in a first iteration. If there is a user that
  // does not implement the interface, we have to abort the rewrite and have to
  // make sure that we didn't change anything so far.
  SmallVector<CallOpMutableInterface> mutableUsers;
  for (auto *user : symbolCache.getUsers(op)) {
    auto callOpMutable = dyn_cast<CallOpMutableInterface>(user);
    if (!callOpMutable)
      return failure();
    mutableUsers.push_back(callOpMutable);
  }

  // Do the actual rewrites.
  for (auto user : mutableUsers)
    for (int i = toDelete.size() - 1; i >= 0; --i)
      if (toDelete[i])
        user.getArgOperandsMutable().erase(i);

  op.eraseArguments(toDelete);
  op.setFunctionType(
      rewriter.getFunctionType(op.getArgumentTypes(), op.getResultTypes()));

  statistics.removeUnusedArcArgumentsPatternNumArgsRemoved += toDelete.count();
  return success();
}

LogicalResult
SinkArcInputsPattern::matchAndRewrite(DefineOp op,
                                      PatternRewriter &rewriter) const {
  // First check that all users implement the interface we need to be able to
  // modify the users.
  auto users = symbolCache.getUsers(op);
  if (llvm::any_of(
          users, [](auto *user) { return !isa<CallOpMutableInterface>(user); }))
    return failure();

  // Find all arguments that use constant operands only.
  SmallVector<Operation *> stateConsts(op.getNumArguments());
  bool first = true;
  for (auto *user : users) {
    auto callOp = cast<CallOpMutableInterface>(user);
    for (auto [constArg, input] :
         llvm::zip(stateConsts, callOp.getArgOperands())) {
      if (auto *constOp = input.getDefiningOp();
          constOp && constOp->template hasTrait<OpTrait::ConstantLike>()) {
        if (first) {
          constArg = constOp;
          continue;
        }
        if (constArg &&
            constArg->getName() == input.getDefiningOp()->getName() &&
            constArg->getAttrDictionary() ==
                input.getDefiningOp()->getAttrDictionary())
          continue;
      }
      constArg = nullptr;
    }
    first = false;
  }

  // Move the constants into the arc and erase the block arguments.
  rewriter.setInsertionPointToStart(&op.getBodyBlock());
  llvm::BitVector toDelete(op.getBodyBlock().getNumArguments());
  for (auto [constArg, arg] : llvm::zip(stateConsts, op.getArguments())) {
    if (!constArg)
      continue;
    auto *inlinedConst = rewriter.clone(*constArg);
    rewriter.replaceAllUsesWith(arg, inlinedConst->getResult(0));
    toDelete.set(arg.getArgNumber());
  }
  op.getBodyBlock().eraseArguments(toDelete);
  op.setType(rewriter.getFunctionType(op.getBodyBlock().getArgumentTypes(),
                                      op.getResultTypes()));

  // Rewrite all arc uses to not pass in the constant anymore.
  for (auto *user : users) {
    auto callOp = cast<CallOpMutableInterface>(user);
    SmallPtrSet<Value, 4> maybeUnusedValues;
    SmallVector<Value> newInputs;
    for (auto [index, value] : llvm::enumerate(callOp.getArgOperands())) {
      if (toDelete[index])
        maybeUnusedValues.insert(value);
      else
        newInputs.push_back(value);
    }
    rewriter.updateRootInPlace(
        callOp, [&]() { callOp.getArgOperandsMutable().assign(newInputs); });
    for (auto value : maybeUnusedValues)
      if (value.use_empty())
        rewriter.eraseOp(value.getDefiningOp());
  }

  return success(toDelete.any());
                                      }

LogicalResult
IndexingConstArray::matchAndRewrite(hw::ArrayGetOp op,
                                    PatternRewriter &rewriter) const {
  auto constArray = op.getInput().getDefiningOp<hw::AggregateConstantOp>();
  if (!constArray)
    return failure();

  Type elementType = op.getResult().getType();

  if (!elementType.isSignlessInteger())
    return failure();

  uint32_t elementBitWidth = elementType.getIntOrFloatBitWidth();
  uint32_t indexBitWidth = op.getIndex().getType().getIntOrFloatBitWidth();

  if (elementBitWidth < indexBitWidth)
    return failure();

  APInt one(elementBitWidth, 1);
  bool isIdentity = true, isShlOfOne = true;

  auto size = constArray.getFields().size();
  for (auto [i, fieldAttr] : llvm::enumerate(constArray.getFields())) {
    APInt elementValue = fieldAttr.cast<IntegerAttr>().getValue();

    if (elementValue != APInt(elementBitWidth, size-i-1))
      isIdentity = false;

    if (elementValue != one << APInt(elementBitWidth, size-i-1))
      isShlOfOne = false;
  }

  Value optionalZext = op.getIndex();
  if (isIdentity || isShlOfOne)
    optionalZext = zextUsingConcatOp(rewriter, op.getLoc(), op.getIndex(),
                                     elementBitWidth);

  if (isIdentity) {
    rewriter.replaceOp(op, optionalZext);
    return success();
  }

  if (isShlOfOne) {
    Value one =
        rewriter.create<hw::ConstantOp>(op.getLoc(), optionalZext.getType(), 1);
    rewriter.replaceOpWithNewOp<comb::ShlOp>(op, one, optionalZext);
    return success();
  }

  return failure();
}

/// Check to see if the condition to the specified mux is an equality
/// comparison `indexValue` and one or more constants.  If so, put the
/// constants in the constants vector and return true, otherwise return false.
///
/// This is part of foldMuxChain.
///
static bool
getMuxChainCondConstant(Value cond, Value extractValue, bool isInverted,
                        const std::function<void(uint32_t)> &constantFn) {
  if (auto cmp = cond.getDefiningOp<comb::ICmpOp>()) {
    // auto pred = isInverted ? comb::ICmpPredicate::ne : comb::ICmpPredicate::eq;
    if (auto constOp = cmp.getRhs().getDefiningOp<hw::ConstantOp>(); constOp && constOp.getValue().isZero()
    //  && cmp.getPredicate() == pred
     ) {
      if (auto extract = cmp.getLhs().getDefiningOp<comb::ExtractOp>())
        return getMuxChainCondConstant(extract, extractValue, true, constantFn);
    }
    return false;
  }

  if(!isInverted) {
    if (auto notOp = cond.getDefiningOp<comb::XorOp>(); notOp && notOp.isBinaryNot())
      return getMuxChainCondConstant(notOp.getOperands()[0], extractValue, !isInverted, constantFn);
    return false;
  }

  if (auto extract = cond.getDefiningOp<comb::ExtractOp>()) {
    if (extract.getInput() == extractValue) {
      auto width = cast<IntegerType>(extract.getResult().getType()).getWidth();
      for (unsigned idx = 0; idx < width; ++idx)
        constantFn(extract.getLowBit());
      return true;
    }
    return false;
  }
  // // Handle mux(`idx == 1 || idx == 3`, value, muxchain).
  // if (auto orOp = cond.getDefiningOp<OrOp>()) {
  //   if (!isInverted)
  //     return false;
  //   for (auto operand : orOp.getOperands())
  //     if (!getMuxChainCondConstant(operand, indexValue, isInverted, constantFn))
  //       return false;
  //   return true;
  // }

  // // Handle mux(`idx != 1 && idx != 3`, muxchain, value).
  // if (auto andOp = cond.getDefiningOp<AndOp>()) {
  //   if (isInverted)
  //     return false;
  //   for (auto operand : andOp.getOperands())
  //     if (!getMuxChainCondConstant(operand, indexValue, isInverted, constantFn))
  //       return false;
  //   return true;
  // }

  return false;
}

static bool foldMuxChain(comb::MuxOp rootMux, bool isFalseSide,
                         PatternRewriter &rewriter) {
  auto rootExtract = rootMux.getCond().getDefiningOp<comb::ExtractOp>();
  if (!rootExtract)
    return false;
  Value extractedValue = rootExtract.getInput();

  // Return the value to use if the equality match succeeds.
  auto getCaseValue = [&](comb::MuxOp mux) -> Value {
    return mux.getOperand(1 + unsigned(!isFalseSide));
  };

  // Return the value to use if the equality match fails.  This is the next
  // mux in the sequence or the "otherwise" value.
  auto getTreeValue = [&](comb::MuxOp mux) -> Value {
    return mux.getOperand(1 + unsigned(isFalseSide));
  };

  // Start scanning the mux tree to see what we've got.  Keep track of the
  // constant comparison value and the SSA value to use when equal to it.
  SmallVector<std::pair<uint32_t, Value>, 4> valuesFound;

  /// Extract constants and values into `valuesFound` and return true if this is
  /// part of the mux tree, otherwise return false.
  auto collectConstantValues = [&](comb::MuxOp mux) -> bool {
    return getMuxChainCondConstant(
        mux.getCond(), extractedValue, isFalseSide, [&](uint32_t cst) {
          valuesFound.push_back({cst, getCaseValue(mux)});
        });
  };

  // Make sure the root is a correct comparison with a constant.
  if (!collectConstantValues(rootMux))
    return false;

  // Make sure that we're not looking at the intermediate node in a mux tree.
  if (rootMux->hasOneUse()) {
    if (auto userMux = dyn_cast<comb::MuxOp>(*rootMux->user_begin())) {
      if (getTreeValue(userMux) == rootMux.getResult() &&
          getMuxChainCondConstant(userMux.getCond(), extractedValue, isFalseSide,
                                  [&](uint32_t cst) {}))
        return false;
    }
  }

  // Scan up the tree linearly.
  auto nextTreeValue = getTreeValue(rootMux);
  while (true) {
    auto nextMux = nextTreeValue.getDefiningOp<comb::MuxOp>();
    if (!nextMux || !nextMux->hasOneUse())
      break;
    if (!collectConstantValues(nextMux))
      break;
    nextTreeValue = getTreeValue(nextMux);
  }

  if (auto concat = nextTreeValue.getDefiningOp<comb::ConcatOp>()) {
    if (concat.getInputs().size() == 2) {
      if (auto constOp = concat.getInputs()[0].getDefiningOp<hw::ConstantOp>(); constOp && cast<IntegerType>(concat.getInputs()[1].getType()).getWidth() == 1) {
        if (getMuxChainCondConstant(concat.getInputs()[1], extractedValue, false, [&](uint32_t cst) {
              APInt concatenatedInput = constOp.getValue().concat(APInt(1, false));
              Value caseVal = rewriter.create<hw::ConstantOp>(rootMux.getLoc(), concatenatedInput);
              valuesFound.push_back({cst, caseVal});
            })) {
              APInt concatenatedInput = constOp.getValue().concat(APInt(1, true));
              Value caseVal = rewriter.create<hw::ConstantOp>(rootMux.getLoc(), concatenatedInput);
              nextTreeValue = caseVal;
            }
        else if (getMuxChainCondConstant(concat.getInputs()[1], extractedValue, true, [&](uint32_t cst) {
              APInt concatenatedInput = constOp.getValue().concat(APInt(1, true));
              Value caseVal = rewriter.create<hw::ConstantOp>(rootMux.getLoc(), concatenatedInput);
              valuesFound.push_back({cst, caseVal});
            })) {
              APInt concatenatedInput = constOp.getValue().concat(APInt(1, false));
              Value caseVal = rewriter.create<hw::ConstantOp>(rootMux.getLoc(), concatenatedInput);
              nextTreeValue = caseVal;
            }
      }
      else if (auto constOp = concat.getInputs()[1].getDefiningOp<hw::ConstantOp>(); constOp && cast<IntegerType>(concat.getInputs()[0].getType()).getWidth() == 1) {
        if (getMuxChainCondConstant(concat.getInputs()[0], extractedValue, false, [&](uint32_t cst) {
              APInt concatenatedInput = APInt(1, false).concat(constOp.getValue());
              Value caseVal = rewriter.create<hw::ConstantOp>(rootMux.getLoc(), concatenatedInput);
              valuesFound.push_back({cst, caseVal});
            })) {
              APInt concatenatedInput = APInt(1, true).concat(constOp.getValue());
              Value caseVal = rewriter.create<hw::ConstantOp>(rootMux.getLoc(), concatenatedInput);
              nextTreeValue = caseVal;
            }
        else if (getMuxChainCondConstant(concat.getInputs()[0], extractedValue, true, [&](uint32_t cst) {
              APInt concatenatedInput = APInt(1, true).concat(constOp.getValue());
              Value caseVal = rewriter.create<hw::ConstantOp>(rootMux.getLoc(), concatenatedInput);
              valuesFound.push_back({cst, caseVal});
            })) {
              APInt concatenatedInput = APInt(1, false).concat(constOp.getValue());
              Value caseVal = rewriter.create<hw::ConstantOp>(rootMux.getLoc(), concatenatedInput);
              nextTreeValue = caseVal;
            }
      }
    }
  }
  

  // We need to have more than three values to create an array.  This is an
  // arbitrary threshold which is saying that one or two muxes together is ok,
  // but three should be folded.
  if (valuesFound.size() < 3)
    return false;

  // If the array is greater that 9 bits, it will take over 512 elements and
  // it will be too large for a single expression.
  auto indexWidth = extractedValue.getType().cast<IntegerType>().getWidth();
  // if (indexWidth >= 9)
  //   return false;

  // Next we need to see if the values are dense-ish.  We don't want to have
  // a tremendous number of replicated entries in the array.  Some sparsity is
  // ok though, so we require the table to be at least 5/8 utilized.
  // uint64_t tableSize = 1ULL << indexWidth;
  // if (valuesFound.size() < (tableSize * 5) / 8)
  //   return false; // Not dense enough.

  // Ok, we're going to do the transformation, start by building the table
  // filled with the "otherwise" value.
  SmallVector<Value, 8> table(indexWidth, nextTreeValue);

  unsigned prevIdx = valuesFound[0].first;
  std::optional<bool> increasing;
  for (auto [extIdx, val] : valuesFound) {
    if (increasing.has_value()) {
      if (*increasing && extIdx < prevIdx)
        return false;
      if (!(*increasing) && extIdx > prevIdx) {
        // llvm::errs() << "Got " << extIdx << " and " << prevIdx << "\n";
        return false;
      }
      prevIdx = extIdx;
      continue;
    }
    if (prevIdx < extIdx)
      increasing = true;
    if (prevIdx > extIdx)
      increasing = false;
    prevIdx = extIdx;
  }

  // Fill in entries in the table from the leaf to the root of the expression.
  // This ensures that any duplicate matches end up with the ultimate value,
  // which is the one closer to the root.
  int pIdx = -1;
  int runner = 1;
  for (auto &elt : llvm::reverse(valuesFound)) {
    int64_t idx = elt.first;
    if (idx == pIdx) {
      if (*increasing)
        idx -= runner++;
      else
        idx += runner++;
    } else {
      pIdx = idx;
      runner = 1;
    }
    assert(idx < table.size() && "constant should be same bitwidth as index");
    table[idx] = elt.second;
  }

  // The hw.array_create operation has the operand list in unintuitive order
  // with a[0] stored as the last element, not the first.
  if (*increasing)
    std::reverse(table.begin(), table.end());

  // Build the array_create and the array_get.
  auto array = rewriter.create<hw::ArrayCreateOp>(rootMux.getLoc(), table);
  Value lcz = rewriter.create<arc::ZeroCountOp>(rootMux.getLoc(), extractedValue,
  *increasing ? ZeroCountPredicate::trailing : ZeroCountPredicate::leading);
  Value ext = rewriter.create<comb::ExtractOp>(rootMux.getLoc(), lcz, 0, std::max(llvm::Log2_64_Ceil(table.size()), 1U));
  rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(rootMux, array, ext);
  return true;
}

LogicalResult
ZeroCountRaising::matchAndRewrite(comb::MuxOp op,
                                  PatternRewriter &rewriter) const {
  if (foldMuxChain(op, true, rewriter))
    return success();
  if (foldMuxChain(op, false, rewriter))
    return success();
  return failure();
          
  // We don't want to match on muxes in the middle of a sequence.
  if (llvm::any_of(op.getResult().getUsers(),
                   [](auto user) { return isa<comb::MuxOp>(user); }))
    return failure();

  comb::MuxOp curr = op;
  uint32_t currIndex = -1;
  Value extractedFrom;
  SmallVector<Value> arrayElements;
  std::optional<bool> isLeading = std::nullopt;
  auto deltaFunc = [](uint32_t input, bool isLeading) {
    return isLeading ? --input : ++input;
  };

  while (true) {
    // Muxes not at the end of the sequence must not be used anywhere else as we
    // cannot remove them then.
    if (curr != op && !curr->hasOneUse())
      return failure();

    // We force the condition to be extracts for now as we otherwise have to
    // insert a concat which might be more expensive than what we gain.
    auto ext = curr.getCond().getDefiningOp<comb::ExtractOp>();
    if (!ext)
      return failure();

    if (ext.getResult().getType().getIntOrFloatBitWidth() != 1)
      return failure();

    if (currIndex == -1U)
      extractedFrom = ext.getInput();

    if (extractedFrom != ext.getInput())
      return failure();

    if (currIndex != -1U) {
      if (!isLeading.has_value()) {
        if (ext.getLowBit() == currIndex - 1)
          isLeading = true;
        else if (ext.getLowBit() == currIndex + 1)
          isLeading = false;
        else
          return failure();
      }

      if (ext.getLowBit() != deltaFunc(currIndex, *isLeading))
        return failure();
    }

    currIndex = ext.getLowBit();

    arrayElements.push_back(curr.getTrueValue());
    Value falseValue = curr.getFalseValue();

    curr = curr.getFalseValue().getDefiningOp<comb::MuxOp>();
    if (!curr) {
      // Check for init value patterns
      if (failed(handleSequenceInitializer(
              rewriter, op.getLoc(), deltaFunc, isLeading.value(), falseValue,
              extractedFrom, arrayElements, currIndex)))
        arrayElements.push_back(falseValue);

      break;
    }
  }

  if (arrayElements.size() < 4)
    return failure();

  Value extForLzc = rewriter.create<comb::ExtractOp>(
      op.getLoc(), extractedFrom,
      *isLeading ? currIndex : ((int)currIndex - (int)arrayElements.size() + 2),
      arrayElements.size() - 1);
  Value lcz = rewriter.create<arc::ZeroCountOp>(
      op.getLoc(), extForLzc,
      *isLeading ? ZeroCountPredicate::leading : ZeroCountPredicate::trailing);
  Value arrayIndex = rewriter.create<comb::ExtractOp>(
      op.getLoc(), lcz, 0, llvm::Log2_64_Ceil(arrayElements.size()));
  Value array = rewriter.create<hw::ArrayCreateOp>(op.getLoc(), arrayElements);
  rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(op, array, arrayIndex);

  return success();
}

LogicalResult ZeroCountRaising::handleSequenceInitializer(
    OpBuilder &rewriter, Location loc, const DeltaFunc &deltaFunc,
    bool isLeading, Value falseValue, Value extractedFrom,
    SmallVectorImpl<Value> &arrayElements, uint32_t &currIndex) const {
  if (auto concat = falseValue.getDefiningOp<comb::ConcatOp>()) {
    if (concat.getInputs().size() != 2) {
      arrayElements.push_back(falseValue);
      return failure();
    }
    Value nonConstant;
    if (auto constAllSet = concat.getOperand(0).getDefiningOp<hw::ConstantOp>())
      nonConstant = concat.getOperand(1);
    else if (auto constAllSet =
                 concat.getOperand(1).getDefiningOp<hw::ConstantOp>())
      nonConstant = concat.getOperand(0);
    else
      return failure();

    Value indirection = nonConstant;
    bool negated = false;
    if (auto xorOp = nonConstant.getDefiningOp<comb::XorOp>();
        xorOp && xorOp.isBinaryNot()) {
      indirection = xorOp.getOperand(0);
      negated = true;
    }

    auto ext = indirection.getDefiningOp<comb::ExtractOp>();
    if (ext.getInput() == extractedFrom &&
        ext.getResult().getType().getIntOrFloatBitWidth() == 1 &&
        ext.getLowBit() == deltaFunc(currIndex, isLeading)) {
      currIndex = ext.getLowBit();
      Value zero =
          rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), 0);
      Value one =
          rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), -1);
      // Value stoppers = rewriter.create<comb::OrOp>(op.getLoc(),
      // extractedFrom, c);
      IRMapping mapping;
      mapping.map(nonConstant, zero);
      auto *clonedZeroConcat = rewriter.clone(*concat, mapping);
      mapping.map(nonConstant, one);
      auto *clonedOneConcat = rewriter.clone(*concat, mapping);

      if (negated)
        arrayElements.push_back(clonedZeroConcat->getResult(0));
      arrayElements.push_back(clonedOneConcat->getResult(0));
      if (!negated)
        arrayElements.push_back(clonedZeroConcat->getResult(0));
      return success();
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ArcCanonicalizerPass implementation
//===----------------------------------------------------------------------===//

namespace {
struct ArcCanonicalizerPass
    : public ArcCanonicalizerBase<ArcCanonicalizerPass> {
  void runOnOperation() override;
};
} // namespace

void ArcCanonicalizerPass::runOnOperation() {
  MLIRContext &ctxt = getContext();
  SymbolTableCollection symbolTable;
  SymbolHandler cache;
  cache.addDefinitions(getOperation());
  cache.collectAllSymbolUses(getOperation(), symbolTable);
  Namespace names;
  names.add(cache);
  DenseMap<StringAttr, StringAttr> arcMapping;

  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = false;
  config.maxIterations = 10;
  config.useTopDownTraversal = true;

  PatternStatistics statistics;
  RewritePatternSet symbolPatterns(&getContext());
  symbolPatterns.add<CallPassthroughArc, StatePassthroughArc, RemoveUnusedArcs,
                     RemoveUnusedArcArgumentsPattern, SinkArcInputsPattern>(
      &getContext(), cache, names, statistics);
  symbolPatterns.add<MemWritePortEnableAndMaskCanonicalizer>(
      &getContext(), cache, names, statistics, arcMapping);

  if (failed(mlir::applyPatternsAndFoldGreedily(
          getOperation(), std::move(symbolPatterns), config)))
    return signalPassFailure();

  numArcArgsRemoved = statistics.removeUnusedArcArgumentsPatternNumArgsRemoved;

  RewritePatternSet patterns(&ctxt);
  for (auto *dialect : ctxt.getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (mlir::RegisteredOperationName op : ctxt.getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, &ctxt);
  patterns.add<ICMPCanonicalizer, ZeroCountRaising, IndexingConstArray>(
      &getContext());

  // Don't test for convergence since it is often not reached.
  (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                           config);
}

std::unique_ptr<mlir::Pass> arc::createArcCanonicalizerPass() {
  return std::make_unique<ArcCanonicalizerPass>();
}
