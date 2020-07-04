//===- DesequentializationPass.cpp - Implement Desequentialization Pass ---===//
//
// Implement pass to lower sequential processes to entities.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/LLHD/Analysis/TemporalRegions.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::llhd;

namespace {

struct DesequentializationPass
    : public llhd::DesequentializationBase<DesequentializationPass> {
  void runOnOperation() override;
};

/// Represents the type of a node in the DNF tree
enum class DnfNodeType {
  Const,
  Val,
  And,
  Or,
};

/// A tree representing a DNF formula
struct Dnf {
  Dnf() = delete;
  /// Create a copy of a DNF tree
  Dnf(const Dnf &e) : type(e.type), value(e.value), inv(e.inv) {
    for (auto &child : e.children) {
      children.push_back(std::make_unique<Dnf>(*child));
    }
  }
  /// Create a new Constant leaf node
  Dnf(bool constant, bool inv)
      : type(DnfNodeType::Const), constant(constant), inv(inv) {}
  /// Create a new value leaf node, this should either be a sample of a signal
  /// or an opaque value
  Dnf(Value val, bool inv) : type(DnfNodeType::Val), value(val), inv(inv) {}
  /// Create a new Dnf tree from two children connected bei either an AND or OR,
  /// given that the two children are both a DNF tree representing a valid DNF
  /// formula, the newly created DNF tree will also be a valid DNF formula
  Dnf(DnfNodeType ty, std::unique_ptr<Dnf> lhs, std::unique_ptr<Dnf> rhs)
      : type(ty) {
    assert(lhs && rhs && "Passed expressions should not be a nullptr!");
    switch (ty) {
    case DnfNodeType::And: {
      if (lhs->isOr() && rhs->isOr()) {
        // (A v B) ^ (C v D) => (A ^ C) v (B ^ C) v (A ^ D) v (B ^ D)
        type = DnfNodeType::Or;
        for (auto &&lhsChild : lhs->children) {
          for (auto &&rhsChild : rhs->children) {
            children.push_back(std::make_unique<Dnf>(
                DnfNodeType::And, std::make_unique<Dnf>(*lhsChild),
                std::make_unique<Dnf>(*rhsChild)));
          }
        }
      } else if (lhs->isOr() && rhs->isAnd()) {
        // (A v B) ^ (C ^ D) => (A ^ C ^ D) v (B ^ C ^ D)
        type = DnfNodeType::Or;
        for (auto &&lhsChild : lhs->children) {
          children.push_back(
              std::make_unique<Dnf>(DnfNodeType::And, std::move(lhsChild),
                                    std::make_unique<Dnf>(*rhs)));
        }
      } else if (lhs->isAnd() && rhs->isOr()) {
        // (A ^ B) ^ (C v D) => (A ^ B ^ C) v (A ^ B ^ D)
        type = DnfNodeType::Or;
        for (auto &&rhsChild : rhs->children) {
          children.push_back(std::make_unique<Dnf>(DnfNodeType::And,
                                                   std::make_unique<Dnf>(*lhs),
                                                   std::move(rhsChild)));
        }
      } else if (lhs->isAnd() && rhs->isAnd()) {
        // (A ^ B) ^ (C ^ D) => A ^ B ^ C ^ D
        std::move(begin(lhs->children), end(lhs->children),
                  std::back_inserter(children));
        std::move(begin(rhs->children), end(rhs->children),
                  std::back_inserter(children));
      } else if (lhs->isLeaf() && rhs->isAnd()) {
        // A ^ (B ^ C) => A ^ B ^ C
        children.push_back(std::move(lhs));
        std::move(begin(rhs->children), end(rhs->children),
                  std::back_inserter(children));
      } else if (lhs->isAnd() && rhs->isLeaf()) {
        // (A ^ B) ^ C => A ^ B ^ C
        std::move(begin(lhs->children), end(lhs->children),
                  std::back_inserter(children));
        children.push_back(std::move(rhs));
      } else if (lhs->isLeaf() && rhs->isLeaf()) {
        // A ^ B
        children.push_back(std::move(lhs));
        children.push_back(std::move(rhs));
      } else if (lhs->isOr() && rhs->isLeaf()) {
        // (A v B) ^ C => (A ^ C) v (B ^ C)
        type = DnfNodeType::Or;
        // std::move(begin(lhs->children), end(lhs->children),
        // std::back_inserter(children));
        for (auto &&lhsChild : lhs->children) {
          children.push_back(
              std::make_unique<Dnf>(DnfNodeType::And, std::move(lhsChild),
                                    std::make_unique<Dnf>(*rhs)));
        }
      } else if (lhs->isLeaf() && rhs->isOr()) {
        // A v (B ^ C) => (A ^ B) v (A ^ C)
        type = DnfNodeType::Or;
        // std::move(begin(rhs->children), end(rhs->children),
        // std::back_inserter(children));
        for (auto &&rhsChild : rhs->children) {
          children.push_back(std::make_unique<Dnf>(DnfNodeType::And,
                                                   std::make_unique<Dnf>(*lhs),
                                                   std::move(rhsChild)));
        }
      } else {
        assert(false && "Unreachable!");
      }
      break;
    }
    case DnfNodeType::Or: {
      if ((lhs->isAnd() && rhs->isAnd()) || (lhs->isLeaf() && rhs->isLeaf()) ||
          (lhs->isLeaf() && rhs->isAnd()) || (lhs->isAnd() && rhs->isLeaf())) {
        // (A ^ B) v (C ^ D)
        // or
        // A v B
        children.push_back(std::move(lhs));
        children.push_back(std::move(rhs));
      } else if (lhs->isOr() && (rhs->isAnd() || rhs->isLeaf())) {
        // (A v B) v (C ^ D) => A v B v (C ^ D)
        // or
        // (A v B) v C => A v B v C
        std::move(begin(lhs->children), end(lhs->children),
                  std::back_inserter(children));
        children.push_back(std::move(rhs));
      } else if ((lhs->isAnd() || lhs->isLeaf()) && rhs->isOr()) {
        // (A ^ B) v (C v D) => (A ^ B) v C v D
        // or
        // A v (B v C) => A v B v C
        children.push_back(std::move(lhs));
        std::move(begin(rhs->children), end(rhs->children),
                  std::back_inserter(children));
      } else if (lhs->isOr() && rhs->isOr()) {
        // (A v B) v (C v D) => A v B v C v D
        std::move(begin(rhs->children), end(rhs->children),
                  std::back_inserter(children));
        std::move(begin(lhs->children), end(lhs->children),
                  std::back_inserter(children));
      } else {
        assert(false && "Unreachable!");
      }
      break;
    }
    default: {
      assert(false &&
             "To create a Const or Val node, use the other constructors!");
      break;
    }
    }
    canonicalize();
  }

  bool isConst() { return type == DnfNodeType::Const; }
  bool isVal() { return type == DnfNodeType::Val; }
  bool isLeaf() { return isConst() || isVal(); }
  bool isAnd() { return type == DnfNodeType::And; }
  bool isOr() { return type == DnfNodeType::Or; }
  bool isNegatedVal() { return isVal() && inv; }
  bool isProbedSignal() {
    return isVal() && isa<llhd::PrbOp>(value.getDefiningOp());
  }

  bool getConst() {
    assert(isConst() && "node has to be Const to return the constant");
    return constant ^ inv;
  }
  Value getProbedSignal() {
    assert(isProbedSignal() && "Can only return probed signal if the value "
                               "actually got probed from a signal!");
    return cast<llhd::PrbOp>(value.getDefiningOp()).signal();
  }

  DnfNodeType type;
  std::vector<std::unique_ptr<Dnf>> children;
  Value value;
  bool constant;
  bool inv;

private:
  void canonicalize() {
    for (auto it1 = children.begin(); it1 != children.end(); ++it1) {
      auto &c1 = *it1;
      if (!c1)
        continue;

      if (c1->isConst()) {
        if (c1->getConst() == true) {
          if (isAnd()) {
            // Remove a constant TRUE in an AND node, because it doesn't change
            // anything about the formula, except if it is the only child
            if (children.size() > 1)
              children.erase(it1--);
          } else if (isOr()) {
            // If there is a constant TRUE in this OR node, remove all other
            // nodes, and replace this node with a constant TRUE, because it
            // will always evaluate to TRUE
            children.clear();
            type = DnfNodeType::Const;
            constant = true;
            inv = false;
            return;
          }
        } else {
          if (isAnd()) {
            // If there is a constant FALSE in this AND node, remove all
            // children and become a constant FALSE node
            children.clear();
            type = DnfNodeType::Const;
            constant = false;
            inv = false;
            return;
          } else if (isOr()) {
            // If there is a constant FALSE in an OR node, remove it because the
            // other nodes alone determine what the OR evaluates to, except if
            // it is the only child
            if (children.size() > 1)
              children.erase(it1--);
          }
        }
      }

      for (auto it = std::next(it1); it != children.end(); ++it) {
        auto &c2 = *it;
        if (c1->isVal() && c2->isVal()) {
          if ((c1->value == c2->value)) {
            if (c1->inv == c2->inv) {
              // Remove duplicate Val children, if this node will end up with
              // only one child, it will become this child at the end of this
              // method
              children.erase(it--);
            } else {
              if (isAnd()) {
                // If this node is an AND node and it has two children which are
                // the opposite (one is the negation of the other), delete all
                // children of this node and replace this node with a constant
                // FALSE because this node will always be FALSE
                children.clear();
                type = DnfNodeType::Const;
                constant = false;
                inv = false;
                return;
              } else if (isOr()) {
                // If this node is an OR node and it has two children which are
                // the opposite (one is the negation of the other), delete all
                // children of this node and replace this node with a constant
                // TRUE because this node will always be TRUE
                children.clear();
                type = DnfNodeType::Const;
                constant = true;
                inv = false;
                return;
              }
            }
          }
        }
      }
    }

    // Erase AND and OR children which do not have children anymore
    for (auto it = children.begin(); it != children.end(); ++it) {
      auto &child = *it;
      if (child->isAnd() || child->isOr()) {
        if (child->children.empty()) {
          children.erase(it--);
        }
      }
    }

    // If you have one child only, become the child
    // if (children.size() == 1) {
    //   auto &child = *children.begin();
    //   type = child->type;
    //   inv = child->inv;
    //   value = child->value;
    //   std::vector<std::unique_ptr<Dnf>> c = std::move(child->children);
    //   children.clear();
    //   children = std::move(c);
    // }
  }
};

/// Represents the data necessary to construct a llhd::RegOp and provides helper
/// methods to easily add triggers one by one
struct RegData {
  void addTrigger(OpBuilder &builder, Value trigger, RegMode mode,
                  Value delay) {
    triggers.push_back(trigger);
    modes.push_back(builder.getI64IntegerAttr((int64_t)mode));
    delays.push_back(delay);
    gateMask.push_back(builder.getI64IntegerAttr(0));
  }

  void addTriggerWithGate(OpBuilder &builder, Value trigger, RegMode mode,
                          Value delay, Value gate) {
    if (!gate) {
      addTrigger(builder, trigger, mode, delay);
      return;
    }
    triggers.push_back(trigger);
    modes.push_back(builder.getI64IntegerAttr((int64_t)mode));
    delays.push_back(delay);
    gates.push_back(gate);
    gateMask.push_back(builder.getI64IntegerAttr(++lastGate));
  }

  SmallVector<Value, 4> triggers;
  SmallVector<Attribute, 4> modes;
  SmallVector<Value, 4> delays;
  SmallVector<Value, 4> gates;
  SmallVector<Attribute, 4> gateMask;

private:
  int64_t lastGate = 0;
};

/// Takes a SSA-value and a boolean which specifies whether this value is
/// inverted and builds a DNF tree from the expression this ssa-value represents
/// by recursively following the defining operations
/// NOTE: only llhd.not, llhd.and, llhd.or, llhd.xor, cmpi "ne", cmpi "eq",
/// constant, llhd.const and llhd.prb are supported directly, values defined by
/// other operations are treated as opaque values
std::unique_ptr<Dnf> buildDnf(Value value, bool inv) {
  if (!value.getType().isSignlessInteger(1))
    emitError(value.getLoc(), "Only one-bit signless integers supported!");

  Operation *defOp = value.getDefiningOp();
  if (isa<ConstantOp>(defOp) || isa<llhd::ConstOp>(defOp)) {
    bool valueAttr = defOp->getAttr("value").cast<BoolAttr>().getValue();
    return std::make_unique<Dnf>(valueAttr, inv);
  } else if (isa<llhd::PrbOp>(defOp)) {
    return std::make_unique<Dnf>(value, inv);
  } else if (llhd::NotOp op = dyn_cast<llhd::NotOp>(defOp)) {
    return buildDnf(op.value(), !inv);
  } else if (llhd::AndOp op = dyn_cast<llhd::AndOp>(defOp)) {
    std::unique_ptr<Dnf> lhs = buildDnf(op.lhs(), inv);
    std::unique_ptr<Dnf> rhs = buildDnf(op.rhs(), inv);
    DnfNodeType type = inv ? DnfNodeType::Or : DnfNodeType::And;
    return std::make_unique<Dnf>(type, std::move(lhs), std::move(rhs));
  } else if (llhd::OrOp op = dyn_cast<llhd::OrOp>(defOp)) {
    std::unique_ptr<Dnf> lhs = buildDnf(op.lhs(), inv);
    std::unique_ptr<Dnf> rhs = buildDnf(op.rhs(), inv);
    DnfNodeType type = inv ? DnfNodeType::And : DnfNodeType::Or;
    return std::make_unique<Dnf>(type, std::move(lhs), std::move(rhs));
  } else if (isa<llhd::XorOp>(defOp) || isa<CmpIOp>(defOp)) {
    auto xorOp = dyn_cast<llhd::XorOp>(defOp);
    auto cmpiOp = dyn_cast<CmpIOp>(defOp);
    if (cmpiOp) {
      if (!cmpiOp.lhs().getType().isSignlessInteger(1)) {
        // Return opaque value
        return std::make_unique<Dnf>(value, inv);
      }
      if (cmpiOp.predicate() != CmpIPredicate::eq &&
          cmpiOp.predicate() != CmpIPredicate::ne) {
        // Return opaque value
        return std::make_unique<Dnf>(value, inv);
      }
      inv = cmpiOp.predicate() == CmpIPredicate::eq ? !inv : inv;
    }
    Value rhs = xorOp ? xorOp.rhs() : cmpiOp.rhs();
    Value lhs = xorOp ? xorOp.lhs() : cmpiOp.lhs();
    std::unique_ptr<Dnf> lhs_pos = buildDnf(lhs, true);
    std::unique_ptr<Dnf> rhs_pos = buildDnf(rhs, true);
    std::unique_ptr<Dnf> lhs_neg = buildDnf(lhs, false);
    std::unique_ptr<Dnf> rhs_neg = buildDnf(rhs, false);
    if (inv)
      return std::make_unique<Dnf>(
          DnfNodeType::Or,
          std::make_unique<Dnf>(DnfNodeType::And, std::move(lhs_pos),
                                std::move(rhs_pos)),
          std::make_unique<Dnf>(DnfNodeType::And, std::move(lhs_neg),
                                std::move(rhs_neg)));
    else
      return std::make_unique<Dnf>(
          DnfNodeType::Or,
          std::make_unique<Dnf>(DnfNodeType::And, std::move(lhs_pos),
                                std::move(rhs_neg)),
          std::make_unique<Dnf>(DnfNodeType::And, std::move(lhs_neg),
                                std::move(rhs_pos)));
  }
  // Return opaque value
  return std::make_unique<Dnf>(value, inv);
}

Value getInvertedValueIfNeeded(OpBuilder builder, std::unique_ptr<Dnf> &expr) {
  assert(expr->isVal() && "Only Value expressions supported!");
  if (expr->isNegatedVal()) {
    return builder.create<llhd::NotOp>(expr->value.getLoc(), expr->value);
  } else {
    return expr->value;
  }
}

void dnfToTriggers(OpBuilder &builder, TemporalRegionAnalysis &trAnalysis,
                   RegData &regData, DrvOp op, int pastTR, int presentTR,
                   std::unique_ptr<Dnf> &expr) {
  if (expr->isConst()) {
    Value trigger = builder.create<ConstantOp>(
        op.getLoc(), builder.getBoolAttr(expr->getConst()));
    regData.addTrigger(builder, trigger, RegMode::high, op.time());
  } else if (expr->isVal()) {
    if (expr->isProbedSignal()) {
      // TODO: this semantics is probably incorrect, shouldn't the value of the
      // signal itself be used?
      regData.addTrigger(builder, getInvertedValueIfNeeded(builder, expr),
                         RegMode::high, op.time());
    } else {
      assert(false && "invalid semantics!");
    }
  } else if (expr->isAnd()) {
    bool triggerAdded = false;
    for (std::unique_ptr<Dnf> &conjTerm : expr->children) {
      assert(conjTerm->isLeaf() &&
             "Only constant or value expressions allowed here!");
      if (!conjTerm->isProbedSignal())
        continue;

      auto iter = std::find_if(expr->children.begin(), expr->children.end(),
                               [&](std::unique_ptr<Dnf> &expr) {
                                 return expr->isProbedSignal() &&
                                        conjTerm->getProbedSignal() ==
                                            expr->getProbedSignal() &&
                                        conjTerm->inv != expr->inv;
                               });
      if (iter != expr->children.end()) {
        std::unique_ptr<Dnf> &sample = *iter;
        Value presentSample;
        bool presentInv;
        Value pastSample;
        bool pastInv;
        if (trAnalysis.getBlockTR(sample->value.getParentBlock()) == pastTR &&
            trAnalysis.getBlockTR(conjTerm->value.getParentBlock()) ==
                presentTR) {
          presentSample = conjTerm->value;
          presentInv = conjTerm->inv;
          pastSample = sample->value;
          pastInv = sample->inv;
        } else if (trAnalysis.getBlockTR(sample->value.getParentBlock()) ==
                       presentTR &&
                   trAnalysis.getBlockTR(conjTerm->value.getParentBlock()) ==
                       pastTR) {
          presentSample = sample->value;
          presentInv = sample->inv;
          pastSample = conjTerm->value;
          pastInv = conjTerm->inv;
        } else {
          assert(false && "What to do if this occurrs?");
        }
        Value res = Value();
        // builder.setInsertionPoint(op);
        for (std::unique_ptr<Dnf> &term : expr->children) {
          if (term->value == presentSample || term->value == pastSample)
            continue;
          if (!res) {
            res = getInvertedValueIfNeeded(builder, term);
          } else {
            res = builder.create<llhd::AndOp>(
                res.getLoc(), res, getInvertedValueIfNeeded(builder, term));
          }
        }
        if (pastInv && !presentInv) {
          // Rising Edge
          regData.addTriggerWithGate(builder, presentSample, RegMode::rise,
                                     op.time(), res);
          triggerAdded = true;
        } else { // pastInv && !presentInv
                 // Falling edge
          regData.addTriggerWithGate(builder, presentSample, RegMode::fall,
                                     op.time(), res);
          triggerAdded = true;
        }
        break;
      }
    }
    if (!triggerAdded) {
      Value res = Value();
      // builder.setInsertionPoint(op);
      for (std::unique_ptr<Dnf> &term : expr->children) {
        Value toAdd = Value();
        if (term->isConst()) {
          toAdd = builder.create<ConstantOp>(
              op.getLoc(), builder.getBoolAttr(term->getConst()));
        } else { // term->isVal()
          toAdd = getInvertedValueIfNeeded(builder, term);
        }
        if (!res) {
          res = toAdd;
        } else {
          res = builder.create<llhd::AndOp>(res.getLoc(), res, toAdd);
        }
      }
      regData.addTrigger(builder, res, RegMode::high, op.time());
    }
  } else if (expr->isOr()) {
    for (std::unique_ptr<Dnf> &disjTerm : expr->children) {
      dnfToTriggers(builder, trAnalysis, regData, op, pastTR, presentTR,
                    disjTerm);
    }
  }
}

void DesequentializationPass::runOnOperation() {
  ModuleOp module = getOperation();
  module.walk([&](ProcOp proc) {
    TemporalRegionAnalysis trAnalysis(proc);
    unsigned numTRs = trAnalysis.getNumTemporalRegions();

    // We only consider the case with three basic blocks and two TRs, because
    // combinatorial circuits have fewer blocks and don't need
    // desequentialization and more are not supported for now
    // NOTE: 3 basic blocks because of the entry block and one for each TR
    if (numTRs != 2 || proc.getBlocks().size() != 3)
      return;

    int pastTR;

    bool seenWait = false;
    WalkResult result = proc.walk([&](WaitOp op) -> WalkResult {
      pastTR = trAnalysis.getBlockTR(op.getOperation()->getBlock());
      if (seenWait) {
        return op.emitError("Only one wait operation per process supported!");
      }
      // Check that the block containing the wait is the only exiting block of
      // that TR
      if (!trAnalysis.hasSingleExitBlock(
              trAnalysis.getBlockTR(op.getOperation()->getBlock()))) {
        return op.emitError(
            "Block with wait terminator has to be the only exiting block "
            "of that temporal region!");
      }
      seenWait = true;
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    if (!seenWait) {
      proc.emitError("Block with wait terminator has to be present for "
                     "desequentialization to be applicable!");
      signalPassFailure();
      return;
    }

    OpBuilder builder(proc);
    proc.walk([&](DrvOp op) {
      if (!op.enable())
        return;

      builder.setInsertionPoint(op);
      int presentTR = trAnalysis.getBlockTR(op.getOperation()->getBlock());

      // Transform the enable condition of the drive into DNF
      std::unique_ptr<Dnf> dnf = buildDnf(op.enable(), false);

      // Translate the DNF to a list of triggers for the reg instruction
      RegData regData;
      dnfToTriggers(builder, trAnalysis, regData, op, pastTR, presentTR, dnf);

      // Replace the drive operation with a reg operation
      auto modeAttr = builder.getArrayAttr(ArrayRef<Attribute>(regData.modes));
      auto valueArr = std::vector<Value>(regData.modes.size(), op.value());
      auto gateMaskAttr =
          builder.getArrayAttr(ArrayRef<Attribute>(regData.gateMask));
      builder.create<llhd::RegOp>(op.getLoc(), op.signal(), modeAttr, valueArr,
                                  regData.triggers, regData.delays,
                                  regData.gates, gateMaskAttr);
      op.getOperation()->dropAllReferences();
      op.getOperation()->erase();
    });

    builder.setInsertionPoint(proc);
    // Create a new entity with the same name and type as the process it's
    // replacing
    EntityOp entity = builder.create<llhd::EntityOp>(proc.getLoc(), proc.ins());
    entity.setName(proc.getName());
    entity.setAttr("type", proc.getAttr("type"));
    // Take the whole region from the process
    entity.body().takeBody(proc.body());
    // Delete the terminator of the entry block
    Operation *terminator = entity.body().front().getTerminator();
    terminator->dropAllReferences();
    terminator->erase();
    // Move all instructions to the entry block except the terminators
    Block &first = entity.body().front();
    for (Block &block : entity.body().getBlocks()) {
      if (block.isEntryBlock())
        continue;
      block.getTerminator()->dropAllReferences();
      block.getTerminator()->erase();
      first.getOperations().splice(first.end(), block.getOperations());
      block.dropAllDefinedValueUses();
      block.dropAllReferences();
    }
    // Delete all blocks except the entry block
    entity.getBlocks().erase(std::next(entity.getBlocks().begin()),
                             entity.getBlocks().end());
    // Add the implicit entity terminator at the end of the entry block
    builder.setInsertionPointToEnd(&entity.body().front());
    builder.create<llhd::TerminatorOp>(entity.getLoc());

    // TODO
    // proc.replaceAllSymbolUses(entity.getName(), module.getOperation());

    // Delete the process
    proc.getOperation()->dropAllDefinedValueUses();
    proc.getOperation()->dropAllReferences();
    proc.getOperation()->erase();
  });
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::llhd::createDesequentializationPass() {
  return std::make_unique<DesequentializationPass>();
}
