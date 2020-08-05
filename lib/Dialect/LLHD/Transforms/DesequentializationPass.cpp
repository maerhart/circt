//===- DesequentializationPass.cpp - Implement Desequentialization Pass ---===//
//
// Implement pass to lower sequential processes to entities.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "DNFUtil.h"
#include "TemporalRegions.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::llhd;

namespace {
struct DesequentializationPass
    : public llhd::DesequentializationBase<DesequentializationPass> {
  void runOnOperation() override;
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
} // namespace

/// Takes a SSA-value and a boolean which specifies whether this value is
/// inverted and builds a DNF tree from the expression this ssa-value represents
/// by recursively following the defining operations
/// NOTE: only llhd.not, llhd.and, llhd.or, llhd.xor, cmpi "ne", cmpi "eq",
/// constant, llhd.const and llhd.prb are supported directly, values defined by
/// other operations are treated as opaque values
static std::unique_ptr<Dnf> buildDnf(Value value, bool inv) {
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

static Value getInvertedValueIfNeeded(OpBuilder builder, std::unique_ptr<Dnf> &expr) {
  assert(expr->isVal() && "Only Value expressions supported!");
  if (expr->isNegatedVal()) {
    return builder.create<llhd::NotOp>(expr->value.getLoc(), expr->value);
  } else {
    return expr->value;
  }
}

static void dnfToTriggers(OpBuilder &builder, TemporalRegionAnalysis &trAnalysis,
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

std::unique_ptr<OperationPass<ModuleOp>>
mlir::llhd::createDesequentializationPass() {
  return std::make_unique<DesequentializationPass>();
}
