//===- DesequentializationPass.cpp - Implement Desequentialization Pass ---===//
//
// Implement pass to lower sequential processes to entities.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "TemporalRegions.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm>
#include <cstdio>
#include <functional>
#include <unordered_set>

using namespace mlir;
using namespace circt;
using namespace circt::llhd;

namespace {
struct DesequentializationPass
    : public llhd::DesequentializationPassBase<DesequentializationPass> {
  void runOnOperation() override;
};

/// Represents the data necessary to construct a llhd::RegOp and provides helper
/// methods to easily add triggers one by one
struct RegData {
  void addTrigger(OpBuilder &builder, Value trigger, RegMode mode,
                          Value delay, Value gate = nullptr) {
    // If this exact trigger already exists, then do not add it again
    // If this trigger already exists with another gate then set the gate of
    // this existing trigger to the disjunction of the existing and the new gate
    // for (auto &&items : zip(triggers, modes, delays, gateMask)) {
    //   if (std::get<0>(items) == trigger &&
    //       std::get<1>(items).cast<IntegerAttr>().getInt() == (int64_t)mode &&
    //       std::get<2>(items) == delay) {
    //         if (std::get<3>(items).cast<IntegerAttr>().getInt() == 0 && !gate) {
    //           return;
    //         }
    //         if (std::get<3>(items).cast<IntegerAttr>().getInt() == 0 && gate) {
    //           // TODO
    //         }
    //         else if (std::get<3>(items).cast<IntegerAttr>().getInt() != 0 && gate) {
    //           Value newGate = builder.create<comb::OrOp>(trigger.getLoc(), gate, gates[std::get<3>(items).cast<IntegerAttr>().getInt()-1]);
    //           gates[std::get<3>(items).cast<IntegerAttr>().getInt() - 1] = newGate;
    //           return;
    //         }
    //       }
    // }

    triggers.push_back(trigger);
    modes.push_back(builder.getI64IntegerAttr((int64_t)mode));
    delays.push_back(delay);
    if (!gate) {
      gateMask.push_back(builder.getI64IntegerAttr(0));
      return;
    }
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

// False is modeled as {} and true is modeled as {{}}
using DNFNode = std::pair<std::pair<Value, int>, bool>;
using DNFConjunct = DenseMap<std::pair<Value, int>, bool>;
using DNF = std::vector<DNFConjunct>;

static void combineAndInner(DNF &lhs, DNF &rhs, DNF &res) {
  for (DNFConjunct lhsConjunct : lhs) {
    for (DNFConjunct rhsConjunct : rhs) {
      DNFConjunct newConj;
      for (DNFNode node : lhsConjunct) {
        newConj.insert(node);
      }
      for (DNFNode node : rhsConjunct) {
        auto present = newConj.find(node.first);
        if (present != newConj.end() && present->second != node.second && present->first.second == node.first.second) {
          newConj.clear();
          break;
        }
        newConj.insert(node);
      }
      if (!newConj.empty()) res.push_back(newConj);
    }
  }
  // res.erase(std::unique(res.begin(), res.end()), res.end());
}

static DNF combineAnd(std::vector<DNF> &conjuncts) {
  assert(conjuncts.size() >= 2);
  DNF res;
  DNF lhs = conjuncts[0];
  for (auto iter = std::next(conjuncts.begin()); iter != conjuncts.end(); iter++) {
    combineAndInner(lhs, *iter, res);
    lhs = res;
    res.clear();
  }
  return lhs;
}

static DNF combineOr(std::vector<DNF> &disjuncts) {
  DNF res;
  for (DNF dnf : disjuncts) {
    for (DNFConjunct conj : dnf) {
      res.push_back(DNFConjunct(conj));
    }
  }
  // res.erase(std::unique(res.begin(), res.end()), res.end());
  return res;
}

static DNF buildDnf(Value value, bool inv, TemporalRegionAnalysis &trAnalysis);

static DNF mapCombine(Operation::operand_range range, std::function<DNF(std::vector<DNF>&)> combiner, bool inv, TemporalRegionAnalysis &trAnalysis) {
  std::vector<DNF> dnf;
  for (Value input : range) {
    dnf.push_back(buildDnf(input, inv, trAnalysis));
  }
  return combiner(dnf);
}

static DNF combineXorHelper(DNF &lhsPos, DNF &rhsPos, DNF &lhsNeg, DNF &rhsNeg, bool inv) {
  std::vector<DNF> dnf(2);
  if (inv) {
    combineAndInner(lhsPos, rhsPos, dnf[0]);
    combineAndInner(lhsNeg, rhsNeg, dnf[1]);
    return combineOr(dnf);
  }

  combineAndInner(lhsPos, rhsNeg, dnf[0]);
  combineAndInner(lhsNeg, rhsPos, dnf[1]);
  return combineOr(dnf);
}

static DNF combineXor(ValueRange range, bool inv, TemporalRegionAnalysis &trAnalysis) {
  std::vector<DNF> dnfPos;
  std::vector<DNF> dnfNeg;
  for (Value input : range) {
    dnfPos.push_back(buildDnf(input, true, trAnalysis));
    dnfNeg.push_back(buildDnf(input, false, trAnalysis));
  }

  DNF res;
  DNF lhsNeg = combineXorHelper(dnfPos[0], dnfPos[1], dnfNeg[0], dnfNeg[1], true);
  DNF lhsPos = combineXorHelper(dnfPos[0], dnfPos[1], dnfNeg[0], dnfNeg[1], false);
  for (int i = 2, e = (int) dnfPos.size(); i < e; i++) {
    lhsNeg = combineXorHelper(lhsPos, dnfPos[i], lhsNeg, dnfNeg[i], true);
    lhsPos = combineXorHelper(lhsPos, dnfPos[i], lhsNeg, dnfNeg[i], false);
  }

  return inv ? lhsNeg : lhsPos;
}

static DNF singleValueDNF(Value value, bool inv, int tr) {
  DNF res;
  DNFConjunct conj;
  conj.insert({{value, tr}, inv});
  res.push_back(conj);
  return res;
}

/// Takes a SSA-value and a boolean which specifies whether this value is
/// inverted and builds a DNF tree from the expression this ssa-value represents
/// by recursively following the defining operations
/// NOTE: only llhd.not, llhd.and, llhd.or, llhd.xor, cmpi "ne", cmpi "eq",
/// constant, llhd.const and llhd.prb are supported directly, values defined by
/// other operations are treated as opaque values
static DNF buildDnf(Value value, bool inv, TemporalRegionAnalysis &trAnalysis) {
  if (!value.getType().isSignlessInteger(1))
    emitError(value.getLoc(), "Only one-bit signless integers supported!");

  return TypeSwitch<Operation*, DNF>(value.getDefiningOp())
    .Case<hw::ConstantOp>([&](hw::ConstantOp op){
      DNF res;
      if (op.value().getBoolValue() != inv) {
        DNFConjunct conj;
        res.push_back(conj);
      }
      return res;
    })
    .Case<comb::OrOp>([&](comb::OrOp op){
      auto combi = inv ? combineAnd : combineOr;
      return mapCombine(op.inputs(), combi, inv, trAnalysis);
    })
    .Case<comb::AndOp>([&](comb::AndOp op){
      auto combi = inv ? combineOr : combineAnd;
      return mapCombine(op.inputs(), combi, inv, trAnalysis);
    })
    .Case<comb::XorOp>([&](comb::XorOp op){
      return combineXor(op.inputs(), inv, trAnalysis);
    })
    .Case<comb::ICmpOp>([&](comb::ICmpOp op){
      if (op.predicate() == comb::ICmpPredicate::eq) 
        return combineXor(ValueRange{op.lhs(), op.rhs()}, !inv, trAnalysis);
      if (op.predicate() == comb::ICmpPredicate::ne)
        return combineXor(ValueRange{op.lhs(), op.rhs()}, inv, trAnalysis);
      assert(false && "Should be unreachable!");
      // return singleValueDNF(op.result(), inv);
    })
    .Case<llhd::PrbOp>([&](llhd::PrbOp op){
      return singleValueDNF(op.signal(), inv, trAnalysis.getBlockTR(op->getBlock()));
    });
    // .Default([] (Operation *op) {;
    //   assert(false && "Should be unreachable!");
    // });
}

static Value getInvertedValueIfNeeded(OpBuilder builder, DNFNode node) {
  if (auto sigTy = node.first.first.getType().dyn_cast<llhd::SigType>()) {
    node.first.first = builder.createOrFold<llhd::PrbOp>(node.first.first.getLoc(), sigTy.getUnderlyingType(), node.first.first);
  }
  if (node.second) {
    Value allset = builder.create<hw::ConstantOp>(node.first.first.getLoc(), builder.getI1Type(), 1);
    return builder.create<comb::XorOp>(node.first.first.getLoc(), node.first.first, allset);
  }
  return node.first.first;
}

static bool dnfToTriggers(OpBuilder &builder,
                   RegData &regData, DrvOp op, int pastTR, int presentTR,
                   DNF &dnf) {
  // Drive is never executed thus no reg has to be inserted.
  if (dnf.empty()) {
    op->dropAllReferences();
    op->erase();
    return false;
  }

  // Drive is always executed and thus no reg needed.
  if (dnf.size() == 1 && dnf.begin()->empty()) {
    // TODO: is this the correct way to delete an operand?
    op.enable().dropAllUses();
    op.enableMutable().clear();
    return false;
  }

  for (DNFConjunct conj : dnf) {
    if (conj.empty())
      continue;

    // if (conj.size() == 1) {
      // TODO: here I assume that every value is a probed signal, if we allow
      // opaque values in buildDnf this here has to be changed.
      // TODO: this semantics is probably incorrect, shouldn't the value of the
      // signal itself be used?
    //   regData.addTrigger(builder, getInvertedValueIfNeeded(builder, *conj.begin()),
    //                      RegMode::high, op.time());
    //   continue;
    // }


    // TODO: here I assume that every value is a probed signal, if we allow
    // opaque values in buildDnf this here has to be changed.
    bool triggerAdded = false;
    std::vector<std::pair<Value, bool>> levels;
    for (DNFNode node : conj) {
      auto f = conj.find({node.first.first, node.first.second == presentTR ? pastTR : presentTR});
      if (f != conj.end() && f->second != node.second) {
        Value signal = node.first.first;
        std::vector<Value> andTerm;

        for (DNFNode term : conj) {
          if (term.first.first != signal) {
            andTerm.push_back(getInvertedValueIfNeeded(builder, term));
          }
        }

        Value gate = Value();
        if (andTerm.size() == 1)
          gate = andTerm[0];
        if (andTerm.size() > 1)
          gate = builder.create<comb::AndOp>(andTerm[0].getLoc(), andTerm);

        if ((f->first.second == pastTR && node.first.second == presentTR && !node.second) ||
         (f->first.second == presentTR && node.first.second == pastTR && node.second)) {
          // Rising edge triggered
          if (auto sigTy = node.first.first.getType().dyn_cast<llhd::SigType>()) {
            node.first.first = builder.createOrFold<llhd::PrbOp>(node.first.first.getLoc(), sigTy.getUnderlyingType(), node.first.first);
          }
          regData.addTrigger(builder, node.first.first, RegMode::rise,
                                     op.time(), gate);
        } else if ((f->first.second == pastTR && node.first.second == presentTR && node.second) ||
         (f->first.second == presentTR && node.first.second == pastTR && !node.second)) {
          // Falling edge triggered
          if (auto sigTy = node.first.first.getType().dyn_cast<llhd::SigType>()) {
            node.first.first = builder.createOrFold<llhd::PrbOp>(node.first.first.getLoc(), sigTy.getUnderlyingType(), node.first.first);
          }
          regData.addTrigger(builder, node.first.first, RegMode::fall,
                                     op.time(), gate);
        } else {
          assert(false && "Unreachable");
        }

        triggerAdded = true;
        break;
      }
      if (node.first.second == presentTR)
        levels.push_back({node.first.first, node.second});
    }

    if (!triggerAdded) {
      for (auto level : levels) {
        if (auto sigTy = level.first.getType().dyn_cast<llhd::SigType>()) {
          level.first = builder.createOrFold<llhd::PrbOp>(level.first.getLoc(), sigTy.getUnderlyingType(), level.first);
        }
        regData.addTrigger(builder, level.first, level.second ? RegMode::low : RegMode::high, op.time());
      }
    }

  }
  return true;
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
        return failure(); //op.emitError("Only one wait operation per process supported!");
      }
      // Check that the block containing the wait is the only exiting block of
      // that TR
      if (!trAnalysis.hasSingleExitBlock(
              trAnalysis.getBlockTR(op.getOperation()->getBlock()))) {
        return failure(); //op.emitError(
            // "Block with wait terminator has to be the only exiting block "
            // "of that temporal region!");
      }
      seenWait = true;
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      // signalPassFailure();
      return;
    }

    if (!seenWait) {
      // proc.emitError("Block with wait terminator has to be present for "
      //                "desequentialization to be applicable!");
      // signalPassFailure();
      return;
    }

    OpBuilder builder(proc);
    proc.walk([&](DrvOp op) {
      if (!op.enable())
        return;

      builder.setInsertionPoint(op);
      int presentTR = trAnalysis.getBlockTR(op.getOperation()->getBlock());

      // Transform the enable condition of the drive into DNF
      DNF dnf = buildDnf(op.enable(), false, trAnalysis);

      // Translate the DNF to a list of triggers for the reg instruction
      RegData regData;
      if (dnfToTriggers(builder, regData, op, pastTR, presentTR, dnf)) {
        if (regData.triggers.empty()) {
          emitError(op->getLoc(), "No valid reg trigger found!");
          signalPassFailure();
          return;
        }

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
      }
    });

    builder.setInsertionPoint(proc);
    // Create a new entity with the same name and type as the process it's
    // replacing
    EntityOp entity = builder.create<llhd::EntityOp>(proc.getLoc(), proc.ins());
    entity.setName(proc.getName());
    entity->setAttr("type", proc->getAttr("type"));
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

    // TODO
    // proc.replaceAllSymbolUses(entity.getName(), module.getOperation());

    // Delete the process
    proc.getOperation()->dropAllDefinedValueUses();
    proc.getOperation()->dropAllReferences();
    proc.getOperation()->erase();
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::llhd::createDesequentializationPass() {
  return std::make_unique<DesequentializationPass>();
}
