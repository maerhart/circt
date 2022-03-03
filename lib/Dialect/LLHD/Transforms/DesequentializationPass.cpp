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
// ((value, TR), inv)
using DNFNode = std::pair<std::pair<uint32_t, int>, bool>;
using DNFConjunct = std::map<std::pair<uint32_t, int>, bool>;

static bool comp(const DNFConjunct &a, const DNFConjunct &b)
  // { return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()); };
  { printf("Do compare\n"); bool res = std::lexicographical_compare(a.cbegin(), a.cend(), b.cbegin(), b.cend()); printf("compared\n"); return res;};

static size_t hash(const DNFConjunct &a) { return llvm::hash_combine_range(a.begin(), a.end()); };

// using DNF = std::set<DNFConjunct, decltype(&comp)>;
using DNF = std::vector<DNFConjunct>;

static void combineAndInner(DNF &lhs, DNF &rhs, DNF &res) {
  for (DNFConjunct lhsConjunct : lhs) {
    for (DNFConjunct rhsConjunct : rhs) {
      DNFConjunct newConj;
      for (DNFNode node : lhsConjunct) {
        auto present = newConj.find(node.first);
        if (present != newConj.end() && present->second != node.second && present->first.second == node.first.second) {
          newConj.clear();
          break;
        }
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
      for (int i = 0; i < (int)res.size(); i++) {
        if(conj.size() == res[i].size() && std::equal(conj.begin(), conj.end(), res[i].begin(), [](const DNFNode &a, const DNFNode &b){ return a.first == b.first; })) {
          auto itera = conj.begin();
          auto iterb = res[i].begin();
          for (; itera != conj.end() && iterb != res[i].end(); itera++, (void) iterb++) {
            if (itera->first == iterb->first && itera->second != iterb->second) {
              conj.erase(itera--);
              res[i].erase(iterb--);
            }
          }
        }
      }
      res.push_back(DNFConjunct(conj));
    }
  }
  return res;
}

static DNF buildDnf(Value value, bool inv, TemporalRegionAnalysis &trAnalysis, DenseMap<Value, uint32_t> &valueMap, std::map<std::pair<uint32_t, bool>, DNF> &memo);

static DNF mapCombine(Operation::operand_range range, std::function<DNF(std::vector<DNF>&)> combiner, bool inv, TemporalRegionAnalysis &trAnalysis, DenseMap<Value, uint32_t> &valueMap, std::map<std::pair<uint32_t, bool>, DNF> &memo) {
  std::vector<DNF> dnf;
  for (Value input : range) {
    dnf.push_back(buildDnf(input, inv, trAnalysis, valueMap, memo));
  }
  DNF res = combiner(dnf);
  sort(res.begin(), res.end(), [] (const DNFConjunct &a, const DNFConjunct &b) {return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); });
  res.erase(std::unique(res.begin(), res.end(), [](const DNFConjunct &a, const DNFConjunct &b) { return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()); }), res.end());


  // printf("%zu\n", res.size());

  // printf("[");
  // for (const DNFConjunct &conj : res) {
  //   printf("[");
  //   for (const DNFNode node : conj) {
  //     printf("(%u,%u,%u), ", node.first.first, node.first.second, node.second);
  //   }
  //   printf("], ");
  //   printf("%zu, ", conj.size());
  // }
  // printf("]\n");

  return res;
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

static DNF combineXor(ValueRange range, bool inv, TemporalRegionAnalysis &trAnalysis, DenseMap<Value, uint32_t> &valueMap, std::map<std::pair<uint32_t, bool>, DNF> &memo) {
  std::vector<DNF> dnfPos;
  std::vector<DNF> dnfNeg;
  for (Value input : range) {
    dnfPos.push_back(buildDnf(input, true, trAnalysis, valueMap, memo));
    dnfNeg.push_back(buildDnf(input, false, trAnalysis, valueMap, memo));
  }

  DNF lhsNeg = combineXorHelper(dnfPos[0], dnfPos[1], dnfNeg[0], dnfNeg[1], true);
  DNF lhsPos = combineXorHelper(dnfPos[0], dnfPos[1], dnfNeg[0], dnfNeg[1], false);
  for (int i = 2, e = (int) dnfPos.size(); i < e; i++) {
    lhsNeg = combineXorHelper(lhsPos, dnfPos[i], lhsNeg, dnfNeg[i], true);
    lhsPos = combineXorHelper(lhsPos, dnfPos[i], lhsNeg, dnfNeg[i], false);
  }

  DNF res = inv ? lhsNeg : lhsPos;
  sort(res.begin(), res.end(), [] (const DNFConjunct &a, const DNFConjunct &b) {return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); });
  res.erase(std::unique(res.begin(), res.end(), [](const DNFConjunct &a, const DNFConjunct &b) { return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()); }), res.end());
  return res;
}

static DNF singleValueDNF(uint32_t value, bool inv, int tr) {
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
static DNF buildDnf(Value value, bool inv, TemporalRegionAnalysis &trAnalysis, DenseMap<Value, uint32_t> &valueMap, std::map<std::pair<uint32_t, bool>, DNF> &memo) {

  auto find = memo.find({valueMap[value], inv});

  if (find != memo.end()) return find->second;

  if (!value.getType().isSignlessInteger(1)) {
    DNF res = singleValueDNF(valueMap[value], inv, 0);
    memo[{valueMap[value], inv}] = res;
    return res;
  }
    // emitError(value.getLoc(), "Only one-bit signless integers supported!");

  DNF res = TypeSwitch<Operation*, DNF>(value.getDefiningOp())
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
      // printf("or\n");
      return mapCombine(op.inputs(), combi, inv, trAnalysis, valueMap, memo);
    })
    .Case<comb::AndOp>([&](comb::AndOp op){
      auto combi = inv ? combineOr : combineAnd;
      // printf("and\n");
      return mapCombine(op.inputs(), combi, inv, trAnalysis, valueMap, memo);
    })
    .Case<comb::XorOp>([&](comb::XorOp op){
      // printf("xor\n");
      if (op.isBinaryNot()) return buildDnf(op.inputs()[0], !inv, trAnalysis, valueMap, memo);
      return combineXor(op.inputs(), inv, trAnalysis, valueMap, memo);
    })
    .Case<comb::ICmpOp>([&](comb::ICmpOp op){
      // printf("icmp\n");
      if (op.predicate() == comb::ICmpPredicate::eq) 
        return combineXor(ValueRange{op.lhs(), op.rhs()}, !inv, trAnalysis, valueMap, memo);
      if (op.predicate() == comb::ICmpPredicate::ne)
        return combineXor(ValueRange{op.lhs(), op.rhs()}, inv, trAnalysis, valueMap, memo);
      assert(false && "Should be unreachable!");
      // return singleValueDNF(op.result(), inv);
    })
    .Case<llhd::PrbOp>([&](llhd::PrbOp op){
      return singleValueDNF(valueMap[op.signal()], inv, trAnalysis.getBlockTR(op->getBlock()));
    })
    .Default([&] (Operation *op) { // opaque value
      return singleValueDNF(valueMap[op->getResult(0)], inv, trAnalysis.getBlockTR(op->getBlock()));
    });

    memo[{valueMap[value], inv}] = res;
    return res;
}

static Value getInvertedValueIfNeeded(OpBuilder builder, DNFNode node, DenseMap<uint32_t, Value> &revValueMap) {
  Value value = revValueMap[node.first.first];
  if (auto sigTy = value.getType().dyn_cast<llhd::SigType>()) {
    value = builder.createOrFold<llhd::PrbOp>(value.getLoc(), sigTy.getUnderlyingType(), value);
  }
  if (node.second) {
    Value allset = builder.create<hw::ConstantOp>(value.getLoc(), builder.getI1Type(), 1);
    return builder.create<comb::XorOp>(value.getLoc(), value, allset);
  }
  return value;
}

static bool dnfToTriggers(OpBuilder &builder,
                   RegData &regData, DrvOp op, int pastTR, int presentTR,
                   DNF &dnf, DenseMap<uint32_t, Value> &revValueMap) {
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
      Value value = revValueMap[node.first.first];
      if (f != conj.end() && f->second != node.second) {
        Value signal = revValueMap[node.first.first];
        std::vector<Value> andTerm;

        for (DNFNode term : conj) {
          if (revValueMap[term.first.first] != signal) {
            andTerm.push_back(getInvertedValueIfNeeded(builder, term, revValueMap));
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
          if (auto sigTy = value.getType().dyn_cast<llhd::SigType>()) {
            value = builder.createOrFold<llhd::PrbOp>(value.getLoc(), sigTy.getUnderlyingType(), value);
          }
          regData.addTrigger(builder, value, RegMode::rise,
                                     op.time(), gate);
        } else if ((f->first.second == pastTR && node.first.second == presentTR && node.second) ||
         (f->first.second == presentTR && node.first.second == pastTR && !node.second)) {
          // Falling edge triggered
          if (auto sigTy = value.getType().dyn_cast<llhd::SigType>()) {
            value = builder.createOrFold<llhd::PrbOp>(value.getLoc(), sigTy.getUnderlyingType(), value);
          }
          regData.addTrigger(builder, value, RegMode::fall,
                                     op.time(), gate);
        } else {
          assert(false && "Unreachable");
        }

        triggerAdded = true;
        break;
      }
      if (node.first.second == presentTR)
        levels.push_back({value, node.second});
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

    DenseMap<Value, uint32_t> valueMap;
    DenseMap<uint32_t, Value> revValueMap;
    uint32_t counter = 0;
    for (Value arg : proc.getArguments()) {
      valueMap.insert({arg, counter});
      revValueMap.insert({counter++, arg});
    }
    proc.walk([&](Operation *op) {
      for (Value res : op->getResults()) {
        valueMap.insert({res, counter});
        revValueMap.insert({counter++, res});
      }
    });

    std::map<std::pair<uint32_t, bool>, DNF> memo;

    OpBuilder builder(proc);
    proc.walk([&](DrvOp op) {
      if (!op.enable())
        return;

      builder.setInsertionPoint(op);
      int presentTR = trAnalysis.getBlockTR(op.getOperation()->getBlock());

      // Transform the enable condition of the drive into DNF
      DNF dnf = buildDnf(op.enable(), false, trAnalysis, valueMap, memo);

      // Translate the DNF to a list of triggers for the reg instruction
      RegData regData;
      if (dnfToTriggers(builder, regData, op, pastTR, presentTR, dnf, revValueMap)) {
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
