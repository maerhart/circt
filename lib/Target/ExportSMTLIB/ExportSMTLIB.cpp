//===- ExportSMTLIB.cpp - SMT-LIB Emitter -----=---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main SMT-LIB emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Target/ExportSMTLIB.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "circt/Dialect/SMT/SMTVisitors.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/Format.h"

using namespace circt;
using namespace smt;
using namespace ExportSMTLIB;

#define DEBUG_TYPE "export-smtlib"

namespace {
class SMTEmitter : public smt::SMTOpVisitor<SMTEmitter>,
                   public smt::SMTTypeVisitor<SMTEmitter> {
public:
  SMTEmitter(mlir::raw_indented_ostream &stream,
             const SMTEmissionOptions &options)
      : stream(stream), options(options) {}

  void emit(Block *block) {
    // Declare custom sorts.
    DenseMap<StringAttr, unsigned> declaredSorts;
    block->walk([&](Operation *op) {
      for (Type resTy : op->getResultTypes()) {
        auto sortTy = dyn_cast<SortType>(resTy);
        if (!sortTy)
          continue;
        unsigned arity = sortTy.getSortParams().size();
        arity = sortTy.getSortParams().size();
        if (declaredSorts.contains(sortTy.getIdentifier())) {
          if (declaredSorts[sortTy.getIdentifier()] != arity)
            op->emitError(
                "custom sorts with same identifier but different arity found");
          return;
        }
        stream << "(declare-sort " << sortTy.getIdentifier().getValue() << " "
               << arity << ")\n";
      }
    });
    // Collect all statement operations (ops with no result value).
    block->walk([&](Operation *op) {
      // Declare constants and then only refer to them by identifier later on.
      if (auto declConstOp = dyn_cast<DeclareConstOp>(op)) {
        stream << "(declare-const " << declConstOp.getDeclName() << " ";
        dispatchSMTTypeVisitor(declConstOp.getType());
        stream << ")\n";
        return;
      }
      if (auto declFuncOp = dyn_cast<DeclareFuncOp>(op)) {
        stream << "(declare-fun " << declFuncOp.getDeclName() << " ";
        dispatchSMTTypeVisitor(declFuncOp.getType());
        stream << ")\n";
        return;
      }
      // Statement operations are checked here.
      if (isa<AssertOp, CheckSatOp>(op)) {
        SmallVector<std::pair<std::variant<Operation *, std::string>, bool>>
            worklist;
        worklist.push_back({op, false});
        dfs(worklist);
        stream << "\n";
        return;
      }
    });
    stream << "(exit)\n";
  }

  void dfs(SmallVector<std::pair<std::variant<Operation *, std::string>, bool>>
               &worklist) {
    auto needParentheses = [&](Operation *op) {
      // TODO: don't count solver typed operands
      return op->getNumOperands() + op->getNumRegions() != 0;
    };

    while (!worklist.empty()) {
      auto [opOrString, visited] = worklist.pop_back_val();

      if (std::holds_alternative<std::string>(opOrString)) {
        stream << std::get<std::string>(opOrString);
        if (worklist.empty() || !worklist.back().second)
          stream << " ";
        continue;
      }

      auto *op = std::get<Operation *>(opOrString);

      // Coming back from a leaf, print the closing parenthesis if we opened one
      // in the first place
      if (visited) {
        if (needParentheses(op))
          stream << ")";

        // If there is a sibling left to be processed, add a space.
        if (worklist.empty() || !worklist.back().second)
          stream << " ";

        continue;
      }

      // Forward direction
      if (needParentheses(op))
        stream << "(";
      dispatchSMTOpVisitor(op);
      worklist.push_back({op, true});

      // Push children in reverse order to the stack.
      // Emit a space if there are any children to be processed.
      StringLiteral delimiter = "";
      for (auto &operand : llvm::reverse(op->getOpOperands())) {
        if (isa<SolverType>(operand.get().getType()))
          continue;
        if (auto arg = dyn_cast<BlockArgument>(operand.get());
            arg && isa<ForallOp, ExistsOp>(arg.getOwner()->getParentOp())) {
          worklist.push_back({arg.getOwner()
                                  ->getParentOp()
                                  ->getAttr("boundVarNames")
                                  .cast<ArrayAttr>()[arg.getArgNumber()]
                                  .cast<StringAttr>()
                                  .getValue()
                                  .str(),
                              false});
          delimiter = " ";
          continue;
        }
        auto *defOp = operand.get().getDefiningOp();
        if (!defOp)
          continue;
        worklist.push_back({defOp, false});
        delimiter = " ";
      }
      // TODO: handle the case where a block arg is directly given to the yield
      if (isa<ForallOp, ExistsOp>(op)) {
        worklist.push_back({op->getRegion(0)
                                .front()
                                .getTerminator()
                                ->getOperand(0)
                                .getDefiningOp(),
                            false});
      }

      if (!isa<ApplyFuncOp>(op))
        stream << delimiter;
    }
  }

  //===--------------------------------------------------------------------===//
  // Bit-vector theory operation visitors
  //===--------------------------------------------------------------------===//

  void visitSMTOp(ConstantOp op) {
    unsigned bvWidth = cast<BitVectorType>(op.getValue().getType()).getWidth();
    // TODO: this should use an APInt
    unsigned value = op.getValue().getValue();
    if (options.printBitVectorsInHex) {
      stream << "#x" << llvm::format_hex_no_prefix(value, bvWidth / 4);
      return;
    }

    stream << "#b";
    for (unsigned i = 0; i < bvWidth; ++i) {
      stream << ((value >> (bvWidth - i - 1)) & 1);
    }
  }

  void visitSMTOp(NegOp op) { stream << "bvneg"; }

  void visitSMTOp(AddOp op) { stream << "bvadd"; }

  void visitSMTOp(SubOp op) { stream << "bvsub"; }

  void visitSMTOp(MulOp op) { stream << "bvmul"; }

  void visitSMTOp(URemOp op) { stream << "bvurem"; }

  void visitSMTOp(SRemOp op) { stream << "bvsrem"; }

  // TODO: this operation should be removed
  void visitSMTOp(UModOp op) { visitUnhandledSMTOp(op); }

  void visitSMTOp(SModOp op) { stream << "bvsmod"; }

  void visitSMTOp(ShlOp op) { stream << "bvshl"; }

  void visitSMTOp(LShrOp op) { stream << "bvlshr"; }

  void visitSMTOp(AShrOp op) { stream << "bvashr"; }

  void visitSMTOp(UDivOp op) { stream << "bvudiv"; }

  void visitSMTOp(SDivOp op) { stream << "bvsdiv"; }

  void visitSMTOp(BVNotOp op) { stream << "bvnot"; }

  void visitSMTOp(BVAndOp op) { stream << "bvand"; }

  void visitSMTOp(BVOrOp op) { stream << "bvor"; }

  void visitSMTOp(BVXOrOp op) { stream << "bvxor"; }

  void visitSMTOp(BVNAndOp op) { stream << "bvnand"; }

  void visitSMTOp(BVNOrOp op) { stream << "bvnor"; }

  void visitSMTOp(BVXNOrOp op) { stream << "bvxnor"; }

  void visitSMTOp(ConcatOp op) { stream << "concat"; }

  void visitSMTOp(ExtractOp op) {
    stream << "(_ extract " << (op.getStart() + op.getType().getWidth() - 1)
           << " " << op.getStart() << ")";
  }

  void visitSMTOp(RepeatOp op) {
    stream << "(_ repeat " << op.getCount() << ")";
  }

  void visitSMTOp(BVCmpOp op) {
    switch (op.getPred()) {
    case Predicate::sge:
      stream << "bvsge";
      return;
    case Predicate::sgt:
      stream << "bvsgt";
      return;
    case Predicate::sle:
      stream << "bvsle";
      return;
    case Predicate::slt:
      stream << "bvslt";
      return;
    case Predicate::uge:
      stream << "bvuge";
      return;
    case Predicate::ugt:
      stream << "bvugt";
      return;
    case Predicate::ule:
      stream << "bvule";
      return;
    case Predicate::ult:
      stream << "bvult";
      return;
    }
  }

  //===--------------------------------------------------------------------===//
  // Int theory operation visitors
  //===--------------------------------------------------------------------===//

  void visitSMTOp(IntConstantOp op) { stream << op.getValue(); }

  void visitSMTOp(IntAddOp op) { stream << "+"; }

  void visitSMTOp(IntMulOp op) { stream << "*"; }

  void visitSMTOp(IntSubOp op) { stream << "-"; }

  void visitSMTOp(IntDivOp op) { stream << "div"; }

  void visitSMTOp(IntModOp op) { stream << "mod"; }

  void visitSMTOp(IntRemOp op) { stream << "rem"; }

  void visitSMTOp(IntPowOp op) { stream << "^"; }

  void visitSMTOp(IntCmpOp op) {
    switch (op.getPred()) {
    case IntPredicate::ge:
      stream << ">=";
      return;
    case IntPredicate::le:
      stream << "<=";
      return;
    case IntPredicate::gt:
      stream << ">";
      return;
    case IntPredicate::lt:
      stream << "<";
      return;
    }
  }

  //===--------------------------------------------------------------------===//
  // Core theory operation visitors
  //===--------------------------------------------------------------------===//

  void visitSMTOp(EqOp op) { stream << "="; }

  void visitSMTOp(DistinctOp op) { stream << "distinct"; }

  void visitSMTOp(IteOp op) { stream << "ite"; }

  void visitSMTOp(DeclareConstOp op) { stream << op.getDeclName(); }

  void visitSMTOp(DeclareFuncOp op) { stream << op.getDeclName(); }

  void visitSMTOp(ApplyFuncOp op) { stream << ""; }

  void visitSMTOp(SolverCreateOp op) { visitUnhandledSMTOp(op); }

  void visitSMTOp(AssertOp op) { stream << "assert"; }

  void visitSMTOp(CheckSatOp op) { stream << "check-sat"; }

  void visitSMTOp(PatternCreateOp op) { stream << ""; }

  void visitSMTOp(ForallOp op) {
    stream << "forall (";
    StringLiteral delimiter = "";
    for (auto [arg, name] : llvm::zip(op.getBody().getArguments(),
                                      op.getBoundVarNames().getValue())) {
      stream << delimiter;
      stream << "(" << name.cast<StringAttr>().getValue() << " ";
      dispatchSMTTypeVisitor(arg.getType());
      stream << ")";
      delimiter = " ";
    }
    stream << ") ";
  }

  void visitSMTOp(ExistsOp op) {
    stream << "exists (";
    StringLiteral delimiter = "";
    for (auto [arg, name] : llvm::zip(op.getBody().getArguments(),
                                      op.getBoundVarNames().getValue())) {
      stream << delimiter;
      stream << "(" << name.cast<StringAttr>().getValue() << " ";
      dispatchSMTTypeVisitor(arg.getType());
      stream << ")";
      delimiter = " ";
    }
    stream << ") ";
  }

  void visitSMTOp(BoolConstantOp op) {
    stream << (op.getValue() ? "true" : "false");
  }

  void visitSMTOp(NotOp op) { stream << "not"; }

  void visitSMTOp(AndOp op) { stream << "and"; }

  void visitSMTOp(OrOp op) { stream << "or"; }

  void visitSMTOp(XOrOp op) { stream << "xor"; }

  void visitSMTOp(ImpliesOp op) { stream << "=>"; }

  void visitSMTOp(YieldOp op) { visitUnhandledSMTOp(op); }

  //===--------------------------------------------------------------------===//
  // Array theory operation visitors
  //===--------------------------------------------------------------------===//

  void visitSMTOp(ArrayStoreOp op) { stream << "store"; }

  void visitSMTOp(ArraySelectOp op) { stream << "select"; }

  void visitSMTOp(ArrayBroadcastOp op) {
    stream << "(as const ";
    dispatchSMTTypeVisitor(op.getType());
    stream << ")";
  }

  // TODO: delete this operation
  void visitSMTOp(ArrayDefaultOp op) { visitUnhandledSMTOp(op); }

  //===--------------------------------------------------------------------===//
  // Type visitors
  //===--------------------------------------------------------------------===//

  void visitSMTType(BoolType type) { stream << "Bool"; }

  void visitSMTType(smt::IntegerType type) { stream << "Int"; }

  void visitSMTType(PatternType type) { stream << ""; }

  void visitSMTType(BitVectorType type) {
    stream << "(BitVec " << type.getWidth() << ")";
  }

  void visitSMTType(SolverType type) { visitUnhandledSMTType(type); }

  void visitSMTType(ArrayType type) {
    stream << "(Array ";
    dispatchSMTTypeVisitor(type.getDomainType());
    stream << " ";
    dispatchSMTTypeVisitor(type.getRangeType());
    stream << ")";
  }

  void visitSMTType(SMTFunctionType type) {
    stream << "(";
    StringLiteral nextToken = "";
    for (Type domainTy : type.getDomainTypes()) {
      stream << nextToken;
      dispatchSMTTypeVisitor(domainTy);
      nextToken = " ";
    }
    stream << ") ";
    dispatchSMTTypeVisitor(type.getRangeType());
  }

  void visitSMTType(SortType type) {
    if (!type.getSortParams().empty())
      stream << "(";
    stream << type.getIdentifier().getValue();
    for (Type paramTy : type.getSortParams()) {
      stream << " ";
      dispatchSMTTypeVisitor(paramTy);
    }
    if (!type.getSortParams().empty())
      stream << ")";
  }

private:
  mlir::raw_indented_ostream &stream;
  const SMTEmissionOptions &options;
};
} // namespace

//===----------------------------------------------------------------------===//
// Unified and Split Emitter implementation
//===----------------------------------------------------------------------===//

LogicalResult ExportSMTLIB::exportSMTLIB(Operation *module,
                                         llvm::raw_ostream &os,
                                         const SMTEmissionOptions &options) {
  if (module->getNumRegions() != 1)
    return module->emitError("must have exactly one region");
  if (!module->getRegion(0).hasOneBlock())
    return module->emitError("op region must have exactly one block");

  mlir::raw_indented_ostream ios(os);
  SMTEmitter emitter(ios, options);
  emitter.emit(&module->getRegion(0).front());
  return success();
}

//===----------------------------------------------------------------------===//
// circt-translate registration
//===----------------------------------------------------------------------===//

void ExportSMTLIB::registerExportSMTLIBTranslation() {

  static mlir::TranslateFromMLIRRegistration toSMTLIB(
      "export-smtlib", "export SMT-LIB",
      [](Operation *module, raw_ostream &output) {
        return ExportSMTLIB::exportSMTLIB(module, output);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<smt::SMTDialect>();
      });
}
