//===- TranslateToVerilog.cpp - Verilog Printer ---------------------------===//
//
// This is the main LLHD to Verilog Printer implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Target/Verilog/TranslateToVerilog.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormattedStream.h"
#include <cstdlib>

using namespace mlir;

namespace {

enum VerilogOperator {
  Lit,
  Not,
  PlusSign,
  MinusSign,
  Concat,
  Mul,
  Div,
  Mod,
  Add,
  Sub,
  Shl,
  Shr,
  Ugt,
  Uge,
  Ult,
  Ule,
  LogEq,
  LogNe,
  And,
  Xor,
  Or,
  Conditional,
};

enum class Associativity {
  Left,
  Right,
};

struct SignalAlias {
  unsigned offset;
  unsigned length;
  Value original;
  Value index;

  SignalAlias() {}

  SignalAlias(Value original, unsigned offset, unsigned length)
      : offset(offset), length(length), original(original) {}

  SignalAlias(Value original, Value index, unsigned length)
      : offset(0), length(length), original(original), index(index) {}
};

class VerilogPrinter {
public:
  VerilogPrinter(llvm::formatted_raw_ostream &output) : out(output) {}

  LogicalResult printModule(ModuleOp op);
  LogicalResult printOperation(Operation *op, std::vector<llhd::DrvOp> &drives,
                               unsigned indentAmount = 0);

private:
  LogicalResult printType(Type type, bool isAlias=false);
  LogicalResult printUnaryOp(Operation *op, StringRef opSymbol,
                             unsigned indentAmount = 0);
  LogicalResult printBinaryOp(Operation *op, StringRef opSymbol,
                              unsigned indentAmount = 0);
  LogicalResult printSignedBinaryOp(Operation *op, StringRef opSymbol,
                                    unsigned indentAmount = 0);

  /// Prints a SSA value. In case no mapping to a name exists yet, a new one is
  /// added.
  std::string getVariableName(Value value);

  Twine getNewInstantiationName() {
    return Twine("inst_") + Twine(instCount++);
  }

  /// Adds an alias for an existing SSA value. In case doesn't exist, it just
  /// adds the alias as a new value.
  void addAliasVariable(Value alias, Value existing);

  /// Adds an alias for a part of a signal (slice or field). 'start' denotes the
  /// lower end of the signal or the index of the field. 'length' denotes the
  /// length of the slice or 0 for a field.
  void addAliasSignal(Value signal, Value existing, unsigned start, unsigned length);
  void addDynAliasSignal(Value signal, Value existing, Value start,
                         unsigned length);

  void addExpression(Value value, std::string expression, unsigned precedence);
  void addUnaryExpression(Value result, Value value, VerilogOperator op);
  void addBinaryExpression(Value result, Value lhs, VerilogOperator op, Value rhs);

  unsigned instCount = 0;
  llvm::formatted_raw_ostream &out;
  unsigned nextValueNum = 0;
  DenseMap<Value, std::pair<std::string, unsigned>> mapValueToExpression;
  DenseMap<Value, unsigned> timeValueMap;
  DenseMap<Value, SignalAlias> sigAliasMap;
};

static std::string stringifyOperator(VerilogOperator op) {
  switch (op) {
  case VerilogOperator::Add: return " + ";
  case VerilogOperator::Sub: return " - ";
  case VerilogOperator::Mul: return " * ";
  case VerilogOperator::Div: return " / ";
  case VerilogOperator::And: return " & ";
  case VerilogOperator::Or: return " | ";
  case VerilogOperator::Xor: return " ^ ";
  case VerilogOperator::Not: return "~";
  case VerilogOperator::MinusSign: return "-";
  case VerilogOperator::LogEq: return " == ";
  case VerilogOperator::LogNe: return " != ";
  case VerilogOperator::Uge: return " >= ";
  case VerilogOperator::Ugt: return " > ";
  case VerilogOperator::Ule: return " <= ";
  case VerilogOperator::Ult: return " < ";
  case VerilogOperator::Mod: return " % ";
  case VerilogOperator::Shl: return " << ";
  case VerilogOperator::Shr: return " >> ";
  }
  assert(false && "Operator stringification not supported!");
}

static Associativity getOpAssoc(VerilogOperator op) {
  switch (op) {
  case VerilogOperator::Add:
  case VerilogOperator::Sub:
  case VerilogOperator::Mul:
  case VerilogOperator::Div:
  case VerilogOperator::Mod:
  case VerilogOperator::Shl:
  case VerilogOperator::Shr:
  case VerilogOperator::And:
  case VerilogOperator::Or:
  case VerilogOperator::Xor: return Associativity::Left;
  case VerilogOperator::Not:
  case VerilogOperator::MinusSign:
  case VerilogOperator::LogEq:
  case VerilogOperator::LogNe:
  case VerilogOperator::Uge:
  case VerilogOperator::Ugt:
  case VerilogOperator::Ule:
  case VerilogOperator::Ult: return Associativity::Right;
  }
  assert(false && "Operator stringification not supported!");
}

void VerilogPrinter::addUnaryExpression(Value result, Value value,
                                         VerilogOperator op) {
  auto expr = mapValueToExpression[value];
  if (expr.second < op) {
    addExpression(result, stringifyOperator(op) + expr.first, op);
  } else {
    addExpression(result, stringifyOperator(op) + "(" + expr.first + ")", op);
  }
}

void VerilogPrinter::addBinaryExpression(Value result, Value lhs,
                                         VerilogOperator op, Value rhs) {
  auto lhsExpr = mapValueToExpression[lhs];
  auto rhsExpr = mapValueToExpression[rhs];
  std::string expr = "";
  if (getOpAssoc(op) == Associativity::Left) {
    if (lhsExpr.second > op)
      expr += "(";
    expr += lhsExpr.first;
    if (lhsExpr.second > op)
      expr += ")";
    expr += stringifyOperator(op);
    if (rhsExpr.second >= op)
      expr += "(";
    expr += rhsExpr.first;
    if (rhsExpr.second >= op)
      expr += ")";
  } else { // Right associative
    if (lhsExpr.second >= op)
      expr += "(";
    expr += lhsExpr.first;
    if (lhsExpr.second >= op)
      expr += ")";
    expr += stringifyOperator(op);
    if (rhsExpr.second > op)
      expr += "(";
    expr += rhsExpr.first;
    if (rhsExpr.second > op)
      expr += ")";
  }
  addExpression(result, expr, op);
}

void VerilogPrinter::addAliasSignal(Value signal, Value existing,
                                    unsigned offset, unsigned length) {
  if (!sigAliasMap.count(existing)) {
    SignalAlias alias(existing, offset, length);
    sigAliasMap.try_emplace(signal, alias);
  } else {
    SignalAlias &alias = sigAliasMap[existing];
    alias.length = length;
    alias.offset += offset;
  }
}

void VerilogPrinter::addDynAliasSignal(Value signal, Value existing,
                                       Value index, unsigned length) {
  if (!sigAliasMap.count(existing)) {
    SignalAlias alias(existing, index, length);
    sigAliasMap.try_emplace(signal, alias);
  } else {
    SignalAlias &alias = sigAliasMap[existing];
    alias.length = length;
    assert(!alias.index && "Nested aliasing of dynamic slices is currently not supported!");
    alias.index = index;
  }
}

LogicalResult VerilogPrinter::printModule(ModuleOp module) {
  WalkResult result = module.walk([this](llhd::EntityOp entity) -> WalkResult {
    // An EntityOp always has a single block
    Block &entryBlock = entity.body().front();

    // Print the module signature
    out << "module _" << entity.getName();
    if (!entryBlock.args_empty()) {
      out << "(";
      for (unsigned int i = 0, e = entryBlock.getNumArguments(); i < e; ++i) {
        out << (i > 0 ? ", " : "")
            << (i < entity.ins().getZExtValue() ? "input " : "output ");
        printType(entryBlock.getArgument(i).getType());
        out << " " << getVariableName(entryBlock.getArgument(i));
      }
      out << ")";
    }
    out << ";\n";

    // Collect all drives to perform them all together in an always_comb block
    // at the end
    std::vector<llhd::DrvOp> drives;

    // Print the operations within the entity
    for (auto iter = entryBlock.begin();
         iter != entryBlock.end() && !isa<llhd::TerminatorOp>(iter); ++iter) {
      if (failed(printOperation(&(*iter), drives, 4)))
        return emitError(iter->getLoc(), "Operation not supported!");
    }
      // if (!drives.empty()) {
      //   out.PadToColumn(4);
      //   out << "always@(*) begin\n";
      //   for (llhd::DrvOp drv : drives) {
      //     out.PadToColumn(8);
      //     if (drv.enable())
      //       out << "if (" << getVariableName(drv.enable()) << ") ";
      //     out << getVariableName(drv.signal()) << " <= #("
      //         << timeValueMap.lookup(drv.time()) << "ns) " << getVariableName(drv.value()) << ";\n";
      //   }
      //   out.PadToColumn(4);
      //   out << "end\n";
      // }

    out << "endmodule\n";
    // Reset variable name counter as variables only have to be unique within a
    // module
    nextValueNum = 0;
    return WalkResult::advance();
  });
  // if printing of a single operation failed, fail the whole translation
  return failure(result.wasInterrupted());
}

LogicalResult VerilogPrinter::printBinaryOp(Operation *inst, StringRef opSymbol,
                                            unsigned indentAmount) {
  // Check that the operation is indeed a binary operation
  if (inst->getNumOperands() != 2) {
    return emitError(inst->getLoc(),
                     "This operation does not have two operands!");
  }
  if (inst->getNumResults() != 1) {
    return emitError(inst->getLoc(),
                     "This operation does not have one result!");
  }

  // Print the operation
  out.PadToColumn(indentAmount);
  //out << "wire ";
  if (failed(printType(inst->getResult(0).getType())))
    return failure();
  out << " " << getVariableName(inst->getResult(0)) << " = "
      << getVariableName(inst->getOperand(0)) << " " << opSymbol << " "
      << getVariableName(inst->getOperand(1)) << ";\n";
  return success();
}

LogicalResult VerilogPrinter::printSignedBinaryOp(Operation *inst,
                                                  StringRef opSymbol,
                                                  unsigned indentAmount) {
  // Note: the wire is not declared as signed, because if you have the result
  // of two signed operation as an input to an unsigned operation, it would
  // perform the signed version (sometimes this is not a problem because it's
  // the same, but e.g. for modulo it would make a difference). Alternatively
  // we could use $unsigned at every operation where it makes a difference,
  // but that would look a lot uglier, we could also track which variable is
  // signed and which unsigned and only do the conversion when necessary, but
  // that is more effort. Also, because we have the explicit $signed at every
  // signed operation, this isn't a problem for further signed operations.
  //
  // Check that the operation is indeed a binary operation
  if (inst->getNumOperands() != 2) {
    return emitError(inst->getLoc(),
                     "This operation does not have two operands!");
  }
  if (inst->getNumResults() != 1) {
    return emitError(inst->getLoc(),
                     "This operation does not have one result!");
  }

  // Print the operation
  out.PadToColumn(indentAmount);
  //out << "wire ";
  if (failed(printType(inst->getResult(0).getType())))
    return failure();
  out << " " << getVariableName(inst->getResult(0)) << " = $signed("
      << getVariableName(inst->getOperand(0)) << ") " << opSymbol << " $signed("
      << getVariableName(inst->getOperand(1)) << ");\n";
  return success();
}

LogicalResult VerilogPrinter::printUnaryOp(Operation *inst, StringRef opSymbol,
                                           unsigned indentAmount) {
  // Check that the operation is indeed a unary operation
  if (inst->getNumOperands() != 1) {
    return emitError(inst->getLoc(),
                     "This operation does not have exactly one operand!");
  }
  if (inst->getNumResults() != 1) {
    return emitError(inst->getLoc(),
                     "This operation does not have one result!");
  }

  // Print the operation
  out.PadToColumn(indentAmount);
  //out << "wire ";
  if (failed(printType(inst->getResult(0).getType())))
    return failure();
  out << " " << getVariableName(inst->getResult(0)) << " = " << opSymbol
      << getVariableName(inst->getOperand(0)) << ";\n";
  return success();
}

LogicalResult VerilogPrinter::printOperation(Operation *inst,
                                             std::vector<llhd::DrvOp> &drives, unsigned indentAmount) {
  if (auto op = dyn_cast<llhd::ConstOp>(inst)) {
    if (IntegerAttr intAttr = op.value().dyn_cast<IntegerAttr>()) {
      // addExpression(op.getResult(),
      //               (Twine(op.getResult().getType().getIntOrFloatBitWidth()) +
      //                "'d" + Twine(intAttr.getValue().getZExtValue()))
      //                   .str(), VerilogOperator::Lit);
      // return success();
      // Integer constant
      out.PadToColumn(indentAmount);
      //out << "wire ";
      if (failed(printType(inst->getResult(0).getType())))
        return failure();
      out << " " << getVariableName(inst->getResult(0)) << " = "
          << op.getResult().getType().getIntOrFloatBitWidth() << "'d"
          << intAttr.getValue().getZExtValue() << ";\n";
      return success();
    }
    if (llhd::TimeAttr timeAttr = op.value().dyn_cast<llhd::TimeAttr>()) {
      // Time Constant
      // if (timeAttr.getTime() == 0 && timeAttr.getDelta() != 1) {
      //   return emitError(
      //       op.getLoc(),
      //       "Not possible to translate a time attribute with 0 real "
      //       "time and non-1 delta.");
      // }
      // Track time value for future use
      timeValueMap.insert(
          std::make_pair(inst->getResult(0), timeAttr.getTime()));
      return success();
    }
    return failure();
  }
  if (auto op = dyn_cast<ConstantOp>(inst)) {
    if (IntegerAttr intAttr = op.value().dyn_cast<IntegerAttr>()) {
      // addExpression(op.getResult(),
      //               (Twine(op.getResult().getType().getIntOrFloatBitWidth()) +
      //                "'d" + Twine(intAttr.getValue().getZExtValue()))
      //                   .str(), VerilogOperator::Lit);
      // return success();
      // Integer constant
      out.PadToColumn(indentAmount);
      //out << "wire ";
      if (failed(printType(inst->getResult(0).getType())))
        return failure();
      out << " " << getVariableName(inst->getResult(0)) << " = "
          << op.getResult().getType().getIntOrFloatBitWidth() << "'d"
          << op.value().cast<IntegerAttr>().getValue().getZExtValue() << ";\n";
      return success();
    }
    return failure();
  }
  if (auto op = dyn_cast<llhd::SigOp>(inst)) {
    out.PadToColumn(indentAmount);
    //out << "var ";
    if (failed(printType(inst->getResult(0).getType())))
      return failure();
    out << " " << getVariableName(inst->getResult(0)) << " = "
        << getVariableName(op.init()) << ";\n";
    return success();
  }
  if (auto op = dyn_cast<llhd::PrbOp>(inst)) {
    // Prb is a nop, it just defines an alias for an already existing wire
    addAliasVariable(inst->getResult(0), op.signal());
    return success();
  }
  if (auto drv = dyn_cast<llhd::DrvOp>(inst)) {
    drives.push_back(drv);
    out.PadToColumn(4);
    out << "always@(*) begin\n";
      out.PadToColumn(8);
      if (drv.enable())
        out << "if (" << getVariableName(drv.enable()) << ") ";
      out << getVariableName(drv.signal()) << " <= ";
      if (timeValueMap.lookup(drv.time()) != 0)
        out << "#(" << timeValueMap.lookup(drv.time()) << "ns) ";
      out << getVariableName(drv.value()) << ";\n";
    out.PadToColumn(4);
    out << "end\n";
    return success();
  }
  if (llhd::AndOp op = dyn_cast<llhd::AndOp>(inst)) {
    addBinaryExpression(op.getResult(), op.lhs(), VerilogOperator::And, op.rhs());
    return success();
    // return printBinaryOp(inst, "&", indentAmount);
  }
  if (llhd::OrOp op = dyn_cast<llhd::OrOp>(inst)) {
    addBinaryExpression(op.getResult(), op.lhs(), VerilogOperator::Or, op.rhs());
    return success();
    // return printBinaryOp(inst, "|", indentAmount);
  }
  if (llhd::XorOp op = dyn_cast<llhd::XorOp>(inst)) {
    addBinaryExpression(op.getResult(), op.lhs(), VerilogOperator::Xor, op.rhs());
    return success();
    // return printBinaryOp(inst, "^", indentAmount);
  }
  if (llhd::EqOp op = dyn_cast<llhd::EqOp>(inst)) {
    addBinaryExpression(op.getResult(), op.lhs(), VerilogOperator::LogEq, op.rhs());
    return success();
  }
  if (llhd::NeOp op = dyn_cast<llhd::NeOp>(inst)) {
    addBinaryExpression(op.getResult(), op.lhs(), VerilogOperator::LogNe, op.rhs());
    return success();
  }
  if (llhd::NotOp op = dyn_cast<llhd::NotOp>(inst)) {
    addUnaryExpression(op.getResult(), op.value(), VerilogOperator::Not);
    // return printUnaryOp(inst, "~", indentAmount);
    return success();
  }
  if (auto op = dyn_cast<llhd::ShlOp>(inst)) {
    assert(!op.base().getType().isa<llhd::SigType>() && "Shifts of signals not supported!");
    unsigned baseWidth = op.getBaseWidth();
    unsigned hiddenWidth = op.getHiddenWidth();
    unsigned combinedWidth = baseWidth + hiddenWidth;

    out.PadToColumn(indentAmount);
    out << "wire [" << (combinedWidth - 1) << ":0] "
        << getVariableName(op.result()) << "tmp0 = {"
        << getVariableName(op.base()) << ", "
        << getVariableName(op.hidden()) << "};\n";

    out.PadToColumn(indentAmount);
    out << "wire [" << (combinedWidth - 1) << ":0] "
        << getVariableName(op.result())
        << "tmp1 = " << getVariableName(op.result()) << "tmp0 << "
        << getVariableName(op.amount()) << ";\n";

    out.PadToColumn(indentAmount);
    if (failed(printType(op.result().getType(), true)))
      return failure();
    out << " " << getVariableName(op.result()) << " = "
        << getVariableName(op.result()) << "tmp1[" << (combinedWidth - 1)
        << ":" << hiddenWidth << "];\n";

    return success();
  }
  if (auto op = dyn_cast<llhd::ShrOp>(inst)) {
    assert(!op.base().getType().isa<llhd::SigType>() && "Shifts of signals not supported!");
    unsigned baseWidth = op.getBaseWidth();
    unsigned hiddenWidth = op.getHiddenWidth();
    unsigned combinedWidth = baseWidth + hiddenWidth;

    out.PadToColumn(indentAmount);
    out << "wire [" << (combinedWidth - 1) << ":0] "
        << getVariableName(op.result()) << "tmp0 = {"
        << getVariableName(op.hidden()) << ", "
        << getVariableName(op.base()) << "};\n";

    out.PadToColumn(indentAmount);
    out << "wire [" << (combinedWidth - 1) << ":0] "
        << getVariableName(op.result())
        << "tmp1 = " << getVariableName(op.result()) << "tmp0 >> "
        << getVariableName(op.amount()) << ";\n";

    out.PadToColumn(indentAmount);
    if (failed(printType(op.result().getType(), true)))
      return failure();
    out << " " << getVariableName(op.result()) << " = "
        << getVariableName(op.result()) << "tmp1[" << (baseWidth - 1)
        << ":0];\n";

    return success();
  }
  if (isa<llhd::NegOp>(inst)) {
    addUnaryExpression(inst->getResult(0), inst->getOperand(0), VerilogOperator::MinusSign);
    return success();
    // return printUnaryOp(inst, "-", indentAmount);
  }
  if (isa<AddIOp>(inst)) {
    addBinaryExpression(inst->getResult(0), inst->getOperand(0), VerilogOperator::Add, inst->getOperand(1));
    return success();
    // return printBinaryOp(inst, "+", indentAmount);
  }
  if (isa<SubIOp>(inst)) {
    addBinaryExpression(inst->getResult(0), inst->getOperand(0), VerilogOperator::Sub, inst->getOperand(1));
    return success();
    // return printBinaryOp(inst, "-", indentAmount);
  }
  if (isa<MulIOp>(inst)) {
    addBinaryExpression(inst->getResult(0), inst->getOperand(0), VerilogOperator::Mul, inst->getOperand(1));
    return success();
    // return printBinaryOp(inst, "*", indentAmount);
  }
  if (isa<UnsignedDivIOp>(inst)) {
    addBinaryExpression(inst->getResult(0), inst->getOperand(0), VerilogOperator::Div, inst->getOperand(1));
    return success();
    // return printBinaryOp(inst, "/", indentAmount);
  }
  if (isa<SignedDivIOp>(inst)) {
    return printSignedBinaryOp(inst, "/", indentAmount);
  }
  if (isa<UnsignedRemIOp>(inst)) {
    addBinaryExpression(inst->getResult(0), inst->getOperand(0), VerilogOperator::Mod, inst->getOperand(1));
    return success();
    // % in Verilog is the remainder in LLHD semantics
    // return printBinaryOp(inst, "%", indentAmount);
  }
  if (isa<SignedRemIOp>(inst)) {
    // % in Verilog is the remainder in LLHD semantics
    return printSignedBinaryOp(inst, "%", indentAmount);
  }
  if (isa<llhd::SModOp>(inst)) {
    return emitError(inst->getLoc(),
                     "Signed modulo operation is not yet supported!");
  }
  if (auto op = dyn_cast<CmpIOp>(inst)) {
    switch (op.getPredicate()) {
    case mlir::CmpIPredicate::eq:
    {
      addBinaryExpression(inst->getResult(0), inst->getOperand(0),
                          VerilogOperator::LogEq, inst->getOperand(1));
      return success();
    }
      // return printBinaryOp(inst, "==", indentAmount);
    case mlir::CmpIPredicate::ne:
    {
      addBinaryExpression(inst->getResult(0), inst->getOperand(0),
                          VerilogOperator::LogNe, inst->getOperand(1));
      return success();
    }
      return printBinaryOp(inst, "!=", indentAmount);
    case mlir::CmpIPredicate::sge:
      return printSignedBinaryOp(inst, ">=", indentAmount);
    case mlir::CmpIPredicate::sgt:
      return printSignedBinaryOp(inst, ">", indentAmount);
    case mlir::CmpIPredicate::sle:
      return printSignedBinaryOp(inst, "<=", indentAmount);
    case mlir::CmpIPredicate::slt:
      return printSignedBinaryOp(inst, "<", indentAmount);
    case mlir::CmpIPredicate::uge:
    {
      addBinaryExpression(inst->getResult(0), inst->getOperand(0),
                          VerilogOperator::Uge, inst->getOperand(1));
      return success();
    }
      // return printBinaryOp(inst, ">=", indentAmount);
    case mlir::CmpIPredicate::ugt:
    {
      addBinaryExpression(inst->getResult(0), inst->getOperand(0),
                          VerilogOperator::Ugt, inst->getOperand(1));
      return success();
    }
      // return printBinaryOp(inst, ">", indentAmount);
    case mlir::CmpIPredicate::ule:
    {
      addBinaryExpression(inst->getResult(0), inst->getOperand(0),
                          VerilogOperator::Ule, inst->getOperand(1));
      return success();
    }
      // return printBinaryOp(inst, "<=", indentAmount);
    case mlir::CmpIPredicate::ult:
    {
      addBinaryExpression(inst->getResult(0), inst->getOperand(0),
                          VerilogOperator::Ult, inst->getOperand(1));
      return success();
    }
      // return printBinaryOp(inst, "<", indentAmount);
    }
    return failure();
  }
  if (auto op = dyn_cast<llhd::InstOp>(inst)) {
    out.PadToColumn(indentAmount);
    out << "_" << op.callee() << " " << getNewInstantiationName();
    if (op.inputs().size() > 0 || op.outputs().size() > 0)
      out << " (";
    unsigned counter = 0;
    for (Value arg : op.inputs()) {
      if (counter++ > 0)
        out << ", ";
      out << getVariableName(arg);
    }
    for (Value arg : op.outputs()) {
      if (counter++ > 0)
        out << ", ";
      out << getVariableName(arg);
    }
    if (op.inputs().size() > 0 || op.outputs().size() > 0)
      out << ")";
    out << ";\n";
    return success();
  }
  if (auto op = dyn_cast<llhd::VecOp>(inst)) {
    return failure();
    // TODO
    out.PadToColumn(indentAmount);
    out << "reg ";
    if (failed(printType(*op.values().getType().begin())))
      return failure();
    out << " " << getVariableName(op.result())
        << " [0:" << (op.values().size() - 1) << "];\n";
    unsigned counter = 0;
    out.PadToColumn(indentAmount);
    out << "initial begin\n";
    for (Value val : op.values()) {
      out.PadToColumn(2 * indentAmount);
      out << getVariableName(op.result()) << "[" << counter
          << "] = " << getVariableName(val) << ";\n";
      counter++;
    }
    out.PadToColumn(indentAmount);
    out << "end\n";
    return success();
  }
  if (auto op = dyn_cast<llhd::ExtsOp>(inst)) {
    // out << "wire ";
    if (op.target().getType().isa<llhd::SigType>()) {
      addAliasSignal(op.result(), op.target(), op.startAttr().getInt(),
                     op.getSliceSize());
      return success();
    }
    out.PadToColumn(indentAmount);
    if (failed(printType(op.result().getType(), true)))
      return failure();
    out << " " << getVariableName(op.result()) << " = "
        << getVariableName(op.target()) << "[" << (op.startAttr().getInt() + op.getSliceSize() - 1);
    // if (op.getSliceSize() > 1) {
      out << ":" << op.startAttr().getInt() << "];\n";
    // } else {
    //   out << "];\n";
    // }
    return success();
  }
  if (auto op = dyn_cast<llhd::ExtfOp>(inst)) {
    if (op.target().getType().isa<llhd::SigType>()) {
      addAliasSignal(op.result(), op.target(), op.indexAttr().getInt(), 0);
      return success();
    }
    out.PadToColumn(indentAmount);
    //out << "wire ";
    if (failed(printType(op.result().getType())))
      return failure();
    out << " " << getVariableName(op.result()) << " = "
        << getVariableName(op.target()) << "[" << op.indexAttr().getInt() << "];\n";
    return success();
  }
  if (auto op = dyn_cast<llhd::DextsOp>(inst)) {
    if (op.target().getType().isa<llhd::SigType>()) {
      addDynAliasSignal(op.result(), op.target(), op.start(),
                        op.getSliceWidth());
      return success();
    }
    out.PadToColumn(indentAmount);
    //out << "wire ";
    if (failed(printType(op.result().getType())))
      return failure();
    out << " " << getVariableName(op.result()) << " = "
        << getVariableName(op.target()) << "[" << getVariableName(op.start());
    // if (op.getSliceWidth() > 1) {
      out << ":(" << getVariableName(op.start()) << " + " << (op.getSliceWidth() - 1)
          << ")];\n";
    // } else {
    //   out << "];\n";
    // }
    return success();
  }
  if (auto op = dyn_cast<llhd::DextfOp>(inst)) {
    if (op.target().getType().isa<llhd::SigType>()) {
      addDynAliasSignal(op.result(), op.target(), op.index(), 0);
      return success();
    }
    out.PadToColumn(indentAmount);
    //out << "wire ";
    if (failed(printType(op.result().getType())))
      return failure();
    out << " " << getVariableName(op.result()) << " = "
        << getVariableName(op.target()) << "[" << getVariableName(op.index()) << "];\n";
    return success();
  }
  if (auto op = dyn_cast<llhd::InssOp>(inst)) {
    out.PadToColumn(indentAmount);
    //out << "wire ";
    if (failed(printType(op.result().getType())))
      return failure();
    out << " " << getVariableName(op.result()) << " = {";
    if (op.startAttr().getInt() + op.getSliceSize() < op.getTargetSize()) {
      out  << getVariableName(op.target()) << "[" << (op.getTargetSize() - 1) << ":" << (op.startAttr().getInt() + op.getSliceSize()) << "], ";
    }
    out << getVariableName(op.slice());
    if (op.startAttr().getInt() > 0) {
      out << ", " << getVariableName(op.target()) << "[" << (op.startAttr().getInt()-1) << ":0]";
    }
    out << "};\n";
    return success();
  }
  if (auto op = dyn_cast<SelectOp>(inst)) {
    out.PadToColumn(indentAmount);
    //out << "wire ";
    if (failed(printType(op.result().getType())))
      return failure();
    out << " " << getVariableName(op.result()) << " = "
        << getVariableName(op.condition()) << " ? "
        << getVariableName(op.getTrueValue()) << " : "
        << getVariableName(op.getFalseValue()) << ";\n";
    return success();
  }
  if (auto op = dyn_cast<llhd::RegOp>(inst)) {
    std::vector<Value> triggers(op.triggers().begin(), op.triggers().end());
    std::vector<Value> values(op.values().begin(), op.values().end());
    std::vector<Value> delays(op.delays().begin(), op.delays().end());
    auto iter = llvm::map_range(op.modes().getValue(), [](Attribute mode) { return llhd::symbolizeRegMode(mode.cast<IntegerAttr>().getInt()).getValue(); });
    std::vector<llhd::RegMode> modes(iter.begin(), iter.end());
    out.PadToColumn(indentAmount);
    out << "always@(";
    bool first = true;
    for (unsigned i = 0; i < triggers.size(); ++i) {
      if (!first)
        out << ", ";
      switch (modes[i]) {
      case llhd::RegMode::rise: {
        out << "posedge ";
        break;
        }
      case llhd::RegMode::fall: {
        out << "negedge ";
        break;
        }
      case llhd::RegMode::low: {
        // TODO
      return failure();
      break;
      }
      case llhd::RegMode::high: {
        // TODO
      return failure();
      break;
      }
      default: {}
      }
      out << getVariableName(triggers[i]);
      if (op.hasGate(i)) {
        out << " iff " << getVariableName(op.getGateAt(i));
      }
      first = false;
    }
    out << ") begin\n";
    // Inside always block
    first = true;
    if (std::equal(values.begin() + 1, values.end(), values.begin()) && std::equal(delays.begin() + 1, delays.end(), delays.begin())) {
      out.PadToColumn(2 * indentAmount);
      out << getVariableName(op.signal()) << " <= ";
      if (timeValueMap.lookup(delays[0]) != 0)
        out << "#(" << timeValueMap.lookup(delays[0]) << "ns) ";
      out << getVariableName(values[0]) << ";\n";
    } else {
    for (unsigned i = 0; i < triggers.size(); ++i) {
      out.PadToColumn(2 * indentAmount);
      if (!first) {
        out << "end else ";
      }
      out << "if (";
      switch (modes[i]) {
      case llhd::RegMode::rise: {
        out << "!$past(" << getVariableName(triggers[i]) << ") && " << getVariableName(triggers[i]);
        break;
        }
      case llhd::RegMode::fall: {
        out << "$past(" << getVariableName(triggers[i]) << ") && !" << getVariableName(triggers[i]);
        break;
        }
      case llhd::RegMode::low: {
        // TODO
      return failure();
      break;
      }
      case llhd::RegMode::high: {
        // TODO
      return failure();
      break;
      }
      case llhd::RegMode::both: {
        // TODO
      return failure();
      break;
      }
      }
      out << ") begin\n";
      out.PadToColumn(3 * indentAmount);
      out << getVariableName(op.signal()) << " <= ";
      if (timeValueMap.lookup(delays[i]) != 0)
        out << "#(" << timeValueMap.lookup(delays[i]) << "ns) ";
      out << getVariableName(values[i]) << ";\n";
      first = false;
    }
    out.PadToColumn(2 * indentAmount);
    out << "end\n";
    }
    // End always block
    out.PadToColumn(indentAmount);
    out << "end\n";
    return success();
  }
  // TODO: insert structural operations here
  return failure();
}

LogicalResult VerilogPrinter::printType(Type type, bool isAlias) {
  if (type.isIntOrFloat()) {
    unsigned int width = type.getIntOrFloatBitWidth();
    out << "wire ";
    if (width != 1)
      out << "[" << (width - 1) << ":0]";
    return success();
  }
  if (llhd::SigType sig = type.dyn_cast<llhd::SigType>()) {
    if (sig.getUnderlyingType().isIntOrFloat()) {
      unsigned int width = sig.getUnderlyingType().getIntOrFloatBitWidth();
      out << (isAlias ? "wire " : "bit ");
      if (width != 1)
        out << "[" << (width - 1) << ":0]";
      return success();
    }
  }
  return failure();
}

std::string VerilogPrinter::getVariableName(Value value) {
  if (sigAliasMap.count(value)) {
    SignalAlias alias = sigAliasMap.lookup(value);
    std::string result =
        mapValueToExpression.lookup(alias.original).first + "[";

    if (alias.index) {
      return (result + getVariableName(alias.index) + "+" +
             Twine(alias.offset) + "+:" + Twine(alias.length) +
             "]").str();
    }
    return (result + Twine(alias.offset + alias.length - 1) + ":" +
           Twine(alias.offset) + "]").str();
  }
  if(mapValueToExpression.insert(std::make_pair(value, std::make_pair(("_" + Twine(nextValueNum)).str(), VerilogOperator::Lit))).second)
    nextValueNum++;
  auto expr = mapValueToExpression[value];
  if (expr.second != VerilogOperator::Lit)
    return "(" + expr.first + ")";
  return expr.first;
}

void VerilogPrinter::addAliasVariable(Value alias, Value existing) {
  assert(mapValueToExpression.count(existing) && "Value to alias already has to be added");
  mapValueToExpression.insert(
      std::make_pair(alias, mapValueToExpression.lookup(existing)));
}

void VerilogPrinter::addExpression(Value value, std::string expression, unsigned precedence) {
  mapValueToExpression.try_emplace(value, std::make_pair(expression, precedence));
}

} // anonymous namespace

LogicalResult mlir::llhd::printVerilog(ModuleOp module, raw_ostream &os) {
  llvm::formatted_raw_ostream out(os);
  VerilogPrinter printer(out);
  return printer.printModule(module);
}

void mlir::llhd::registerToVerilogTranslation() {
  TranslateFromMLIRRegistration registration(
      "llhd-to-verilog", [](ModuleOp module, raw_ostream &output) {
        return printVerilog(module, output);
      });
}
