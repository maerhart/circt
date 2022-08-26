//===- FuncEmissionPatterns.cpp - Func Dialect Emission Patterns ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the func dialect.
//
//===----------------------------------------------------------------------===//

#include "FuncEmissionPatterns.h"
#include "../EmissionPrinter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::func;
using namespace circt;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit a SystemC module using the SC_MODULE macro and emit all ports as fields
/// of the module. Users of the ports request an expression to be inlined and we
/// simply return the name of the port.
struct FuncEmitter : OpEmissionPattern<FuncOp> {
  using OpEmissionPattern::OpEmissionPattern;
  MatchResult matchInlinable(Value value) override {
    if (value.isa<BlockArgument>() &&
        value.getParentRegion()->getParentOfType<FuncOp>())
      return Precedence::VAR;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto func = value.getParentRegion()->getParentOfType<FuncOp>();
    for (size_t i = 0, e = func.getNumArguments(); i < e; ++i) {
      if (func.getArgument(i) == value) {
        // FIXME: this is hacky and could be improved.
        p << "_funcArg" << i;
        return;
      }
    }
  }

  bool matchStatement(Operation *op) override {
    return isa<FuncOp>(op) && cast<FuncOp>(op).getFunctionType().getNumResults() <= 1;
  }

  void emitStatement(FuncOp func, EmissionPrinter &p) override {
    // Emit a newline at the start to ensure an empty line before the module for
    // better readability.
    p << "\n";
    if (func.getFunctionType().getNumResults() == 0)
      p << "void";
    else
      p.emitType(func.getFunctionType().getResult(0));

    p << " " << func.getSymName() << "(";

    for (size_t i = 0, e = func.getFunctionType().getNumInputs(); i < e; ++i) {
      if (i > 0)
        p << ", ";
      p.emitType(func.getFunctionType().getInput(i));
      p << " _funcArg" << i;
    }

    p << ")";

    if (func.isDeclaration()) {
      p << ";\n";
    } else {
      p << " ";
      p.emitRegion(func.getRegion());
    }
  }
};

/// Emit a systemc.thread operation by using the SC_THREAD macro.
class CallEmitter : public OpEmissionPattern<CallOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<CallOp>())
      return Precedence::FUNCTION_CALL;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    printCall(value.getDefiningOp<CallOp>(), p);
  }

  void emitStatement(CallOp op, EmissionPrinter &p) override {
    // If the call returns values, then it is treated as an expression rather than a statement.
    if (op.getNumResults() > 0)
      return;
    
    printCall(op, p);
    p << ";\n";
  }

private:
  void printCall(CallOp op, EmissionPrinter &p) {
    p << op.getCallee()
      << "(";
    bool first = true;
    for (Value arg : op.getOperands()) {
      if (!first)
        p << ", ";
      p.getInlinable(arg).emit();
      first = false;
    }
    p << ")";
  }
};

/// Emit a systemc.thread operation by using the SC_THREAD macro.
class CallIndirectEmitter : public OpEmissionPattern<CallIndirectOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<CallIndirectOp>())
      return Precedence::FUNCTION_CALL;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    printCall(value.getDefiningOp<CallIndirectOp>(), p);
  }

  void emitStatement(CallIndirectOp op, EmissionPrinter &p) override {
    // If the call returns values, then it is treated as an expression rather than a statement.
    if (op.getNumResults() > 0)
      return;
    
    printCall(op, p);
    p << ";\n";
  }

private:
  void printCall(CallIndirectOp op, EmissionPrinter &p) {
    p.getInlinable(op.getCallee()).emit();
    p << "(";
    bool first = true;
    for (Value arg : op.getCalleeOperands()) {
      if (!first)
        p << ", ";
      p.getInlinable(arg).emit();
      first = false;
    }
    p << ")";
  }
};

/// Emit a systemc.thread operation by using the SC_THREAD macro.
struct ReturnEmitter : OpEmissionPattern<ReturnOp> {
  using OpEmissionPattern::OpEmissionPattern;

  bool matchStatement(Operation *op) override {
    return isa<ReturnOp>(op) && cast<ReturnOp>(op)->getNumOperands() <= 1;
  }

  void emitStatement(ReturnOp op, EmissionPrinter &p) override {
    p << "return";
    if (op->getNumOperands() == 1) {
      p << " ";
      p.getInlinable(op.getOperand(0)).emit();
    }
    p << ";\n";
  }
};

/// Emit a systemc.signal operation.
struct FuncConstantEmitter : OpEmissionPattern<func::ConstantOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<func::ConstantOp>())
      return Precedence::VAR;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    p << value.getDefiningOp<func::ConstantOp>().getValue();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Type emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit SystemC signal and port types according to the specification listed in
/// their ODS description.
struct FunctionTypeEmitter : public TypeEmissionPattern<FunctionType> {
  bool match(Type type) override {
    return type.isa<FunctionType>() && type.cast<FunctionType>().getNumResults() <= 1;
  }

  void emitType(FunctionType type, EmissionPrinter &p) override {
    p << "std::function<";
    if (type.getNumResults() == 0)
      p << "void";
    else
     p.emitType(type.getResult(0));
    p << "(";
    bool first = true;
    for (Type ty : type.getInputs()) {
      if (!first)
        p << ", ";
      p.emitType(ty);
      first = false;
    }
    p << ")>";
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateFuncOpEmitters(
    OpEmissionPatternSet &patterns, MLIRContext *context) {
  patterns.add<FuncEmitter, FuncConstantEmitter, ReturnEmitter, CallEmitter, CallIndirectEmitter>(context);
}

void circt::ExportSystemC::populateFuncTypeEmitters(
    TypeEmissionPatternSet &patterns) {
  patterns.add<FunctionTypeEmitter>();
}
