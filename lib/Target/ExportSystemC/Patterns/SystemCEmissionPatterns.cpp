//===- SystemCEmissionPatterns.cpp - SystemC Dialect Emission Patterns ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the systemc dialect.
//
//===----------------------------------------------------------------------===//

#include "SystemCEmissionPatterns.h"
#include "../EmissionPattern.h"
#include "../EmissionPrinter.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::systemc;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {
struct SCModuleEmitter : OpEmissionPattern<SCModuleOp> {
  using OpEmissionPattern::OpEmissionPattern;
  MatchResult matchInlinable(Value value, EmissionConfig &config) override {
    if (value.isa<BlockArgument>() &&
        value.getParentRegion()->getParentOfType<SCModuleOp>())
      return MatchResult(Precedence::VAR);
    return MatchResult();
  }

  void emitInlined(Value value, EmissionConfig &config,
                   EmissionPrinter &p) override {
    auto module = value.getParentRegion()->getParentOfType<SCModuleOp>();
    for (size_t i = 0, e = module.getNumArguments(); i < e; ++i) {
      if (module.getArgument(i) == value) {
        p << module.getPortNames()[i].cast<StringAttr>().getValue();
        return;
      }
    }
  }

  void emitStatement(SCModuleOp module, EmissionConfig &config,
                     EmissionPrinter &p) override {
    p << "\nSC_MODULE(" << module.getModuleName() << ") ";
    auto scope = p.getOstream().scope("{\n", "};\n");
    for (size_t i = 0, e = module.getNumArguments(); i < e; ++i) {
      p.emitType(module.getArgument(i).getType());

      auto portName = module.getPortNames()[i].cast<StringAttr>().getValue();
      p << " " << portName << ";\n";
    }

    p.emitRegion(module.getRegion(), scope);
  }
};

struct SignalWriteEmitter : OpEmissionPattern<SignalWriteOp> {
  using OpEmissionPattern::OpEmissionPattern;
  void emitStatement(SignalWriteOp op, EmissionConfig &config,
                     EmissionPrinter &p) override {
    p.getInlinable(op.getDest()).emit();

    bool implicitWrite = config.get(implicitReadWriteFlag);

    if (implicitWrite)
      p << " = ";
    else
      p << ".write(";

    p.getInlinable(op.getSrc()).emit();

    if (!implicitWrite)
      p << ")";

    p << ";\n";
  }
};

struct SignalReadEmitter : OpEmissionPattern<SignalReadOp> {
  using OpEmissionPattern::OpEmissionPattern;
  MatchResult matchInlinable(Value value, EmissionConfig &config) override {
    if (llvm::isa_and_nonnull<SignalReadOp>(value.getDefiningOp()))
      return MatchResult(Precedence::VAR);
    return MatchResult();
  }
  void emitInlined(Value value, EmissionConfig &config,
                   EmissionPrinter &p) override {
    p.getInlinable(value.getDefiningOp<SignalReadOp>().getInput()).emit();
    if (!config.get(implicitReadWriteFlag))
      p << ".read()";
  }
};

struct CtorEmitter : OpEmissionPattern<CtorOp> {
  using OpEmissionPattern::OpEmissionPattern;
  void emitStatement(CtorOp op, EmissionConfig &config,
                     EmissionPrinter &p) override {
    p << "\nSC_CTOR(" << op->getParentOfType<SCModuleOp>().getModuleName()
      << ") ";
    p.emitRegion(op.getBody());
  }
};

struct SCFuncEmitter : OpEmissionPattern<SCFuncOp> {
  using OpEmissionPattern::OpEmissionPattern;
  MatchResult matchInlinable(Value value, EmissionConfig &config) override {
    if (llvm::isa_and_nonnull<SCFuncOp>(value.getDefiningOp()))
      return MatchResult(Precedence::VAR);
    return MatchResult();
  }
  void emitInlined(Value value, EmissionConfig &config,
                   EmissionPrinter &p) override {
    p << value.getDefiningOp<SCFuncOp>().getName();
  }

  void emitStatement(SCFuncOp op, EmissionConfig &config,
                     EmissionPrinter &p) override {
    p << "\nvoid " << op.getName() << "() ";
    p.emitRegion(op.getBody());
  }
};

struct MethodEmitter : OpEmissionPattern<MethodOp> {
  using OpEmissionPattern::OpEmissionPattern;
  void emitStatement(MethodOp op, EmissionConfig &config,
                     EmissionPrinter &p) override {
    p << "SC_METHOD(";
    p.getInlinable(op.getFuncHandle()).emit();
    p << ");\n";
  }
};

struct ThreadEmitter : OpEmissionPattern<ThreadOp> {
  using OpEmissionPattern::OpEmissionPattern;
  void emitStatement(ThreadOp op, EmissionConfig &config,
                     EmissionPrinter &p) override {
    p << "SC_THREAD(";
    p.getInlinable(op.getFuncHandle()).emit();
    p << ");\n";
  }
};

struct SignalEmitter : OpEmissionPattern<SignalOp> {
  using OpEmissionPattern::OpEmissionPattern;
  MatchResult matchInlinable(Value value, EmissionConfig &config) override {
    if (llvm::isa_and_nonnull<SignalOp>(value.getDefiningOp()))
      return MatchResult(Precedence::VAR);
    return MatchResult();
  }
  void emitInlined(Value value, EmissionConfig &config,
                   EmissionPrinter &p) override {
    p << value.getDefiningOp<SignalOp>().getName();
  }

  void emitStatement(SignalOp op, EmissionConfig &config,
                     EmissionPrinter &p) override {
    p.emitType(op.getSignal().getType());
    p << " " << op.getName() << ";\n";
  }
};

struct BuiltinModuleEmitter : OpEmissionPattern<ModuleOp> {
  using OpEmissionPattern::OpEmissionPattern;
  void emitStatement(ModuleOp op, EmissionConfig &config,
                     EmissionPrinter &p) override {
    auto scope = p.getOstream().scope("", "", false);
    p.emitRegion(op.getRegion(), scope);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Type emission patterns.
//===----------------------------------------------------------------------===//

namespace {
template <typename Ty, const char S[]>
struct SignalTypeEmitter : public TypeEmissionPattern<Ty> {
  void emitType(Ty type, EmissionConfig &config, EmissionPrinter &p) override {
    p << "sc_core::" << S << "<";
    p.emitType(type.getBaseType());
    p << ">";
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateSystemCOpEmitters(
    OpEmissionPatternSet &patterns, MLIRContext *context) {
  patterns.add<BuiltinModuleEmitter, SCModuleEmitter, SignalWriteEmitter,
               SignalReadEmitter, CtorEmitter, SCFuncEmitter, MethodEmitter,
               ThreadEmitter, SignalEmitter>(context);
}

void circt::ExportSystemC::populateSystemCTypeEmitters(
    TypeEmissionPatternSet &patterns) {
  static constexpr const char in[] = "sc_in";
  static constexpr const char out[] = "sc_in";
  static constexpr const char inout[] = "sc_inout";
  static constexpr const char signal[] = "sc_signal";
  patterns
      .add<SignalTypeEmitter<InputType, in>, SignalTypeEmitter<OutputType, out>,
           SignalTypeEmitter<InOutType, inout>,
           SignalTypeEmitter<SignalType, signal>>();
}
