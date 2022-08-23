//===- InteropOpInterfaceImpl.cpp - Implement interop for SystemC ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the hw::InteropOpInterface for SystemC ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/Transforms/InteropOpInterfaceImpl.h"
#include "circt/Dialect/HW/InteropOpInterfaces.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace circt::hw;
using namespace circt::systemc;

namespace {

struct SCModuleOpContainerInterface
    : ProceduralContainerInteropOpInterface::ExternalModel<
          SCModuleOpContainerInterface, systemc::SCModuleOp> {
  SmallVector<InteropMechanism> getInteropSupport(Operation *op) const {
    return SmallVector<InteropMechanism>({InteropMechanism::CPP});
  }

  SmallVector<Value> wrapInterop(Operation *op, ArrayRef<Type> state,
                                 ArrayRef<Value> input,
                                 InteropAllocFunc &allocator,
                                 InteropUpdateFunc &updater,
                                 InteropMechanism mechanism) const {
    SCModuleOp module = cast<SCModuleOp>(op);

    if (state.empty())
      return ValueRange{};

    auto stateBuilder = OpBuilder::atBlockBegin(module.getBodyBlock());
    auto initBuilder =
        OpBuilder::atBlockEnd(&module.getOrCreateCtor().getBody().front());
    auto deallocBuilder = OpBuilder::atBlockEnd(
        &module.getOrCreateDestructor().getBody().front());

    SmallVector<Value> variables;
    for (Type stateTy : state) {
      // Build state
      auto variable =
          stateBuilder
              .create<VariableOp>(
                  module.getLoc(), stateTy,
                  StringAttr::get(stateBuilder.getContext(), "vName"), Value())
              .getResult();

      // Build state alloc
      auto newOp =
          initBuilder.create<NewOp>(module.getLoc(), stateTy, ValueRange{});
      initBuilder.create<AssignOp>(module.getLoc(), variable,
                                   newOp.getResult());

      // Build state dealloc
      deallocBuilder.create<DeleteOp>(module.getLoc(), variable);

      variables.push_back(variable);
    }
    auto initValues = allocator(initBuilder);
    if (initValues.has_value()) {
      for (size_t i = 0; i < variables.size(); ++i) {
        initBuilder.create<AssignOp>(module.getLoc(), variables[i],
                                     initValues.value()[i]);
      }
    }

    return updater(variables, input);
  }
};

//===----------------------------------------------------------------------===//
// ModelVerilatedOp
//===----------------------------------------------------------------------===//

struct VerilatedOpInstanceInterface
    : ProceduralInstanceInteropOpInterface::ExternalModel<
          VerilatedOpInstanceInterface, systemc::ModelVerilatedOp> {
  SmallVector<InteropMechanism> getInteropSupport(Operation *op) const {
    return SmallVector<InteropMechanism>({InteropMechanism::CPP});
  }

  SmallVector<Type> getRequiredState(Operation *op,
                                     InteropMechanism mechanism) const {
    std::string tn = "V";
    tn += cast<ModelVerilatedOp>(op).getModuleName();
    auto ptrType =
        emitc::PointerType::get(emitc::OpaqueType::get(op->getContext(), tn));
    return {ptrType};
  }

  Optional<SmallVector<Value>> allocState(Operation *op, OpBuilder &builder,
                                          InteropMechanism mechanism) const {
    return {};
  }

  SmallVector<Value> updateState(Operation *op, OpBuilder &builder,
                                 ArrayRef<Value> state, ArrayRef<Value> input,
                                 InteropMechanism mechanism) const {
    ModelVerilatedOp verilatedOp = cast<ModelVerilatedOp>(op);
    ModelVerilatedOp::Adaptor adaptor(input);
    Location loc = verilatedOp->getLoc();

    std::string tn = "V";
    tn += verilatedOp.getModuleName();

    // Replace external HW module with include
    auto *extModule = SymbolTable::lookupNearestSymbolFrom(
        verilatedOp, verilatedOp.getModuleNameAttr());
    OpBuilder includeBuilder(extModule);
    includeBuilder.create<emitc::IncludeOp>(loc, tn + ".h", false);
    extModule->erase();

    // Build state update
    for (size_t i = 0; i < verilatedOp.getInputs().size(); ++i) {
      Value input = verilatedOp.getInputs()[i];
      auto member =
          builder
              .create<MemberAccessOp>(
                  loc, input.getType(), state[0],
                  verilatedOp.getInputNames()[i].cast<StringAttr>().getValue(),
                  true)
              .getResult();
      builder.create<AssignOp>(loc, member, input);
    }

    auto evalFunc = builder.create<MemberAccessOp>(
        loc, FunctionType::get(builder.getContext(), {}, {}), state[0], "eval",
        true);
    builder.create<func::CallIndirectOp>(loc, evalFunc.getResult());

    SmallVector<Value> results;
    for (size_t i = 0; i < verilatedOp.getNumResults(); ++i) {
      results.push_back(
          builder
              .create<MemberAccessOp>(
                  loc, verilatedOp.getResults()[i].getType(), state[0],
                  verilatedOp.getResultNames()[i].cast<StringAttr>().getValue(),
                  true)
              .getResult());
    }

    return results;
  }
};

} // namespace

void circt::systemc::registerInteropOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, systemc::SystemCDialect *dialect) {
        SCModuleOp::attachInterface<SCModuleOpContainerInterface>(*ctx);
        ModelVerilatedOp::attachInterface<VerilatedOpInstanceInterface>(*ctx);
      });
}
