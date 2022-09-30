//===- HWToSystemC.cpp - HW To SystemC Conversion Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HW to SystemC Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/InteropOpInterfaces.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace systemc;

//===----------------------------------------------------------------------===//
// Interop Bridges
//===----------------------------------------------------------------------===//

///
struct InteropBridgeBase {
  virtual SmallVector<Type>
  bridgeRequiredState(ArrayRef<Type> requiredState) = 0;
  virtual InteropAllocFunc bridgeAlloc(const InteropAllocFunc &allocFunc,
                                       ArrayRef<Type> state) = 0;
  virtual InteropUpdateFunc
  bridgeUpdate(const InteropUpdateFunc &updateFunc) = 0;
  virtual ~InteropBridgeBase() = default;
};

struct IdentityInteropBridge : InteropBridgeBase {
  SmallVector<Type> bridgeRequiredState(ArrayRef<Type> requiredState) override {
    return SmallVector<Type>(requiredState);
  }

  InteropAllocFunc bridgeAlloc(const InteropAllocFunc &allocFunc,
                               ArrayRef<Type> requiredState) override {
    return allocFunc;
  }

  InteropUpdateFunc bridgeUpdate(const InteropUpdateFunc &updateFunc) override {
    return updateFunc;
  }
};

/// The instance accepts only C++ and the container only C code
/*

*/
struct CPPToCInteropBridge : InteropBridgeBase {
  SmallVector<Type> bridgeRequiredState(ArrayRef<Type> requiredState) override {
    // We should ask for an uninitialized opaque pointer here, I guess and do
    // the alloc and init ourselves in this bridge.
    return SmallVector<Type>(requiredState);
  }

  InteropAllocFunc bridgeAlloc(const InteropAllocFunc &allocFunc,
                               ArrayRef<Type> requiredState) override {
    // insert a call to an extern C function that does the alloc in C++ code and
    // is compiled separately
    return [&](OpBuilder &builder) -> Optional<SmallVector<Value>> {
      Location loc = builder.getUnknownLoc();

      Operation *parent = builder.getBlock()->getParentOp();
      if (!isa<ModuleOp>(parent))
        parent = parent->getParentOfType<ModuleOp>();

      OpBuilder globalBuilder =
          OpBuilder::atBlockBegin(cast<ModuleOp>(parent).getBody());
      // auto externOp = globalBuilder.create<ExternOp>(loc, true);
      // globalBuilder.setInsertionPointToStart(externOp.getBodyBlock());
      auto funcOp = globalBuilder.create<func::FuncOp>(
          loc, "allocWrapper",
          builder.getFunctionType({bridgeRequiredState(requiredState)}, {}));
      funcOp->setAttr("externC", UnitAttr::get(builder.getContext()));
      funcOp.getBody().push_back(new Block);
      globalBuilder.setInsertionPointToStart(&funcOp.getBody().front());

      auto innerValues = allocFunc(globalBuilder);
      // TODO: do type/value conversion from C++ to C if necessary.
      if (innerValues.has_value()) {
        globalBuilder.create<func::ReturnOp>(loc, innerValues.value());
        auto callOp = builder.create<func::CallOp>(loc, funcOp);
        return SmallVector<Value>(callOp->getResults());
      }

      funcOp->erase();
      return {};
    };
  }

  InteropUpdateFunc bridgeUpdate(const InteropUpdateFunc &updateFunc) override {
    // insert a call to an extern C function that does the update in C++ code
    // and is compiled separately
    return [&](OpBuilder &builder, ArrayRef<Value> state,
               ArrayRef<Value> input) {
      Location loc = builder.getUnknownLoc();

      Operation *parent = builder.getBlock()->getParentOp();
      if (!isa<ModuleOp>(parent))
        parent = parent->getParentOfType<ModuleOp>();

      OpBuilder globalBuilder =
          OpBuilder::atBlockBegin(cast<ModuleOp>(parent).getBody());
      // auto externOp = globalBuilder.create<ExternOp>(loc, true);
      // globalBuilder.setInsertionPointToStart(externOp.getBodyBlock());
      SmallVector<Value> arguments;
      arguments.append(state.begin(), state.end());
      arguments.append(input.begin(), input.end());
      SmallVector<Type> argTypes;
      for (auto val : arguments)
        argTypes.push_back(val.getType());
      auto funcOp = globalBuilder.create<func::FuncOp>(
          loc, "updateWrapper", builder.getFunctionType(argTypes, {}));
      funcOp->setAttr("externC", UnitAttr::get(builder.getContext()));
      auto *funcBlock = new Block;
      funcBlock->addArguments(argTypes,
                              SmallVector<Location>(argTypes.size(), loc));
      globalBuilder.setInsertionPointToStart(funcBlock);

      SmallVector<Value> innerState(
          funcBlock->getArguments().take_front(state.size()));
      SmallVector<Value> innerInput(
          funcBlock->getArguments().take_back(input.size()));
      auto innerValues = updateFunc(globalBuilder, innerState, innerInput);
      SmallVector<Type> resultTypes;
      for (auto val : innerValues)
        resultTypes.push_back(val.getType());
      // TODO: do type/value conversion from C++ to C if necessary.
      globalBuilder.create<func::ReturnOp>(loc, innerValues);
      auto inputs = funcOp.getFunctionType().getInputs();
      funcOp.setFunctionTypeAttr(
          TypeAttr::get(globalBuilder.getFunctionType(inputs, resultTypes)));
      funcOp.getBody().push_back(funcBlock);

      auto callOp = builder.create<func::CallOp>(loc, funcOp, arguments);
      return SmallVector<Value>(callOp->getResults());
    };
  }

  static InteropMechanism getSourceMechanism() { return InteropMechanism::CPP; }

  static InteropMechanism getTargetMechanism() {
    return InteropMechanism::CFFI;
  }
};

class InteropBridgeCollection {
public:
  InteropBridgeBase *get(InteropMechanism instanceInterop,
                         InteropMechanism containerInterop) {
    if (instanceInterop == containerInterop)
      return &identityBridge;

    return bridges.lookup(std::make_pair(instanceInterop, containerInterop));
  }

  template <typename B>
  void addBridge() {
    bridges[std::make_pair(B::getSourceMechanism(), B::getTargetMechanism())] =
        new B();
  }

  ~InteropBridgeCollection() {
    for (auto entry : bridges) {
      delete entry.getSecond();
    }
  }

private:
  DenseMap<std::pair<InteropMechanism, InteropMechanism>, InteropBridgeBase *>
      bridges;
  IdentityInteropBridge identityBridge;
};

//===----------------------------------------------------------------------===//
// Lower Interop Pass
//===----------------------------------------------------------------------===//

namespace {
class LowerInteropPass : public LowerInteropBase<LowerInteropPass> {
public:
  explicit LowerInteropPass(const DialectRegistry &registry, StringRef cliName)
      : reg(registry), cliName(cliName) {}
  void runOnOperation() override;
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    // for (const auto &dialect : dialects) {
    //   registry.insert(dialect->getTypeID(), dialect->getNamespace(),
    //   static_cast<DialectAllocatorFunction>(([&](MLIRContext *ctx) {
    //     return ctx->getOrLoadDialect(dialect->getNamespace());
    //   })));
    // }
    reg.appendTo(registry);
    registry.insert<systemc::SystemCDialect>();
    registry.insert<mlir::emitc::EmitCDialect>();
    registry.insert<mlir::func::FuncDialect>();
    llvm::errs() << "In LowerInteropPass::getDependentDialects: ";
    llvm::interleaveComma(registry.getDialectNames(), llvm::errs());
    llvm::errs() << "\n";
  }
  StringRef getArgument() const override { return cliName; }

private:
  const DialectRegistry &reg;
  StringRef cliName;
};
} // namespace

/// Create a HW to SystemC dialects conversion pass.
// std::unique_ptr<Pass> circt::hw::createLowerInteropPass() {
//   return std::make_unique<LowerInteropPass>();
// }

/// Create a HW to SystemC dialects conversion pass.
std::unique_ptr<Pass>
circt::hw::createLowerInteropPass(const DialectRegistry &registry,
                                  StringRef cliName) {
  return std::make_unique<LowerInteropPass>(registry, cliName);
}
std::unique_ptr<Pass> circt::hw::createLowerInteropPass() {
  DialectRegistry registry;
  return createLowerInteropPass(registry, "lower-interop");
}

// void LowerInteropPass::getDependentDialects(::mlir::DialectRegistry
// &registry) const {
// }

/// This is the main entrypoint for the HW to SystemC conversion pass.
void LowerInteropPass::runOnOperation() {
  ModuleOp module = getOperation();
  InteropBridgeCollection bridges;
  bridges.addBridge<CPPToCInteropBridge>();

  WalkResult result =
      module->walk([&](ProceduralInstanceInteropOpInterface op) -> WalkResult {
        auto parent =
            op->getParentOfType<ProceduralContainerInteropOpInterface>();
        if (!parent) {
          op->emitError() << "No parent op accepting interop";
          return WalkResult::interrupt();
        }

        InteropMechanism interopType;
        bool intersect = false;
        InteropBridgeBase *bridge;
        for (auto interop : op.getInteropSupport()) {
          for (auto pi : parent.getInteropSupport()) {
            if (auto *b = bridges.get(interop, pi)) {
              interopType = interop;
              intersect = true;
              bridge = b;
            }
          }
        }

        if (!intersect)
          return WalkResult::interrupt();

        OpBuilder builder(op);
        InteropAllocFunc allocator =
            std::bind(&ProceduralInstanceInteropOpInterface::allocState, op,
                      std::placeholders::_1, interopType);
        InteropUpdateFunc updater =
            std::bind(&ProceduralInstanceInteropOpInterface::updateState, op,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, interopType);

        SmallVector<Value> inputs(op->getOperands());
        // auto newValues =
        //     parent.wrapInterop(builder, op.getRequiredState(interopType),
        //     inputs,
        //                        allocator, updater, interopType);

        auto newValues = parent.wrapInterop(
            builder,
            bridge->bridgeRequiredState(op.getRequiredState(interopType)),
            inputs,
            bridge->bridgeAlloc(allocator, op.getRequiredState(interopType)),
            bridge->bridgeUpdate(updater), interopType);

        op->replaceAllUsesWith(newValues);
        op->erase();

        return WalkResult::advance();
      });

  if (result.wasInterrupted())
    signalPassFailure();
}
