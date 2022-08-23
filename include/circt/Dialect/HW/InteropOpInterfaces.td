//===- InteropOpInterfaces.td - Interop Interfaces ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes interop operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_INTEROPOPINTERFACES
#define CIRCT_DIALECT_HW_INTEROPOPINTERFACES

include "mlir/IR/OpBase.td"

def ProceduralContainerInteropOpInterface :
    OpInterface<"ProceduralContainerInteropOpInterface"> {
  let cppNamespace = "circt::hw";
  let description = [{
  }];

  let methods = [
    InterfaceMethod<"Get the supported interop mechanisms",
      "::llvm::SmallVector<::circt::systemc::InteropMechanism>",
      "getInteropSupport", (ins)>,

    InterfaceMethod<"Get the interop builder",
      "::llvm::SmallVector<::mlir::Value>", "wrapInterop",
      (ins "::llvm::ArrayRef<::mlir::Type>":$state,
          "::llvm::ArrayRef<::mlir::Value>":$input,
          "::circt::systemc::InteropAllocFunc&":$allocator,
          "::circt::systemc::InteropUpdateFunc&":$updater,
          "::circt::systemc::InteropMechanism":$mechanism)>,
  ];
}

def ProceduralInstanceInteropOpInterface :
    OpInterface<"ProceduralInstanceInteropOpInterface"> {
  let cppNamespace = "circt::hw";
  let description = [{
  }];

  let methods = [
    InterfaceMethod<"Get the supported interop mechanisms",
      "::llvm::SmallVector<::circt::systemc::InteropMechanism>",
      "getInteropSupport", (ins)>,

    InterfaceMethod<"Get the state builder",
      "::llvm::SmallVector<::mlir::Type>", "getRequiredState",
      (ins "::circt::systemc::InteropMechanism":$mechanism)>,

    InterfaceMethod<"Get the state builder",
      "::llvm::Optional<::llvm::SmallVector<::mlir::Value>>", "allocState",
      (ins "::mlir::OpBuilder&":$builder,
           "::circt::systemc::InteropMechanism":$mechanism)>,

    InterfaceMethod<"Get the state builder",
      "::llvm::SmallVector<::mlir::Value>", "updateState",
      (ins "::mlir::OpBuilder&":$builder,
          "::llvm::ArrayRef<::mlir::Value>":$state,
          "::llvm::ArrayRef<::mlir::Value>":$input,
          "::circt::systemc::InteropMechanism":$mechanism)>,
  ];
}

#endif // CIRCT_DIALECT_HW_INTEROPOPINTERFACES
