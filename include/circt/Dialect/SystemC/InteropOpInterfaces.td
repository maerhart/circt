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

#ifndef CIRCT_DIALECT_SYSTEMC_INTEROPOPINTERFACES
#define CIRCT_DIALECT_SYSTEMC_INTEROPOPINTERFACES

include "mlir/IR/OpBase.td"

def ModuleProceduralInteropOpInterface : OpInterface<"ModuleProceduralInteropOpInterface"> {
  let cppNamespace = "circt::systemc";
  let description = [{
  }];

  let methods = [
    InterfaceMethod<"Get the supported interop mechanisms",
    "::llvm::SmallVector<::circt::systemc::InteropMechanism>", "getInteropSupport", (ins)>,

    InterfaceMethod<"Get the state builder",
    "::mlir::OpBuilder", "getStateBuilder", (ins "::circt::systemc::InteropMechanism":$interopType)>,

    InterfaceMethod<"Get the state init builder",
    "::mlir::OpBuilder", "getStateInitBuilder", (ins "::circt::systemc::InteropMechanism":$interopType)>,

    InterfaceMethod<"Get the state update builder",
    "::mlir::OpBuilder", "getStateUpdateBuilder", (ins "::mlir::Operation*":$interopOp, "::circt::systemc::InteropMechanism":$interopType)>,

    InterfaceMethod<"Get the state dealloc builder",
    "::mlir::OpBuilder", "getStateDeallocBuilder", (ins "::circt::systemc::InteropMechanism":$interopType)>
  ];
}

def InstanceProceduralInteropOpInterface : OpInterface<"InstanceProceduralInteropOpInterface"> {
  let cppNamespace = "circt::systemc";
  let description = [{
  }];

  let methods = [
    InterfaceMethod<"Get the supported interop mechanisms",
    "::llvm::SmallVector<::circt::systemc::InteropMechanism>", "getInteropSupport", (ins)>,

    InterfaceMethod<"Get the state builder",
    "::llvm::SmallVector<::mlir::Value>", "buildState", (ins "::mlir::OpBuilder&":$builder)>,

    InterfaceMethod<"Get the state init builder",
    "void", "buildStateInit", (ins "::mlir::OpBuilder&":$builder, "::llvm::ArrayRef<::mlir::Value>":$state)>,

    InterfaceMethod<"Get the state update builder",
    "::llvm::SmallVector<::mlir::Value>", "buildStateUpdate", (ins "::mlir::OpBuilder&":$builder, "::llvm::ArrayRef<::mlir::Value>":$state)>,

    InterfaceMethod<"Get the state dealloc builder",
    "void", "buildStateDealloc", (ins "::mlir::OpBuilder&":$builder, "::llvm::ArrayRef<::mlir::Value>":$state)>
  ];
}

#endif // CIRCT_DIALECT_SYSTEMC_INTEROPOPINTERFACES
