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

    InterfaceMethod<"Get the interop builder",
    "std::unique_ptr<::circt::systemc::InteropBuilder>", "getInteropBuilder", (ins "::mlir::Operation*":$interopOp, "::circt::systemc::InteropMechanism":$interopType)>,
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
    "::llvm::SmallVector<::mlir::Value>", "buildInterop", (ins "::circt::systemc::InteropBuilder&":$builder)>,
  ];
}

#endif // CIRCT_DIALECT_SYSTEMC_INTEROPOPINTERFACES
