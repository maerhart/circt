//===-- Passes.td - HW pass definition file ------0000------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_PASSES_TD
#define CIRCT_DIALECT_HW_PASSES_TD

include "mlir/Pass/PassBase.td"

def PrintInstanceGraph : Pass<"hw-print-instance-graph", "mlir::ModuleOp"> {
  let summary = "Print a DOT graph of the module hierarchy.";
  let constructor =  "circt::hw::createPrintInstanceGraphPass()";
}

def HWSpecialize : Pass<"hw-specialize", "mlir::ModuleOp"> {
  let summary = "Specializes instances of parametric hw.modules";
  let constructor = "circt::hw::createHWSpecializePass()";
  let description = [{
    Any `hw.instance` operation instantiating a parametric `hw.module` will
    trigger a specialization procedure which resolves all parametric types and
    values within the module based on the set of provided parameters to the
    `hw.instance` operation. This specialized module is created as a new
    `hw.module` and the referring `hw.instance` operation is rewritten to
    instantiate the newly specialized module.
  }];
}

//===----------------------------------------------------------------------===//
// LowerInterop
//===----------------------------------------------------------------------===//

def LowerInterop : Pass<"lower-interop", "mlir::ModuleOp"> {
  let summary = "";
  let description = [{
  }];
  let constructor = "circt::hw::createLowerInteropPass()";
  let dependentDialects = ["systemc::SystemCDialect", "mlir::emitc::EmitCDialect", "mlir::func::FuncDialect"];
}

def WrappedModuleNameOption : Option<"wrappedModuleName", "wrappedModule",
                                     "std::string", "",
                                     "Name of the module to be wrapped">;

def CreateWrapper : Pass<"create-wrapper", "mlir::ModuleOp"> {
  let summary = "";
  let description = [{
  }];
  let options = [WrappedModuleNameOption];
  let constructor = "circt::hw::createWrapperPass()";
  let dependentDialects = ["hw::HWDialect"];
}

#endif // CIRCT_DIALECT_HW_PASSES_TD
