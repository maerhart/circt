//===-- Passes.td - BMC Transforms pass definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for BMC transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_BMC_PASSES_TD
#define CIRCT_TOOLS_CIRCT_BMC_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LowerToBMC : Pass<"lower-to-bmc", "::mlir::ModuleOp"> {
  let summary = "Lower CIRCTs core dialects to a BMC problem statement";
  let description = [{
  }];
  let options = [
    Option<"topModule", "top-module", "std::string",
           /*default=*/"",
           "Name of the top module to verify.">,
    Option<"bound", "bound", "unsigned",
           /*default=*/"",
           "Cycle bound.">,
  ];

  let dependentDialects = [
    "smt::SMTDialect", "mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect"
  ];
}

#endif // CIRCT_TOOLS_CIRCT_BMC_PASSES_TD

