//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for CIRCT LEC transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_LEC_PASSES_H
#define CIRCT_TOOLS_CIRCT_LEC_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace mlir {
class Operation;
} // namespace mlir

namespace circt {

struct ConstructLECOptions {
  std::string firstModule;
  std::string secondModule;
  bool insertMainFunc = false;

  std::function<LogicalResult(mlir::Operation *, mlir::Operation *)>
      areTriviallyNotEquivalent;
};

std::unique_ptr<mlir::Pass> createConstructLEC();
std::unique_ptr<mlir::Pass>
createConstructLEC(const ConstructLECOptions &options);

/// Generate the code for registering passes.
// #define GEN_PASS_DECL_CONSTRUCTLEC
#define GEN_PASS_REGISTRATION
#include "circt/Tools/circt-lec/Passes.h.inc"

} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_LEC_PASSES_H
