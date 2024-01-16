//===- PassDetail.h - BMC Transforms Pass class details ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef TOOLS_CIRCT_BMC_PASSDETAIL_H
#define TOOLS_CIRCT_BMC_PASSDETAIL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace func {
class FuncDialect;
} // namespace func

namespace LLVM {
class LLVMDialect;
} // namespace LLVM
} // end namespace mlir

namespace circt {
namespace comb {
class CombDialect;
} // namespace comb

namespace hw {
class HWDialect;
} // namespace hw

namespace smt {
class SMTDialect;
} // namespace smt

} // namespace circt

#endif // TOOLS_CIRCT_BMC_PASSDETAIL_H
