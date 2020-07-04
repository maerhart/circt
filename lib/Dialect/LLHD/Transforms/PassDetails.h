//===- PassDetails.h - LLHD pass class details ------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LLHD_TRANSFORMS_PASSDETAILS_H
#define DIALECT_LLHD_TRANSFORMS_PASSDETAILS_H

#include "mlir/Pass/Pass.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"

namespace mlir {
namespace llhd {

#define GEN_PASS_CLASSES
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"

} // namespace llhd
} // namespace mlir

#endif // DIALECT_LLHD_TRANSFORMS_PASSDETAILS_H
