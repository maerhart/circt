//===- InteropOpInterfaces.h - Declare interop op interfaces ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for dialect interoperability.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_INTEROPOPINTERFACES_H
#define CIRCT_DIALECT_HW_INTEROPOPINTERFACES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

namespace circt {
namespace hw {

using InteropAllocFunc =
    std::function<Optional<SmallVector<Value>>(OpBuilder &)>;
using InteropUpdateFunc = std::function<SmallVector<Value>(
    OpBuilder &, ArrayRef<Value>, ArrayRef<Value>)>;

enum class InteropMechanism { CFFI, CPP };

// NOLINTNEXTLINE(readability-identifier-naming)
inline llvm::hash_code hash_value(const InteropMechanism &x) {
  return llvm::hash_value(static_cast<unsigned>(x));
}

} // namespace hw
} // namespace circt

namespace llvm {

template <>
struct DenseMapInfo<circt::hw::InteropMechanism> {
  using Mechanism = circt::hw::InteropMechanism;
  static inline Mechanism getEmptyKey() { return Mechanism(-1); }
  static inline Mechanism getTombstoneKey() { return Mechanism(-2); }
  static unsigned getHashValue(const Mechanism &x) {
    return circt::hw::hash_value(x);
  }
  static bool isEqual(const Mechanism &lhs, const Mechanism &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

#include "circt/Dialect/HW/InteropOpInterfaces.h.inc"

#endif // CIRCT_DIALECT_HW_INTEROPOPINTERFACES_H
