//===- RTGAttributes.h - RTG dialect attributes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGATTRIBUTES_H
#define CIRCT_DIALECT_RTG_IR_RTGATTRIBUTES_H

#include "circt/Dialect/RTG/IR/RTGAttrInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyAttrInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APSInt.h"

namespace circt {
namespace rtg {

/// IPInt (Infinite Precision Integer) is a wrapper around APSInt that ensures
/// values are normalized to the minimum bitwidth required to represent them.
/// This class is used for rtg.int attributes to support arbitrary precision
/// integer arithmetic with automatic overflow prevention.
class IPInt {
public:
  /// Construct an IPInt from an APSInt. The value will be normalized.
  explicit IPInt(llvm::APSInt value);

  /// Construct an IPInt from an int64_t value.
  explicit IPInt(int64_t value);

  /// Get the underlying APSInt value.
  const llvm::APSInt &getValue() const { return value; }

  /// Arithmetic operations that return normalized results
  IPInt add(const IPInt &rhs) const;
  IPInt sub(const IPInt &rhs) const;
  IPInt mul(const IPInt &rhs) const;
  IPInt sdiv(const IPInt &rhs) const;
  IPInt smod(const IPInt &rhs) const;
  IPInt pow(const IPInt &exponent) const;

  /// Bitwise operations
  IPInt and_(const IPInt &rhs) const;
  IPInt or_(const IPInt &rhs) const;
  IPInt xor_(const IPInt &rhs) const;
  IPInt shl(const IPInt &rhs) const;
  IPInt ashr(const IPInt &rhs) const;

  /// Comparison operations
  bool eq(const IPInt &rhs) const;
  bool ne(const IPInt &rhs) const;
  bool slt(const IPInt &rhs) const;
  bool sle(const IPInt &rhs) const;
  bool sgt(const IPInt &rhs) const;
  bool sge(const IPInt &rhs) const;

  /// Hash for use in DenseMap
  friend llvm::hash_code hash_value(const IPInt &val);

  /// Equality comparison
  bool operator==(const IPInt &rhs) const { return eq(rhs); }
  bool operator!=(const IPInt &rhs) const { return ne(rhs); }

private:
  /// Normalize the value to the minimum bitwidth required to represent it.
  static llvm::APSInt normalize(llvm::APSInt value);

  /// Extend both operands to a common bitwidth sufficient for the operation.
  static std::pair<llvm::APSInt, llvm::APSInt>
  extendToCommonWidth(const llvm::APSInt &lhs, const llvm::APSInt &rhs);

  llvm::APSInt value;
};

namespace detail {

struct IntAttrStorage;

} // namespace detail
} // namespace rtg
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/RTG/IR/RTGAttributes.h.inc"

#endif // CIRCT_DIALECT_RTG_IR_RTGATTRIBUTES_H
