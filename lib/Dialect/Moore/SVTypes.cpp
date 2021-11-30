//===- SVTypes.cpp - Moore SystemVerilog types -------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemVerilog type system for the Moore dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/SVTypes.h"
#include "circt/Dialect/Moore/MooreDialect.h"

#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace moore;

llvm::hash_code hash_value(const moore::TypeRange &range) {
  return llvm::hash_combine(range.size, range.dir, range.offset);
}

//===----------------------------------------------------------------------===//
// Packed Core
//===----------------------------------------------------------------------===//

namespace circt {
namespace moore {
namespace detail {

struct IntTypeStorage : TypeStorage {
  using KeyTy = int;
  IntTypeStorage(const KeyTy &key) : value(key) {}
  bool operator==(const KeyTy &key) const { return key == value; }
  static IntTypeStorage *construct(TypeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<IntTypeStorage>()) IntTypeStorage(key);
  }
  KeyTy value;
};

} // namespace detail
} // namespace moore
} // namespace circt

IntVecPackedCore IntVecPackedCore::get(MLIRContext *context, IntVecType type) {
  return Base::get(context, type);
}

IntAtomPackedCore IntAtomPackedCore::get(MLIRContext *context,
                                         IntAtomType type) {
  return Base::get(context, type);
}

IntVecPackedCore::operator IntVecType() const {
  return static_cast<IntVecType>(getImpl()->value);
}

IntAtomPackedCore::operator IntAtomType() const {
  return static_cast<IntAtomType>(getImpl()->value);
}

//===----------------------------------------------------------------------===//
// Packed Dimension
//===----------------------------------------------------------------------===//

namespace circt {
namespace moore {
namespace detail {
struct TypeRangeStorage : TypeStorage {
  using KeyTy = TypeRange;
  TypeRangeStorage(const KeyTy &key) : value(key) {}
  bool operator==(const KeyTy &key) const { return key == value; }
  static llvm::hash_code hashKey(const KeyTy &key) { return hash_value(key); }
  static TypeRangeStorage *construct(TypeStorageAllocator &allocator,
                                     KeyTy key) {
    return new (allocator.allocate<TypeRangeStorage>()) TypeRangeStorage(key);
  }
  KeyTy value;
};
} // namespace detail
} // namespace moore
} // namespace circt

RangePackedDim::operator const TypeRange &() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// Packed Type
//===----------------------------------------------------------------------===//

namespace circt {
namespace moore {
namespace detail {
struct PackedTypeStorage : TypeStorage {
  using KeyTy = std::tuple<PackedCore, TypeSign, bool, ArrayRef<PackedDim>>;

  PackedTypeStorage(PackedCore core, TypeSign sign, bool explicitSign,
                    ArrayRef<PackedDim> dims)
      : core(core), sign(sign), explicitSign(explicitSign),
        dims(dims.begin(), dims.end()) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(core, sign, explicitSign, dims);
  }

  static PackedTypeStorage *construct(TypeStorageAllocator &allocator,
                                      KeyTy key) {
    return new (allocator.allocate<PackedTypeStorage>()) PackedTypeStorage(
        std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key));
  }

  /// The core packed type.
  PackedCore core;
  /// The type sign.
  TypeSign sign;
  /// Whether the sign was explicitly mentioned in the source code.
  bool explicitSign;
  /// The packed dimensions.
  SmallVector<PackedDim, 0> dims;

  /// This type with one level of name/reference resolved.
  PackedType resolved;
  /// This type with all names/references recursively resolved.
  PackedType fullyResolved;
};
} // namespace detail
} // namespace moore
} // namespace circt

LogicalResult printPackedType(PackedType type, DialectAsmPrinter &p) {
  p << "packed<";

  p << ">";
}
