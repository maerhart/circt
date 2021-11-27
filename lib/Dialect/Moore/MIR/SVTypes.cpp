//===- MIRTypes.cpp - Implement the Moore MIR types -----------------------===//
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

#include "circt/Dialect/Moore/MIR/SVTypes.h"
#include "circt/Dialect/Moore/MIR/MIRDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace moore;

//===----------------------------------------------------------------------===//
// Packed Type
//===----------------------------------------------------------------------===//

namespace circt {
namespace firrtl {
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
} // namespace firrtl
} // namespace circt
