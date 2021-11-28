//===- SVTypes.h - Moore SystemVerilog types ---------------------*- C++-*-===//
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

#ifndef CIRCT_DIALECT_MOORE_SV_TYPES_H
#define CIRCT_DIALECT_MOORE_SV_TYPES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Types.h"

namespace circt {
namespace moore {

namespace detail {
struct IntTypeStorage;
struct StructCoreStorage;
struct EnumCoreStorage;
struct PackedTypeStorage;
struct TypeRangeStorage;
} // namespace detail

/// A common base class for all SystemVerilog types.
class SVType : public Type {
protected:
  using Type::Type;
};

/// The number of values each bit of a type can assume.
enum class TypeDomain {
  /// Two-valued types such as `bit` or `int`.
  TwoValued,
  /// Four-valued types such as `logic` or `integer`.
  FourValued,
};

/// Whether a type is signed or unsigned.
enum class TypeSign {
  /// A `signed` type.
  Signed,
  /// An `unsigned` type.
  Unsigned,
};

/// Which side is greater in a range `[a:b]`.
enum class TypeRangeDir {
  /// `a < b`
  Up,
  /// `a > b`
  Down,
};

/// The `[a:b]` part in a vector/array type such as `logic [a:b]`.
struct TypeRange {
  /// The total number of bits, given as `|a-b|+1`.
  uint32_t size;
  /// The direction of the vector, i.e. whether `a > b` or `a < b`.
  TypeRangeDir dir;
  /// The starting offset of the range.
  int32_t offset;

  explicit TypeRange(uint32_t size)
      : size(size), dir(TypeRangeDir::Down), offset(0) {}
  TypeRange(uint32_t size, TypeRangeDir dir, int32_t offset)
      : size(size), dir(dir), offset(offset) {}

  bool operator==(const TypeRange &other) const {
    return size == other.size && dir == other.dir && offset == other.offset;
  }
};

llvm::hash_code hash_value(const TypeRange &range);

/// An integer vector type.
///
/// These are the builtin single-bit integer types.
enum class IntVecType {
  /// A `bit`.
  Bit,
  /// A `logic`.
  Logic,
  /// A `reg`.
  Reg,
};

/// An integer atom type.
///
/// These are the builtin multi-bit integer types.
enum class IntAtomType {
  /// A `byte`.
  Byte,
  /// A `shortint`.
  ShortInt,
  /// An `int`.
  Int,
  /// A `longint`.
  LongInt,
  /// An `integer`.
  Integer,
  /// A `time`.
  Time,
};

//===----------------------------------------------------------------------===//
// Packed Core
//===----------------------------------------------------------------------===//

/// A core packed type.
class PackedCore : public Type {
protected:
  using Type::Type;
};

/// An error occurred during type computation.
class ErrorPackedCore : public PackedCore::TypeBase<ErrorPackedCore, PackedCore,
                                                    DefaultTypeStorage> {
public:
  using Base::Base;
  static ErrorPackedCore get(MLIRContext *context) {
    return Base::get(context);
  }
};

/// Void.
class VoidPackedCore : public PackedCore::TypeBase<VoidPackedCore, PackedCore,
                                                   DefaultTypeStorage> {
public:
  using Base::Base;
  static VoidPackedCore get(MLIRContext *context) { return Base::get(context); }
};

/// An integer vector type.
class IntVecPackedCore
    : public PackedCore::TypeBase<IntVecPackedCore, PackedCore,
                                  detail::IntTypeStorage> {
public:
  using Base::Base;
  static IntVecPackedCore get(MLIRContext *context, IntVecType type);
  operator IntVecType() const;
};

/// An integer atom type.
class IntAtomPackedCore
    : public PackedCore::TypeBase<IntAtomPackedCore, PackedCore,
                                  detail::IntTypeStorage> {
public:
  using Base::Base;
  static IntAtomPackedCore get(MLIRContext *context, IntAtomType type);
  operator IntAtomType() const;
};

/// A packed struct.
class StructPackedCore
    : public PackedCore::TypeBase<StructPackedCore, PackedCore,
                                  DefaultTypeStorage> {
public:
  using Base::Base;
  static StructPackedCore get(MLIRContext *context) {
    return Base::get(context);
  }
};

/// An enum.
class EnumPackedCore : public PackedCore::TypeBase<EnumPackedCore, PackedCore,
                                                   DefaultTypeStorage> {
public:
  using Base::Base;
  static EnumPackedCore get(MLIRContext *context) { return Base::get(context); }
};

/// A named type.
class NamedPackedCore : public PackedCore::TypeBase<NamedPackedCore, PackedCore,
                                                    DefaultTypeStorage> {
public:
  using Base::Base;
  static NamedPackedCore get(MLIRContext *context) {
    return Base::get(context);
  }
};

/// A type reference.
class RefPackedCore : public PackedCore::TypeBase<RefPackedCore, PackedCore,
                                                  DefaultTypeStorage> {
public:
  using Base::Base;
  static RefPackedCore get(MLIRContext *context) { return Base::get(context); }
};

//===----------------------------------------------------------------------===//
// Packed Dimension
//===----------------------------------------------------------------------===//

/// A packed dimension.
class PackedDim : public Type {
protected:
  using Type::Type;
};

/// A range dimension, like `[a:b]`.
class RangePackedDim : public PackedDim::TypeBase<RangePackedDim, PackedDim,
                                                  detail::TypeRangeStorage> {
public:
  using Base::Base;

  static RangePackedDim get(MLIRContext *context, const TypeRange &range) {
    return Base::get(context, range);
  }

  template <typename... Args>
  static RangePackedDim get(MLIRContext *context, Args &&...args) {
    return Base::get(context, TypeRange(args...));
  }

  operator const TypeRange &() const;
};

/// An unsized dimension, like `[]`.
class UnsizedPackedDim : public PackedDim::TypeBase<UnsizedPackedDim, PackedDim,
                                                    DefaultTypeStorage> {
public:
  using Base::Base;
  static UnsizedPackedDim get(MLIRContext *context) {
    return Base::get(context);
  }
};

//===----------------------------------------------------------------------===//
// Packed Type
//===----------------------------------------------------------------------===//

/// A packed SystemVerilog type.
class PackedType
    : public SVType::TypeBase<PackedType, SVType, detail::PackedTypeStorage> {
public:
  using Base::Base;
};

} // namespace moore
} // namespace circt

#endif // CIRCT_DIALECT_MOORE_SV_TYPES_H
