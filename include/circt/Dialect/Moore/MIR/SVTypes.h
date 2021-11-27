//===- MIRTypes.h - Declare Moore MIR dialect types --------------*- C++-*-===//
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
struct PackedTypeStorage;
} // namespace detail

/// A common base class for all SystemVerilog types.
class SVType : public Type {};

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
};

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
  Count,
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
  Count,
};

//===----------------------------------------------------------------------===//
// Packed Core
//===----------------------------------------------------------------------===//

/// A core packed type.
class PackedCore {
public:
  bool operator==(const PackedCore &other) const;

  class Error;
  class Void;
  class IntVec;
  class IntAtom;
  class Struct;
  class Enum;
  class Named;
  class Ref;

protected:
  enum class Kind {
    Error,
    Void,
    IntVecFirst,
    IntVecLast = IntVecFirst + static_cast<int>(IntVecType::Count) - 1,
    IntAtomFirst,
    IntAtomLast = IntAtomFirst + static_cast<int>(IntAtomType::Count) - 1,
    Struct,
    Enum,
    Named,
    Ref,
  };
  const Kind kind;
  PackedCore(Kind kind) : kind(kind) {}

  template <class ConcreteTy>
  friend class PackedCoreImpl;
};

template <class ConcreteTy>
class PackedCoreImpl : public PackedCore {
protected:
  PackedCoreImpl() : PackedCore(ConcreteTy::ConcreteKind) {}

public:
  static bool classof(const PackedCore *core) {
    return core->kind == ConcreteTy::ConcreteKind;
  }
};

/// An error occurred during type computation.
class PackedCore::Error : public PackedCoreImpl<PackedCore::Error> {
  static constexpr auto ConcreteKind = Kind::Error;
};

/// Void.
class PackedCore::Void : public PackedCoreImpl<PackedCore::Void> {
  static constexpr auto ConcreteKind = Kind::Void;
};

/// An integer vector type.
class PackedCore::IntVec : public PackedCore {
public:
  IntVec(IntVecType type)
      : PackedCore(static_cast<Kind>(static_cast<int>(type) +
                                     static_cast<int>(Kind::IntVecFirst))) {}
  IntVecType operator()() const {
    return static_cast<IntVecType>(static_cast<int>(kind) -
                                   static_cast<int>(Kind::IntVecFirst));
  }
  static bool classof(const PackedCore *core) {
    return core->kind >= Kind::IntVecFirst && core->kind <= Kind::IntVecLast;
  }
};

/// An integer atom type.
class PackedCore::IntAtom : public PackedCore {
public:
  IntAtom(IntAtomType type)
      : PackedCore(static_cast<Kind>(static_cast<int>(type) +
                                     static_cast<int>(Kind::IntAtomFirst))) {}
  IntAtomType operator()() const {
    return static_cast<IntAtomType>(static_cast<int>(kind) -
                                    static_cast<int>(Kind::IntAtomFirst));
  }
  static bool classof(const PackedCore *core) {
    return core->kind >= Kind::IntAtomFirst && core->kind <= Kind::IntAtomLast;
  }
};

/// A packed struct.
class PackedCore::Struct : public PackedCoreImpl<PackedCore::Struct> {
  static constexpr auto ConcreteKind = Kind::Struct;
};

/// An enum.
class PackedCore::Enum : public PackedCoreImpl<PackedCore::Enum> {
  static constexpr auto ConcreteKind = Kind::Enum;
};

/// A named type.
class PackedCore::Named : public PackedCoreImpl<PackedCore::Named> {
  static constexpr auto ConcreteKind = Kind::Named;
};

/// A type reference.
class PackedCore::Ref : public PackedCoreImpl<PackedCore::Ref> {
  static constexpr auto ConcreteKind = Kind::Ref;
};

//===----------------------------------------------------------------------===//
// Packed Dimension
//===----------------------------------------------------------------------===//

/// A packed dimension.
class PackedDim {
public:
  bool operator==(const PackedDim &other) const;

protected:
  enum class Kind {
    Range,
    Unsized,
  };
  const Kind kind;
  PackedDim(Kind kind) : kind(kind) {}

  template <class ConcreteTy>
  friend class PackedDimImpl;
};

template <class ConcreteTy>
class PackedDimImpl : public PackedDim {
protected:
  PackedDimImpl() : PackedDim(ConcreteTy::ConcreteKind) {}

public:
  static bool classof(const PackedDim *dim) {
    return dim->kind == ConcreteTy::ConcreteKind;
  }
};

/// A range dimension, like `[a:b]`.
class RangePackedDim : public PackedDimImpl<RangePackedDim> {
  static constexpr auto ConcreteKind = Kind::Range;
};

/// An unsized dimension, like `[]`.
class UnsizedPackedDim : public PackedDimImpl<UnsizedPackedDim> {
  static constexpr auto ConcreteKind = Kind::Unsized;
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
