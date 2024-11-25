//===- ElaborationPass.cpp - RTG ElaborationPass implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass elaborates the random parts of the RTG dialect.
// It performs randomization top-down, i.e., random constructs in a sequence
// that is invoked multiple times can yield different randomization results
// for each invokation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/ArithVisitors.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/IR/RTGVisitors.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/FoldingSet.h"
#include <deque>
#include <random>

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_ELABORATIONPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::rtg;
using llvm::MapVector;

#define DEBUG_TYPE "rtg-elaboration"

//===----------------------------------------------------------------------===//
// Uniform Distribution Helper
//
// Simplified version of
// https://github.com/llvm/llvm-project/blob/main/libcxx/include/__random/uniform_int_distribution.h
// We use our custom version here to get the same results when compiled with
// different compiler versions and standard libraries.
//===----------------------------------------------------------------------===//

static uint32_t computeMask(size_t w) {
  size_t n = w / 32 + (w % 32 != 0);
  size_t w0 = w / n;
  return w0 > 0 ? uint32_t(~0) >> (32 - w0) : 0;
}

/// Get a number uniformly at random in the in specified range.
static uint32_t getUniformlyInRange(std::mt19937 &rng, uint32_t a, uint32_t b) {
  const uint32_t diff = b - a + 1;
  if (diff == 1)
    return a;

  const uint32_t digits = std::numeric_limits<uint32_t>::digits;
  if (diff == 0)
    return rng();

  uint32_t width = digits - llvm::countl_zero(diff) - 1;
  if ((diff & (std::numeric_limits<uint32_t>::max() >> (digits - width))) != 0)
    ++width;

  uint32_t mask = computeMask(diff);
  uint32_t u;
  do {
    u = rng() & mask;
  } while (u >= diff);

  return u + a;
}

//===----------------------------------------------------------------------===//
// Elaborator Values
//===----------------------------------------------------------------------===//

namespace {

/// The abstract base class for elaborated values.
class ElaboratorValue {
public:
  enum class ValueKind { Attribute, Set, Bag, Integer, Index, NullValue };

  union StorageTy {
    const void * ptr;
    size_t index;
   };

  ElaboratorValue() : ElaboratorValue(ValueKind::NullValue, nullptr) { }
  ElaboratorValue(std::nullptr_t storage) : ElaboratorValue() { }
  // ElaboratorValue &operator=(const ElaboratorValue &other) = default;
  // ~ElaboratorValue() = default;
  // {
  //   kind = other.kind;
  //   storage = other.storage;
  //   return *this;
  // }

  llvm::hash_code getHashValue() const {
    return llvm::hash_value(storage.ptr);
    // ArrayRef<const uint8_t> data(reinterpret_cast<const uint8_t *>(&storage), sizeof(storage));
    // return llvm::hash_combine(kind, llvm::hash_combine_range(data.begin(), data.end()));
  }

  bool operator ==(const ElaboratorValue &other) const {
    return kind == other.kind && storage.ptr == other.storage.ptr;// && std::memcmp(&storage, &other.storage, sizeof(storage));
  }

  operator bool() const {
    return kind != ValueKind::NullValue;
  }

  ValueKind kind = ValueKind::NullValue;

  friend struct llvm::DenseMapInfo<ElaboratorValue>;
  
// protected:
  ElaboratorValue(ValueKind kind, const void *storagePtr) : kind(kind) {
    std::memset(&storage, 0, sizeof(storage));
    storage.ptr = storagePtr;
  }
  ElaboratorValue(ValueKind kind, uint64_t index) : kind(kind) {
    std::memset(&storage, 0, sizeof(storage));
    storage.index = index;
  }
  ElaboratorValue(ValueKind kind, StorageTy storage) : kind(kind), storage(storage) { }

  StorageTy storage;
};

// NOLINTNEXTLINE(readability-identifier-naming)
llvm::hash_code hash_value(const ElaboratorValue &val) {
  return val.getHashValue();
}
} // namespace

namespace llvm {

/// Add support for llvm style casts. We provide a cast between To and From if
/// From is mlir::Attribute or derives from it.
template <typename To, typename From>
struct CastInfo<To, From,
                std::enable_if_t<std::is_same_v<ElaboratorValue,
                                                std::remove_const_t<From>> ||
                                 std::is_base_of_v<ElaboratorValue, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  /// Arguments are taken as mlir::Attribute here and not as `From`, because
  /// when casting from an intermediate type of the hierarchy to one of its
  /// children, the val.getTypeID() inside T::classof will use the static
  /// getTypeID of the parent instead of the non-static Type::getTypeID that
  /// returns the dynamic ID. This means that T::classof would end up comparing
  /// the static TypeID of the children to the static TypeID of its parent,
  /// making it impossible to downcast from the parent to the child.
  static inline bool isPossible(ElaboratorValue ty) {
    /// Return a constant true instead of a dynamic true when casting to self or
    /// up the hierarchy.
    if constexpr (std::is_base_of_v<To, From>) {
      return true;
    } else {
      return To::classof(ty);
    }
  }
  static inline To doCast(ElaboratorValue value) { return To(value.storage); }
};

template<>
struct DenseMapInfo<ElaboratorValue> {
  static inline ElaboratorValue getEmptyKey() { return ElaboratorValue(); }

  static inline ElaboratorValue getTombstoneKey() { return ElaboratorValue(ElaboratorValue::ValueKind::NullValue, reinterpret_cast<const void *>(~0ULL)); }

  static unsigned getHashValue(const ElaboratorValue &value) {
    return value.getHashValue();
  }

  static bool isEqual(const ElaboratorValue &lhs, const ElaboratorValue &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace {
struct SetStorage : public llvm::FoldingSetNode {
  SetStorage(SetVector<ElaboratorValue> &&set, Type type)
      : cachedHash(llvm::hash_combine(
            llvm::hash_combine_range(set.begin(), set.end()), type)), set(std::move(set)), type(type) { }
  // SetStorage &operator=(const SetStorage &other) {
  //   set = other.set;
  //   type
  //   return *this;
  // }

  bool operator==(const SetStorage &other) const {
    // Compare against the hash first and short circuit if it doesn't match.
    return cachedHash == other.cachedHash && set == other.set && type == other.type;
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  void Profile(llvm::FoldingSetNodeID &ID) const {
    for (auto el : set) {
      ID.AddPointer(el.storage.ptr);
    }
    ID.AddPointer(type.getAsOpaquePointer());
  }

  // Compute the hash only once at constructor time.
  llvm::hash_code cachedHash;

  // Stores the elaborated values of the set.
  SetVector<ElaboratorValue> set;

  // Store the set type such that we can materialize this evaluated value
  // also in the case where the set is empty.
  Type type;
};

struct BagStorage : public llvm::FoldingSetNode {
  BagStorage(MapVector<ElaboratorValue, uint64_t> &&bag, Type type)
      : cachedHash(llvm::hash_combine(
            llvm::hash_combine_range(bag.begin(), bag.end()), type)), bag(std::move(bag)), type(type) {}
  // BagStorage &operator=(const BagStorage &other) = default;

  bool operator==(const BagStorage &other) const {
    // Compare against the hash first and short circuit if it doesn't match.
    return cachedHash == other.cachedHash && llvm::equal(bag, other.bag) && type == other.type;
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  void Profile(llvm::FoldingSetNodeID &ID) const {
    // ID.AddInteger(bag.size());
    for (auto el : bag) {
      ID.AddPointer(el.first.storage.ptr);
      ID.AddInteger(el.second);
    }
    ID.AddPointer(type.getAsOpaquePointer());
  }

  // Compute the hash only once at constructor time.
  llvm::hash_code cachedHash;

  // Stores the elaborated values of the bag.
  MapVector<ElaboratorValue, uint64_t> bag;

  // Store the bag type such that we can materialize this evaluated value
  // also in the case where the bag is empty.
  Type type;
};

} // namespace

namespace llvm {
template<>
struct DenseMapInfo<SetStorage> {
  static inline SetStorage getEmptyKey() { return SetStorage({}, llvm::DenseMapInfo<Type>::getEmptyKey()); }

  static inline SetStorage getTombstoneKey() { return SetStorage({}, llvm::DenseMapInfo<Type>::getTombstoneKey()); }

  static unsigned getHashValue(const SetStorage &value) {
    return value.cachedHash;
  }

  static bool isEqual(const SetStorage &lhs, const SetStorage &rhs) {
    return lhs == rhs;
  }
};

template<>
struct DenseMapInfo<BagStorage> {
  static inline BagStorage getEmptyKey() { return BagStorage({}, llvm::DenseMapInfo<Type>::getEmptyKey()); }

  static inline BagStorage getTombstoneKey() { return BagStorage({}, llvm::DenseMapInfo<Type>::getTombstoneKey()); }

  static unsigned getHashValue(const BagStorage &value) {
    return value.cachedHash;
  }

  static bool isEqual(const BagStorage &lhs, const BagStorage &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

// struct StorageBase {
//   virtual ~StorageBase() {}
//   virtual llvm::hash_code getHashValue() const;
//   virtual bool isEqual(const StorageBase &other) const;
// };
namespace {
class Internalizer {
public:
  Internalizer() = default;

  // template<typename StorageAdaptor>
  // StorageBase *internalize(const StorageAdaptor &adaptor) {
  //   // If such an object is already interned, don't actually construct it and just return the interned version.
  //   if (auto iter = interned.find(adaptor); iter != interned.end())
  //     return *iter;

  //   // Otherwise we must allocate memory and construct the object.
  //   // Unfortunately this means we need to do another lookup.
  //   return interned.insert(adaptor.allocateAndGetOwnership()).second;
  // }

  // ~Internalizer() {
  //   for (auto *el : interned)
  //     delete el;
  // }

  SetStorage *internalize(SetStorage &&storage) {
    llvm::FoldingSetNodeID profile;
    storage.Profile(profile);
    void *insertPos = nullptr;
    if (auto *set = internedSets.FindNodeOrInsertPos(profile, insertPos))
      return set;

    auto *storagePtr = new SetStorage(std::move(storage));
    internedSets.InsertNode(storagePtr, insertPos);
    return storagePtr;
  }

  BagStorage *internalize(BagStorage &&storage) {
    llvm::FoldingSetNodeID profile;
    storage.Profile(profile);
    void *insertPos = nullptr;
    if (auto *bag = internedBags.FindNodeOrInsertPos(profile, insertPos))
      return bag;

    auto *storagePtr = new BagStorage(std::move(storage));
    internedBags.InsertNode(storagePtr, insertPos);
    return storagePtr;
  }

  // SetStorage *internalize(SetStorage &&storage) {
  //   return &*internedSets.insert(std::move(storage)).first;
  // }

  // BagStorage *internalize(BagStorage &&storage) {
  //   return &*internedBags.insert(std::move(storage)).first;
  // }

  ~Internalizer() {
    // for (const auto &el : internedSets)
    //   delete &el;
    // internedSets.clear();

    // for (const auto &el : internedBags)
    //   delete &el;
    // internedBags.clear();
  }

private:
  // A map used to intern elaborator values. We do this such that we can
  // compare pointers when, e.g., computing set differences, uniquing the
  // elements in a set, etc. Otherwise, we'd need to do a deep value comparison
  // in those situations.
  // Use a pointer as the key with custom MapInfo because of object slicing when
  // inserting an object of a derived class of ElaboratorValue.
  // The custom MapInfo makes sure that we do a value comparison instead of
  // comparing the pointers.
  llvm::FoldingSet<SetStorage> internedSets;
  llvm::FoldingSet<BagStorage> internedBags;
  // DenseSet<SetStorage> internedSets;
  // DenseSet<BagStorage> internedBags;
};

/// Holds any typed attribute. Wrapping around an MLIR `Attribute` allows us to
/// use this elaborator value class for any values that have a corresponding
/// MLIR attribute rather than one per kind of attribute. We only support typed
/// attributes because for materialization we need to provide the type to the
/// dialect's materializer.
class AttributeValue : public ElaboratorValue {
public:
  AttributeValue(TypedAttr attr)
      : ElaboratorValue(ValueKind::Attribute, attr.getAsOpaquePointer()) {
    assert(attr && "null attributes not allowed");
    assert(!isa<IndexType>(attr.getType()) && "IndexValue should be used for constant indices");
  }

  AttributeValue(StorageTy storage) : ElaboratorValue(ValueKind::Attribute, storage) {}
  AttributeValue(std::nullptr_t storage) : ElaboratorValue() { }
  // AttributeValue &operator=(const AttributeValue &other) = default;

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue &val) {
    return val.kind == ValueKind::Attribute;
  }

  TypedAttr getAttr() const { return cast<TypedAttr>(Attribute::getFromOpaquePointer(storage.ptr)); }
};

/// Holds an evaluated value of a `SetType`'d value.
struct SetValue : public ElaboratorValue {
  SetValue(Internalizer &internalizer, SetVector<ElaboratorValue> &&set, Type type)
      : ElaboratorValue(ValueKind::Set, internalizer.internalize(SetStorage(std::move(set), type))) {
      }

  SetValue(StorageTy storage) : ElaboratorValue(ValueKind::Set, storage) {}
  SetValue(std::nullptr_t storage) : ElaboratorValue() { }
  // SetValue &operator=(const SetValue &other) = default;

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue &val) {
    return val.kind == ValueKind::Set;
  }

  const SetVector<ElaboratorValue> &getSet() const { return static_cast<const SetStorage*>(storage.ptr)->set; }

  Type getType() const { return static_cast<const SetStorage*>(storage.ptr)->type; }
};

/// Holds an evaluated value of a `BagType`'d value.
struct BagValue : public ElaboratorValue {
  BagValue(Internalizer &internalizer, MapVector<ElaboratorValue, uint64_t> &&bag, Type type)
      : ElaboratorValue(ValueKind::Bag, internalizer.internalize(BagStorage(std::move(bag), type))) {}

  BagValue(StorageTy storage) : ElaboratorValue(ValueKind::Bag, storage) {}
  BagValue(std::nullptr_t storage) : ElaboratorValue() { }
  // BagValue &operator=(const BagValue &other) = default;
  //   kind = other.kind;
  //   storage = other.storage;
  //   return *this;
  // }

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue &val) {
    return val.kind == ValueKind::Bag;
  }

  const MapVector<ElaboratorValue, uint64_t> &getBag() const { return static_cast<const BagStorage*>(storage.ptr)->bag; }

  Type getType() const { return static_cast<const BagStorage*>(storage.ptr)->type; }
};

/// Holds an evaluated index value.
struct IndexValue : public ElaboratorValue {
  IndexValue(size_t value)
      : ElaboratorValue(ValueKind::Index, value) {}

  IndexValue(StorageTy storage) : ElaboratorValue(ValueKind::Index, storage) {}
  IndexValue(std::nullptr_t storage) : ElaboratorValue() { }
  // IndexValue &operator=(const IndexValue &other) = default;

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue &val) {
    return val.kind == ValueKind::Index;
  }

  size_t getIndex() const { return storage.index; }
};

} // namespace

// /// Holds an evaluated integer value of a specific type.
// class IntegerValue : public ElaboratorValue {
// public:
//   IntegerValue(const APInt &value, Type type)
//       : ElaboratorValue(ValueKind::Integer, nullptr), value(value), type(type) {}

//   // Implement LLVMs RTTI
//   static bool classof(const ElaboratorValue &val) {
//     return val.kind == ValueKind::Integer;
//   }

//   llvm::hash_code getHashValue() const override {
//     return llvm::hash_combine(value, getType());
//   }

//   bool operator ==(const ElaboratorValue &other) const override {
//     if (auto intVal = dyn_cast<IntegerValue>(other))
//       return type == intVal.type && value == intVal.value;

//     return false;
//   }

//   const APInt &getValue() const { return value; }

//   Type getType() const { return type; }

// private:
//   APInt value;
//   Type type;
// };

#ifndef NDEBUG
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ElaboratorValue &value) {
  TypeSwitch<ElaboratorValue>(value)
      .Case<AttributeValue>(
          [&](auto val) { 
            os << "<attr " << val.getAttr() << ">";
            })
      .Case<SetValue>(
          [&](auto val) { 
              os << "<set {";
              llvm::interleaveComma(val.getSet(), os);
              os << "} at " << val.storage.ptr << ">";
            })
      .Case<BagValue>(
          [&](auto val) { 
              os << "<bag {";
              llvm::interleaveComma(val.getBag(), os,
                                    [&](const std::pair<ElaboratorValue, uint64_t> &el) {
                                      os << el.first << " -> " << el.second;
                                    });
              os << "} at " << val.storage.ptr << ">";
            })
      .Case<IndexValue>(
          [&](auto val) { 
          os << "<index " << val.getIndex() << ">";
            })
      .Default([](auto val) {
        assert(false && "all cases must be covered above");
        return Value();
      });
  return os;
}
#endif

//===----------------------------------------------------------------------===//
// Main Elaborator Implementation
//===----------------------------------------------------------------------===//

namespace {

/// Construct an SSA value from a given elaborated value.
class Materializer {
public:
  Value materialize(const ElaboratorValue &val, Block *block, Location loc,
                    function_ref<InFlightDiagnostic()> emitError) {
    auto iter = materializedValues.find({val, block});
    if (iter != materializedValues.end())
      return iter->second;

    auto [builderIter, _] =
        builderPerBlock.insert({block, OpBuilder::atBlockBegin(block)});
    OpBuilder builder = builderIter->second;

    return TypeSwitch<ElaboratorValue, Value>(val)
        .Case<AttributeValue, SetValue, BagValue, IndexValue>(
            [&](auto val) { return visit(val, builder, loc, emitError); })
        .Default([](auto val) {
          assert(false && "all cases must be covered above");
          return Value();
        });
  }

  void clear() {
    materializedValues.clear();
    builderPerBlock.clear();
  }

private:
  Value visit(const AttributeValue &val, OpBuilder &builder, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    // Call the materializer of the dialect defining that attribute.
    auto attr = val.getAttr();

    if (isa<IntegerAttr>(attr)) {
      Value res = builder.create<arith::ConstantOp>(loc, attr);
      materializedValues[{val, builder.getBlock()}] = res;
      return res;
    }

    auto *op = attr.getDialect().materializeConstant(builder, attr,
                                                     attr.getType(), loc);
    if (!op) {
      emitError() << "materializer of dialect '"
                  << attr.getDialect().getNamespace()
                  << "' unable to materialize value for attribute '" << attr
                  << "'";
      return Value();
    }

    Value res = op->getResult(0);
    materializedValues[{val, builder.getBlock()}] = res;
    return res;
  }

  Value visit(const SetValue &val, OpBuilder &builder, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> elements;
    elements.reserve(val.getSet().size());
    for (const auto &el : val.getSet()) {
      auto materialized = materialize(el, builder.getBlock(), loc, emitError);
      if (!materialized)
        return Value();

      elements.push_back(materialized);
    }

    auto res = builder.create<SetCreateOp>(loc, val.getType(), elements);
    materializedValues[{val, builder.getBlock()}] = res;
    return res;
  }

  Value visit(const BagValue &val, OpBuilder &builder, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> values, weights;
    values.reserve(val.getBag().size());
    weights.reserve(val.getBag().size());
    for (auto [val, weight] : val.getBag()) {
      auto materializedVal =
          materialize(val, builder.getBlock(), loc, emitError);
      auto materializedWeight =
          materialize(IndexValue(weight), builder.getBlock(), loc, emitError);
      if (!materializedVal || !materializedWeight)
        return Value();

      values.push_back(materializedVal);
      weights.push_back(materializedWeight);
    }

    auto res =
        builder.create<BagCreateOp>(loc, val.getType(), values, weights);
    materializedValues[{val, builder.getBlock()}] = res;
    return res;
  }

  Value visit(const IndexValue &val, OpBuilder &builder, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    auto attr = IntegerAttr::get(builder.getIndexType(), val.getIndex());
    Value res = builder.create<arith::ConstantOp>(loc, attr);
    materializedValues[{val, builder.getBlock()}] = res;
    return res;
  }

private:
  /// Cache values we have already materialized to reuse them later. We start
  /// with an insertion point at the start of the block and cache the (updated)
  /// insertion point such that future materializations can also reuse previous
  /// materializations without running into dominance issues (or requiring
  /// additional checks to avoid them).
  DenseMap<std::pair<ElaboratorValue, Block *>, Value> materializedValues;

  /// Cache the builders to continue insertions at their current insertion point
  /// for the reason stated above.
  DenseMap<Block *, OpBuilder> builderPerBlock;
};

/// Used to signal to the elaboration driver whether the operation should be
/// removed.
enum class DeletionKind { Keep, Delete };

/// Interprets the IR to perform and lower the represented randomizations.
class Elaborator
    : public RTGOpVisitor<Elaborator, FailureOr<DeletionKind>,
                          function_ref<void(Operation *)>>,
      public mlir::arith::ArithOpVisitor<Elaborator, FailureOr<DeletionKind>,
                                         function_ref<void(Operation *)>> {
public:
  using RTGBase = RTGOpVisitor<Elaborator, FailureOr<DeletionKind>,
                               function_ref<void(Operation *)>>;
  using ArithBase = ArithOpVisitor<Elaborator, FailureOr<DeletionKind>,
                                   function_ref<void(Operation *)>>;

  using ArithBase::visitOp;
  using RTGBase::visitOp;
  using RTGBase::visitRegisterOp;

  Elaborator(SymbolTable &table, std::mt19937 &rng) : rng(rng) {}

  inline void store(Value val, const ElaboratorValue &eval) {
    if (!val.hasOneUse() || *val.getUsers().begin() != nextOp)
      state[val] = eval;
    stateCache = {val, eval};
  }

  template<typename ValueTy>
  inline ValueTy get(Value val) {
    if (stateCache.first == val)
      return cast<ValueTy>(stateCache.second);

    return cast<ValueTy>(state.at(val));
  }
  inline ElaboratorValue get(Value val) {
    if (stateCache.first == val)
      return stateCache.second;

    return state.at(val);
  }

  /// Print a nice error message for operations we don't support yet.
  FailureOr<DeletionKind>
  visitUnhandledOp(Operation *op,
                   function_ref<void(Operation *)> addToWorklist) {
    return op->emitOpError("elaboration not supported");
  }

  FailureOr<DeletionKind>
  visitExternalOp(Operation *op,
                  function_ref<void(Operation *)> addToWorklist) {
    // TODO: we only have this to be able to write tests for this pass without
    // having to add support for more operations for now, so it should be
    // removed once it is not necessary anymore for writing tests
    if (op->use_empty()) {
      for (auto &operand : op->getOpOperands()) {
        auto emitError = [&]() {
          auto diag = op->emitError();
          diag.attachNote(op->getLoc())
              << "while materializing value for operand#"
              << operand.getOperandNumber();
          return diag;
        };
        Value val = materializer.materialize(
            get(operand.get()), op->getBlock(), op->getLoc(), emitError);
        if (!val)
          return failure();
        operand.set(val);
      }
      return DeletionKind::Keep;
    }

    return visitUnhandledOp(op, addToWorklist);
  }

  FailureOr<DeletionKind>
  visitOp(arith::AddIOp op, function_ref<void(Operation *)> addToWorklist) {
    store(op.getResult(), IndexValue(get<IndexValue>(op.getLhs()).getIndex() + get<IndexValue>(op.getRhs()).getIndex()));
    return DeletionKind::Delete;
  }

  // FailureOr<DeletionKind>
  // visitOp(SetCreateOp op, function_ref<void(Operation *)> addToWorklist) {
  //   SetVector<ElaboratorValue> set;
  //   for (auto val : op.getElements())
  //     set.insert(state.at(val));

  //   state[op.getSet()] = SetValue(internalizer, std::move(set), op.getSet().getType());
  //   return DeletionKind::Delete;
  // }

  // FailureOr<DeletionKind>
  // visitOp(SetSelectRandomOp op, function_ref<void(Operation *)> addToWorklist) {
  //   auto set = cast<SetValue>(state.at(op.getSet()));

  //   size_t selected;
  //   if (auto intAttr =
  //           op->getAttrOfType<IntegerAttr>("rtg.elaboration_custom_seed")) {
  //     std::mt19937 customRng(intAttr.getInt());
  //     selected = getUniformlyInRange(customRng, 0, set.getSet().size() - 1);
  //   } else {
  //     selected = getUniformlyInRange(rng, 0, set.getSet().size() - 1);
  //   }

  //   state[op.getResult()] = set.getSet()[selected];
  //   return DeletionKind::Delete;
  // }

  // FailureOr<DeletionKind>
  // visitOp(SetDifferenceOp op, function_ref<void(Operation *)> addToWorklist) {
  //   auto original = cast<SetValue>(state.at(op.getOriginal())).getSet();
  //   auto diff = cast<SetValue>(state.at(op.getDiff())).getSet();

  //   SetVector<ElaboratorValue> result(original);
  //   result.set_subtract(diff);

  //   state[op.getResult()] = SetValue(internalizer, std::move(result),
  //                               op.getResult().getType());
  //   return DeletionKind::Delete;
  // }

  // FailureOr<DeletionKind>
  // visitOp(SetUnionOp op, function_ref<void(Operation *)> addToWorklist) {
  //   SetVector<ElaboratorValue> result;
  //   for (auto set : op.getSets())
  //     result.set_union(cast<SetValue>(state.at(set)).getSet());

  //   state[op.getResult()] = SetValue(internalizer, std::move(result),
  //                               op.getType());
  //   return DeletionKind::Delete;
  // }

  // FailureOr<DeletionKind>
  // visitOp(SetSizeOp op, function_ref<void(Operation *)> addToWorklist) {
  //   auto size = cast<SetValue>(state.at(op.getSet())).getSet().size();
  //   state[op.getResult()] = IndexValue(size);
  //   return DeletionKind::Delete;
  // }

  // FailureOr<DeletionKind>
  // visitOp(BagCreateOp op, function_ref<void(Operation *)> addToWorklist) {
  //   MapVector<ElaboratorValue, uint64_t> bag;
  //   for (auto [val, multiple] :
  //        llvm::zip(op.getElements(), op.getMultiples())) {
  //     auto interpValue = state.at(val);
  //     // If the multiple is not stored as an AttributeValue, the elaboration
  //     // must have already failed earlier (since we don't have
  //     // unevaluated/opaque values).
  //     auto interpMultiple = cast<IndexValue>(state.at(multiple));
  //     bag[interpValue] += interpMultiple.getIndex();
  //   }

  //   state[op.getBag()] = BagValue(internalizer, std::move(bag), op.getType());
  //   return DeletionKind::Delete;
  // }

  // FailureOr<DeletionKind>
  // visitOp(BagSelectRandomOp op, function_ref<void(Operation *)> addToWorklist) {
  //   auto bag = cast<BagValue>(state.at(op.getBag()));

  //   SmallVector<std::pair<ElaboratorValue, uint32_t>> prefixSum;
  //   prefixSum.reserve(bag.getBag().size());
  //   uint32_t accumulator = 0;
  //   for (auto [val, weight] : bag.getBag()) {
  //     accumulator += weight;
  //     prefixSum.push_back({val, accumulator});
  //   }

  //   auto customRng = rng;
  //   if (auto intAttr =
  //           op->getAttrOfType<IntegerAttr>("rtg.elaboration_custom_seed")) {
  //     customRng = std::mt19937(intAttr.getInt());
  //   }

  //   auto idx = getUniformlyInRange(customRng, 0, accumulator - 1);
  //   auto *iter = llvm::upper_bound(
  //       prefixSum, idx,
  //       [](uint32_t a, const std::pair<ElaboratorValue, uint32_t> &b) {
  //         return a < b.second;
  //       });
  //   state[op.getResult()] = iter->first;
  //   return DeletionKind::Delete;
  // }

  // FailureOr<DeletionKind>
  // visitOp(BagDifferenceOp op, function_ref<void(Operation *)> addToWorklist) {
  //   auto original = cast<BagValue>(state.at(op.getOriginal()));
  //   auto diff = cast<BagValue>(state.at(op.getDiff()));

  //   MapVector<ElaboratorValue, uint64_t> result;
  //   for (const auto &el : original.getBag()) {
  //     if (!diff.getBag().contains(el.first)) {
  //       result.insert(el);
  //       continue;
  //     }

  //     if (op.getInf())
  //       continue;

  //     auto toDiff = diff.getBag().lookup(el.first);
  //     if (el.second <= toDiff)
  //       continue;

  //     result.insert({el.first, el.second - toDiff});
  //   }

  //   state[op.getResult()] = BagValue(internalizer, std::move(result),
  //                               op.getType());
  //   return DeletionKind::Delete;
  // }

  // FailureOr<DeletionKind>
  // visitOp(BagUnionOp op, function_ref<void(Operation *)> addToWorklist) {
  //   MapVector<ElaboratorValue, uint64_t> result;
  //   for (auto bag : op.getBags()) {
  //     auto val = cast<BagValue>(state.at(bag));
  //     for (auto [el, multiple] : val.getBag())
  //       result[el] += multiple;
  //   }

  //   state[op.getResult()] = BagValue(internalizer, std::move(result),
  //                               op.getType());
  //   return DeletionKind::Delete;
  // }

  // FailureOr<DeletionKind>
  // visitOp(BagUniqueSizeOp op, function_ref<void(Operation *)> addToWorklist) {
  //   auto size = cast<BagValue>(state.at(op.getBag())).getBag().size();
  //   state[op.getResult()] = IndexValue(size);
  //   return DeletionKind::Delete;
  // }

  FailureOr<DeletionKind>
  dispatchOpVisitor(Operation *op,
                    function_ref<void(Operation *)> addToWorklist) {
    if (op->hasTrait<OpTrait::ConstantLike>()) {
      SmallVector<OpFoldResult, 1> result;
      auto foldResult = op->fold(result);
      (void)foldResult; // Make sure there is a user when assertions are off.
      assert(succeeded(foldResult) &&
             "constant folder of a constant-like must always succeed");

      // We have a special elaboration value for integers for better performance.
      if (auto intAttr = dyn_cast<IntegerAttr>(result[0].dyn_cast<Attribute>()); intAttr && isa<IndexType>(intAttr.getType())) {
        store(op->getResult(0), IndexValue(intAttr.getInt()));
        return DeletionKind::Delete;
      }

      auto attr = dyn_cast<TypedAttr>(result[0].dyn_cast<Attribute>());
      if (!attr)
        return op->emitError(
            "only typed attributes supported for constant-like operations");

      store(op->getResult(0), AttributeValue(attr));
      return DeletionKind::Delete;
    }

    if (op->getDialect()->getNamespace() == "rtg")
      return RTGBase::dispatchOpVisitor(op, addToWorklist);

    return ArithBase::dispatchOpVisitor(op, addToWorklist);
  }

  LogicalResult elaborate(TestOp testOp) {
    LLVM_DEBUG(llvm::dbgs()
               << "\n=== Elaborating Test @" << testOp.getSymName() << "\n\n");

    // SmallVector<Node> visited;
    // std::deque<Node> worklist;
    SmallVector<Operation *> toDelete;

    // SmallVector<Node> nodes;
    // nodes.reserve(testOp.getBody()->getOperations().size());
    // for (auto [i, op] : llvm::enumerate(*testOp.getBody())) {
    //   auto node = nodes.emplace_back(op);
    //   if (op.use_empty())
    //     worklist.push_back(node);
    // }
    for (auto &op : *testOp.getBody()) {
      // auto *curr = worklist.back();
      // if (visited.contains(curr)) {
      //   worklist.pop_back();
      //   continue;
      // }
      nextOp = op.getNextNode();

      if (op.getNumRegions() != 0)
        return op.emitOpError("nested regions not supported");

      // bool addedSomething = false;
      // for (auto val : curr->getOperands()) {
      //   if (state.contains(val))
      //     continue;

      //   auto *defOp = val.getDefiningOp();
      //   assert(defOp && "cannot be a BlockArgument here");
      //   if (!visited.contains(defOp)) {
      //     worklist.push_back(defOp);
      //     addedSomething = true;
      //   }
      // }

      // if (addedSomething)
      //   continue;

      auto addToWorklist = [&](Operation *op) {
        // if (op->use_empty())
        //   worklist.push_front(op);
      };
      auto result = dispatchOpVisitor(&op, addToWorklist);
      if (failed(result))
        return failure();

      LLVM_DEBUG({
        llvm::dbgs() << "Elaborating " << op << " to\n[";

        llvm::interleaveComma(op.getResults(), llvm::dbgs(), [&](Value res) {
          llvm::dbgs() << get(res);
        });

        llvm::dbgs() << "]\n\n";
      });

      if (*result == DeletionKind::Delete)
        toDelete.push_back(&op);

      // visited.insert(curr);
      // worklist.pop_back();
    }

    // FIXME: this assumes that we didn't query the opaque value from an
    // interpreted elaborator value in a way that it can remain used in the IR.
    for (auto *op : llvm::reverse(toDelete)) {
      // op->dropAllUses();
      op->erase();
    }

    // Reduce max memory consumption and make sure the values cannot be accessed
    // anymore because we deleted the ops above.
    state.clear();
    materializer.clear();

    return success();
  }

  // LogicalResult elaborate(TestOp testOp) {
  //   LLVM_DEBUG(llvm::dbgs()
  //              << "\n=== Elaborating Test @" << testOp.getSymName() << "\n\n");

  //   DenseSet<Operation *> visited;
  //   std::deque<Operation *> worklist;
  //   DenseSet<Operation *> toDelete;
  //   for (auto &op : *testOp.getBody())
  //     if (op.use_empty())
  //       worklist.push_back(&op);

  //   while (!worklist.empty()) {
  //     auto *curr = worklist.back();
  //     if (visited.contains(curr)) {
  //       worklist.pop_back();
  //       continue;
  //     }

  //     if (curr->getNumRegions() != 0)
  //       return curr->emitOpError("nested regions not supported");

  //     bool addedSomething = false;
  //     for (auto val : curr->getOperands()) {
  //       if (state.contains(val))
  //         continue;

  //       auto *defOp = val.getDefiningOp();
  //       assert(defOp && "cannot be a BlockArgument here");
  //       if (!visited.contains(defOp)) {
  //         worklist.push_back(defOp);
  //         addedSomething = true;
  //       }
  //     }

  //     if (addedSomething)
  //       continue;

  //     auto addToWorklist = [&](Operation *op) {
  //       if (op->use_empty())
  //         worklist.push_front(op);
  //     };
  //     auto result = dispatchOpVisitor(curr, addToWorklist);
  //     if (failed(result))
  //       return failure();

  //     LLVM_DEBUG({
  //       llvm::dbgs() << "Elaborating " << *curr << " to\n[";

  //       llvm::interleaveComma(curr->getResults(), llvm::dbgs(), [&](Value res) {
  //         if (state.contains(res))
  //           llvm::dbgs() << state.at(res);
  //         else
  //           llvm::dbgs() << "unknown";
  //       });

  //       llvm::dbgs() << "]\n\n";
  //     });

  //     if (*result == DeletionKind::Delete)
  //       toDelete.insert(curr);

  //     visited.insert(curr);
  //     worklist.pop_back();
  //   }

  //   // FIXME: this assumes that we didn't query the opaque value from an
  //   // interpreted elaborator value in a way that it can remain used in the IR.
  //   for (auto *op : toDelete) {
  //     op->dropAllUses();
  //     op->erase();
  //   }

  //   // Reduce max memory consumption and make sure the values cannot be accessed
  //   // anymore because we deleted the ops above.
  //   state.clear();
  //   materializer.clear();

  //   return success();
  // }

private:
  std::mt19937 rng;

  Internalizer internalizer;

  // A map from SSA values to a pointer of an interned elaborator value.
  DenseMap<Value, ElaboratorValue> state;
  std::pair<Value, ElaboratorValue> stateCache;
  Operation *nextOp = nullptr;

  // Allows us to materialize ElaboratorValues to the IR operations necessary to
  // obtain an SSA value representing that elaborated value.
  Materializer materializer;
};
} // namespace

//===----------------------------------------------------------------------===//
// Elaborator Pass
//===----------------------------------------------------------------------===//

namespace {
struct ElaborationPass
    : public rtg::impl::ElaborationPassBase<ElaborationPass> {
  using Base::Base;

  void runOnOperation() override;
  void cloneTargetsIntoTests(SymbolTable &table);
};
} // namespace

void ElaborationPass::runOnOperation() {
  auto moduleOp = getOperation();
  SymbolTable table(moduleOp);

  cloneTargetsIntoTests(table);

  std::mt19937 rng(seed);
  Elaborator elaborator(table, rng);
  for (auto testOp : moduleOp.getOps<TestOp>())
    if (failed(elaborator.elaborate(testOp)))
      return signalPassFailure();
}

void ElaborationPass::cloneTargetsIntoTests(SymbolTable &table) {
  auto moduleOp = getOperation();
  for (auto target : llvm::make_early_inc_range(moduleOp.getOps<TargetOp>())) {
    for (auto test : moduleOp.getOps<TestOp>()) {
      // If the test requires nothing from a target, we can always run it.
      if (test.getTarget().getEntries().empty())
        continue;

      // If the target requirements do not match, skip this test
      // TODO: allow target refinements, just not coarsening
      if (target.getTarget() != test.getTarget())
        continue;

      IRRewriter rewriter(test);
      // Create a new test for the matched target
      auto newTest = cast<TestOp>(test->clone());
      newTest.setSymName(test.getSymName().str() + "_" +
                         target.getSymName().str());
      table.insert(newTest, rewriter.getInsertionPoint());

      // Copy the target body into the newly created test
      IRMapping mapping;
      rewriter.setInsertionPointToStart(newTest.getBody());
      for (auto &op : target.getBody()->without_terminator())
        rewriter.clone(op, mapping);

      for (auto [returnVal, result] :
           llvm::zip(target.getBody()->getTerminator()->getOperands(),
                     newTest.getBody()->getArguments()))
        result.replaceAllUsesWith(mapping.lookup(returnVal));

      newTest.getBody()->eraseArguments(0,
                                        newTest.getBody()->getNumArguments());
      newTest.setTarget(DictType::get(&getContext(), {}));
    }

    target->erase();
  }

  // Erase all remaining non-matched tests.
  for (auto test : llvm::make_early_inc_range(moduleOp.getOps<TestOp>()))
    if (!test.getTarget().getEntries().empty())
      test->erase();
}
