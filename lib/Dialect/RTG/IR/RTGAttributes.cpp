//===- RTGAttributes.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "circt/Dialect/RTG/IR/RTGTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace rtg;

//===----------------------------------------------------------------------===//
// IPInt Implementation
//===----------------------------------------------------------------------===//

IPInt::IPInt(llvm::APSInt value) : value(normalize(std::move(value))) {}

IPInt::IPInt(int64_t val)
    : value(normalize(llvm::APSInt(llvm::APInt(64, val, true), false))) {}

llvm::APSInt IPInt::normalize(llvm::APSInt value) {
  // Compute the minimum bitwidth needed to represent this value
  unsigned minBitWidth = value.getSignificantBits();

  // Ensure at least 1 bit
  if (minBitWidth == 0)
    minBitWidth = 1;

  // Truncate to minimum bitwidth if needed
  if (value.getBitWidth() > minBitWidth)
    value = value.truncSSat(minBitWidth);

  return value;
}

std::pair<llvm::APSInt, llvm::APSInt>
IPInt::extendToCommonWidth(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  // Calculate the width needed (use the larger of the two)
  unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());

  // Sign-extend both operands to the common width
  llvm::APSInt extLhs = lhs;
  llvm::APSInt extRhs = rhs;

  if (extLhs.getBitWidth() < width)
    extLhs = extLhs.extend(width);
  if (extRhs.getBitWidth() < width)
    extRhs = extRhs.extend(width);

  return {extLhs, extRhs};
}

IPInt IPInt::add(const IPInt &rhs) const {
  // For addition: max(width_a, width_b) + 1 bit needed to prevent overflow
  unsigned maxWidth = std::max(value.getBitWidth(), rhs.value.getBitWidth());
  unsigned resultWidth = maxWidth + 1;

  llvm::APSInt lhsExt = value.extend(resultWidth);
  llvm::APSInt rhsExt = rhs.value.extend(resultWidth);

  return IPInt(lhsExt + rhsExt);
}

IPInt IPInt::sub(const IPInt &rhs) const {
  // For subtraction: max(width_a, width_b) + 1 bit needed to prevent overflow
  unsigned maxWidth = std::max(value.getBitWidth(), rhs.value.getBitWidth());
  unsigned resultWidth = maxWidth + 1;

  llvm::APSInt lhsExt = value.extend(resultWidth);
  llvm::APSInt rhsExt = rhs.value.extend(resultWidth);

  return IPInt(lhsExt - rhsExt);
}

IPInt IPInt::mul(const IPInt &rhs) const {
  // For multiplication: width_a + width_b bits needed to prevent overflow
  unsigned resultWidth = value.getBitWidth() + rhs.value.getBitWidth();

  llvm::APSInt lhsExt = value.extend(resultWidth);
  llvm::APSInt rhsExt = rhs.value.extend(resultWidth);

  return IPInt(lhsExt * rhsExt);
}

IPInt IPInt::sdiv(const IPInt &rhs) const {
  auto [lhsExt, rhsExt] = extendToCommonWidth(value, rhs.value);
  llvm::APSInt result(lhsExt.sdiv(rhsExt), false);
  return IPInt(result);
}

IPInt IPInt::smod(const IPInt &rhs) const {
  auto [lhsExt, rhsExt] = extendToCommonWidth(value, rhs.value);
  llvm::APSInt result(lhsExt.srem(rhsExt), false);
  return IPInt(result);
}

IPInt IPInt::pow(const IPInt &exponent) const {
  // Exponent must be non-negative
  assert(exponent.value.isNonNegative() && "Exponent must be non-negative");

  // Handle special cases
  if (exponent.value.isZero())
    return IPInt(llvm::APSInt(llvm::APInt(1, 1), false)); // Any number^0 = 1

  if (value.isZero())
    return IPInt(llvm::APSInt(llvm::APInt(1, 0), false)); // 0^n = 0 (n > 0)

  if (exponent.value.isOne())
    return *this; // base^1 = base

  // For power operation: approximate result bitwidth as base_width * exponent
  // This is conservative but prevents overflow
  uint64_t exp = exponent.value.getZExtValue();
  unsigned resultWidth = value.getBitWidth() * exp;

  // Perform exponentiation by repeated multiplication
  llvm::APSInt base = value.extend(resultWidth);
  llvm::APSInt result(llvm::APInt(resultWidth, 1), false);

  for (uint64_t i = 0; i < exp; ++i) {
    result *= base;
  }

  return IPInt(result);
}

IPInt IPInt::and_(const IPInt &rhs) const {
  auto [lhsExt, rhsExt] = extendToCommonWidth(value, rhs.value);
  return IPInt(lhsExt & rhsExt);
}

IPInt IPInt::or_(const IPInt &rhs) const {
  auto [lhsExt, rhsExt] = extendToCommonWidth(value, rhs.value);
  return IPInt(lhsExt | rhsExt);
}

IPInt IPInt::xor_(const IPInt &rhs) const {
  auto [lhsExt, rhsExt] = extendToCommonWidth(value, rhs.value);
  return IPInt(lhsExt ^ rhsExt);
}

IPInt IPInt::shl(const IPInt &rhs) const {
  // For shift operations, we need to ensure the result has enough bits
  // Extend LHS by the shift amount to prevent overflow
  llvm::APSInt lhsExt = value;
  uint64_t shiftAmt = rhs.value.getExtValue();
  lhsExt = lhsExt.extend(lhsExt.getBitWidth() + shiftAmt);
  return IPInt(lhsExt << shiftAmt);
}

IPInt IPInt::ashr(const IPInt &rhs) const {
  uint64_t shiftAmt = rhs.value.getExtValue();
  llvm::APSInt result(value.ashr(shiftAmt), false);
  return IPInt(result);
}

bool IPInt::eq(const IPInt &rhs) const {
  auto [lhsExt, rhsExt] = extendToCommonWidth(value, rhs.value);
  return lhsExt == rhsExt;
}

bool IPInt::ne(const IPInt &rhs) const { return !eq(rhs); }

bool IPInt::slt(const IPInt &rhs) const {
  auto [lhsExt, rhsExt] = extendToCommonWidth(value, rhs.value);
  return lhsExt < rhsExt;
}

bool IPInt::sle(const IPInt &rhs) const {
  auto [lhsExt, rhsExt] = extendToCommonWidth(value, rhs.value);
  return lhsExt <= rhsExt;
}

bool IPInt::sgt(const IPInt &rhs) const {
  auto [lhsExt, rhsExt] = extendToCommonWidth(value, rhs.value);
  return lhsExt > rhsExt;
}

bool IPInt::sge(const IPInt &rhs) const {
  auto [lhsExt, rhsExt] = extendToCommonWidth(value, rhs.value);
  return lhsExt >= rhsExt;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace circt {
namespace rtg {

// Hash function for IPInt (must be in rtg namespace for ADL)
llvm::hash_code hash_value(const IPInt &val) {
  return llvm::hash_value(val.getValue());
}

} // namespace rtg
} // namespace circt

namespace llvm {
template <typename T>
// NOLINTNEXTLINE(readability-identifier-naming)
llvm::hash_code hash_value(const DenseSet<T> &set) {
  // TODO: improve collision resistance
  unsigned hash = 0;
  for (auto element : set)
    hash ^= element;
  return hash;
}

template <typename K, typename V>
// NOLINTNEXTLINE(readability-identifier-naming)
llvm::hash_code hash_value(const DenseMap<K, V> &map) {
  // TODO: improve collision resistance
  unsigned hash = 0;
  for (auto [key, value] : map)
    hash ^= (key ^ value);
  return hash;
}
} // namespace llvm

//===----------------------------------------------------------------------===//
// SetAttr
//===----------------------------------------------------------------------===//

LogicalResult
SetAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                rtg::SetType type, const DenseSet<TypedAttr> *elements) {

  // check that all elements have the right type
  // iterating over the set is fine here because the iteration order is not
  // visible to the outside (it would not be fine to print the earliest invalid
  // element)
  if (!llvm::all_of(*elements, [&](auto element) {
        return element.getType() == type.getElementType();
      })) {
    return emitError() << "all elements must be of the set element type "
                       << type.getElementType();
  }

  return success();
}

Attribute SetAttr::parse(AsmParser &odsParser, Type odsType) {
  DenseSet<TypedAttr> elements;
  Type elementType;
  if (odsParser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::LessGreater,
                                        [&]() {
                                          TypedAttr element;
                                          if (odsParser.parseAttribute(element))
                                            return failure();
                                          elements.insert(element);
                                          elementType = element.getType();
                                          return success();
                                        }))
    return {};

  auto setType = llvm::dyn_cast_or_null<SetType>(odsType);
  if (odsType && !setType) {
    odsParser.emitError(odsParser.getNameLoc())
        << "type must be a an '!rtg.set' type";
    return {};
  }

  if (!setType && elements.empty()) {
    odsParser.emitError(odsParser.getNameLoc())
        << "type must be explicitly provided: cannot infer set element type "
           "from empty set";
    return {};
  }

  if (!setType && !elements.empty())
    setType = SetType::get(elementType);

  return SetAttr::getChecked(
      odsParser.getEncodedSourceLoc(odsParser.getNameLoc()),
      odsParser.getContext(), setType, &elements);
}

void SetAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  // Sort elements lexicographically by their printed string representation
  SmallVector<std::string> sortedElements;
  for (auto element : *getElements()) {
    std::string &elementStr = sortedElements.emplace_back();
    llvm::raw_string_ostream elementOS(elementStr);
    element.print(elementOS);
  }
  llvm::sort(sortedElements);
  llvm::interleaveComma(sortedElements, odsPrinter);
  odsPrinter << ">";
}

//===----------------------------------------------------------------------===//
// MapAttr
//===----------------------------------------------------------------------===//

LogicalResult
MapAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                rtg::MapType type,
                const DenseMap<TypedAttr, TypedAttr> *entries) {

  // check that all keys and values have the right type
  if (!llvm::all_of(*entries, [&](auto entry) {
        return entry.first.getType() == type.getKeyType() &&
               entry.second.getType() == type.getValueType();
      })) {
    return emitError() << "all keys must be of type " << type.getKeyType()
                       << " and all values must be of type "
                       << type.getValueType();
  }

  return success();
}

Attribute MapAttr::parse(AsmParser &odsParser, Type odsType) {
  DenseMap<TypedAttr, TypedAttr> entries;
  Type keyType, valueType;
  if (odsParser.parseCommaSeparatedList(
          mlir::AsmParser::Delimiter::LessGreater, [&]() {
            TypedAttr key, value;
            if (odsParser.parseAttribute(key) || odsParser.parseArrow() ||
                odsParser.parseAttribute(value))
              return failure();
            entries.insert({key, value});
            keyType = key.getType();
            valueType = value.getType();
            return success();
          }))
    return {};

  auto mapType = llvm::dyn_cast_or_null<MapType>(odsType);
  if (odsType && !mapType) {
    odsParser.emitError(odsParser.getNameLoc())
        << "type must be an '!rtg.map' type";
    return {};
  }

  if (!mapType && entries.empty()) {
    odsParser.emitError(odsParser.getNameLoc())
        << "type must be explicitly provided: cannot infer map key and value "
           "types from empty map";
    return {};
  }

  if (!mapType && !entries.empty())
    mapType = MapType::get(keyType, valueType);

  return MapAttr::getChecked(
      odsParser.getEncodedSourceLoc(odsParser.getNameLoc()),
      odsParser.getContext(), mapType, &entries);
}

void MapAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  // Sort entries lexicographically by their printed string representation
  SmallVector<std::pair<std::string, std::string>> sortedEntries;
  for (auto [key, value] : *getEntries()) {
    std::string keyStr, valueStr;
    llvm::raw_string_ostream keyOS(keyStr);
    llvm::raw_string_ostream valueOS(valueStr);
    key.print(keyOS);
    value.print(valueOS);
    sortedEntries.emplace_back(std::move(keyStr), std::move(valueStr));
  }
  llvm::sort(sortedEntries);
  llvm::interleaveComma(sortedEntries, odsPrinter, [&](auto &entry) {
    odsPrinter << entry.first << " -> " << entry.second;
  });
  odsPrinter << ">";
}

//===----------------------------------------------------------------------===//
// TupleAttr
//===----------------------------------------------------------------------===//

Type TupleAttr::getType() const {
  SmallVector<Type> elementTypes(llvm::map_range(
      getElements(), [](auto element) { return element.getType(); }));
  return TupleType::get(getContext(), elementTypes);
}

//===----------------------------------------------------------------------===//
// VirtualRegisterConfigAttr
//===----------------------------------------------------------------------===//

Type VirtualRegisterConfigAttr::getType() const {
  return getAllowedRegs()[0].getType();
}

LogicalResult VirtualRegisterConfigAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ArrayRef<rtg::RegisterAttrInterface> allowedRegs) {
  if (allowedRegs.empty())
    return emitError() << "must have at least one allowed register";

  if (!llvm::all_of(allowedRegs, [&](auto reg) {
        return reg.getType() == allowedRegs[0].getType();
      })) {
    return emitError() << "all allowed registers must be of the same type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LabelAttr
//===----------------------------------------------------------------------===//

Type LabelAttr::getType() const { return LabelType::get(getContext()); }

//===----------------------------------------------------------------------===//
// IntAttr Storage
//===----------------------------------------------------------------------===//

namespace circt {
namespace rtg {
namespace detail {
struct IntAttrStorage : public mlir::AttributeStorage {
  using KeyTy = IPInt;
  IntAttrStorage(IPInt value) : value(std::move(value)) {}

  KeyTy getAsKey() const { return value; }

  bool operator==(const KeyTy &key) const { return value == key; }

  static llvm::hash_code hashKey(const KeyTy &key) { return hash_value(key); }

  static IntAttrStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                   KeyTy &&key) {
    return new (allocator.allocate<IntAttrStorage>())
        IntAttrStorage(std::move(key));
  }

  IPInt value;
};
} // namespace detail
} // namespace rtg
} // namespace circt

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/RTG/IR/RTGAttributes.cpp.inc"

void RTGDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/RTG/IR/RTGAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// IntAttr
//===----------------------------------------------------------------------===//

Type IntAttr::getType() const { return IntType::get(getContext()); }

IPInt IntAttr::getValue() const { return getImpl()->value; }

Attribute IntAttr::parse(AsmParser &odsParser, Type odsType) {
  llvm::APInt val;
  if (odsParser.parseLess() || odsParser.parseInteger(val) ||
      odsParser.parseGreater())
    return {};

  // Convert APInt to APSInt (always treat as signed)
  llvm::APSInt sval(val, false);
  return IntAttr::get(odsParser.getContext(), IPInt(sval));
}

void IntAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<" << getValue().getValue() << ">";
}
