//===- SMTOps.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SMT/SMTOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/APSInt.h"

using namespace circt;
using namespace smt;
using namespace mlir;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  if (getValue().getType() != getType())
    return emitError(
        "smt.bv.constant attribute bitwidth doesn't match return type");

  return success();
}

LogicalResult ConstantOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(
      cast<TypedAttr>(attributes.get("value")).getType());
  return success();
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c' << getValue().getValue() << "_bv"
              << cast<BitVectorType>(getType()).getWidth();
  setNameFn(getResult(), specialName.str());
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// BoolConstantOp
//===----------------------------------------------------------------------===//

void BoolConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << (getValue() ? "true" : "false");
  setNameFn(getResult(), specialName.str());
}

OpFoldResult BoolConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// IntConstantOp
//===----------------------------------------------------------------------===//

void IntConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "c" << getValue();
  setNameFn(getResult(), specialName.str());
}

OpFoldResult IntConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

void IntConstantOp::print(OpAsmPrinter &p) {
  p << " " << getValue();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

ParseResult IntConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  APInt value;
  auto res = parser.parseOptionalInteger(value);
  if (!res.has_value())
    return failure();

  result.addAttribute("value",
                      IntegerAttr::get(parser.getContext(), APSInt(value)));

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes(smt::IntegerType::get(parser.getContext()));
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(BitVectorType::get(
      context, cast<BitVectorType>(operands[0].getType()).getWidth() +
                   cast<BitVectorType>(operands[1].getType()).getWidth()));
  return success();
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::verify() {
  if (getStart() + getType().getWidth() >
      cast<BitVectorType>(getInput().getType()).getWidth())
    return emitOpError("slice too big");
  return success();
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//

LogicalResult RepeatOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(BitVectorType::get(
      context, cast<BitVectorType>(operands[0].getType()).getWidth() *
                   cast<IntegerAttr>(attributes.get("count")).getInt()));
  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/SMT/SMT.cpp.inc"
