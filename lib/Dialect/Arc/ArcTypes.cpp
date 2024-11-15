//===- ArcTypes.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcTypes.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace arc;
using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Arc/ArcTypes.cpp.inc"

static int guessLLVMBitWidth(Type type) {
  if (llvm::isa<seq::ClockType>(type))
    return 1;
  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
    unsigned width = guessLLVMBitWidth(arrayType.getElementType());
    width = llvm::alignToPowerOf2(width, llvm::bit_ceil(std::min(width, 16U)));
    return width * arrayType.getNumElements();
  }
  if (auto structType = dyn_cast<hw::StructType>(type)) {
    unsigned width = 0;
    for (auto element : structType.getElements()) {
      unsigned elementWidth = guessLLVMBitWidth(element.type);
      elementWidth = (elementWidth + 7) / 8 * 8;
      width += elementWidth;
    }
    return width;
  }
  return hw::getBitWidth(type);
}

unsigned StateType::getBitWidth() { return guessLLVMBitWidth(getType()); }

LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (guessLLVMBitWidth(innerType) < 0)
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}

unsigned MemoryType::getStride() {
  unsigned stride = (getWordType().getWidth() + 7) / 8;
  return llvm::alignToPowerOf2(stride, llvm::bit_ceil(std::min(stride, 16U)));
}

void ArcDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Arc/ArcTypes.cpp.inc"
      >();
}
