//===- InitAllExternModels.h - CIRCT Global Extern Model Reg. ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all external models
// to the system.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_INITALLEXTERNMODELS_H_
#define CIRCT_INITALLEXTERNMODELS_H_

#include "circt/Dialect/Arc/Transforms/VectorizationInterfacesImpl.h"
#include "circt/Dialect/Comb/Transforms/VectorizationInterfacesImpl.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace circt {

inline void registerAllExternModels(mlir::DialectRegistry &registry) {
  arc::registerVectorizeOpInterfaceExternalModels(registry);
  comb::registerVectorizeOpInterfaceExternalModels(registry);
}

} // namespace circt

#endif // CIRCT_INITALLEXTERNMODELS_H_
