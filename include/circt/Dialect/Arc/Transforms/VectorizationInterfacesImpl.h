//===- VectorizationInterfacesImpl.h - Impl. of VectorizationInterfaces ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_VECTORIZATIONINTERFACESIMPL_H
#define CIRCT_DIALECT_ARC_VECTORIZATIONINTERFACESIMPL_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace circt {
namespace arc {
void registerVectorizeOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);
} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_VECTORIZATIONINTERFACESIMPL_H
