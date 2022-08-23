//===- InteropOpInterfaceImpl.h - Impl. of InteropOpInterface -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_INTEROPOPINTERFACEIMPL_H
#define CIRCT_DIALECT_SYSTEMC_INTEROPOPINTERFACEIMPL_H

// Forward declarations.
namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace circt {
namespace systemc {
void registerInteropOpInterfaceExternalModels(mlir::DialectRegistry &registry);
} // namespace systemc
} // namespace circt

#endif // CIRCT_DIALECT_SYSTEMC_INTEROPOPINTERFACEIMPL_H
