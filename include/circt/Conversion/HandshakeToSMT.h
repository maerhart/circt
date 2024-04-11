//===- HandshakeToSMT.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which convert the Handshake dialect SMT.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HANDSHAKETOSMT_H
#define CIRCT_CONVERSION_HANDSHAKETOSMT_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

std::unique_ptr<mlir::Pass> createHandshakeToSMTPass();

} // namespace circt

#endif // CIRCT_CONVERSION_HANDSHAKETOSMT_H
