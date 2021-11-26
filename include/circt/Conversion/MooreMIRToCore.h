//===- MooreMIRToCore.h - Moore MIR to Core pass entry point ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the MooreMIRToCore pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_MOOREMIRTOCORE_H
#define CIRCT_CONVERSION_MOOREMIRTOCORE_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

/// Create an Moore MIR to Comb/HW/LLHD conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createConvertMooreMIRToCorePass();

} // namespace circt

#endif // CIRCT_CONVERSION_MOOREMIRTOCORE_H
