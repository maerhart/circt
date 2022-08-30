//===- RegisterAllEmitters.h - Register all emitters to ExportSystemC -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This registers the all the emitters of various dialects to the
// ExportSystemC pass.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_TARGET_EXPORTSYSTEMC_REGISTERALLEMITTERS_H
#define CIRCT_TARGET_EXPORTSYSTEMC_REGISTERALLEMITTERS_H

#include "Patterns/EmitCEmissionPatterns.h"
#include "Patterns/HWEmissionPatterns.h"
#include "Patterns/SystemCEmissionPatterns.h"
#include "Patterns/FuncEmissionPatterns.h"

namespace circt {
namespace ExportSystemC {

/// Collects the operation emission patterns of all supported dialects.
inline void registerAllOpEmitters(OpEmissionPatternSet &patterns,
                                  MLIRContext *context) {
  populateHWEmitters(patterns, context);
  populateSystemCOpEmitters(patterns, context);
  populateEmitCOpEmitters(patterns, context);
  populateFuncOpEmitters(patterns, context);
}

/// Collects the type emission patterns of all supported dialects.
inline void registerAllTypeEmitters(TypeEmissionPatternSet &patterns) {
  populateHWTypeEmitters(patterns);
  populateSystemCTypeEmitters(patterns);
  populateEmitCTypeEmitters(patterns);
  populateFuncTypeEmitters(patterns);
}

} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_REGISTERALLEMITTERS_H
