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

#ifndef REGISTERALLEMITTERS_H
#define REGISTERALLEMITTERS_H

#include "EmissionPattern.h"
#include "Patterns/CombEmissionPatterns.h"
#include "Patterns/HWEmissionPatterns.h"
#include "Patterns/SCFEmissionPatterns.h"
#include "Patterns/SystemCEmissionPatterns.h"

namespace circt {
namespace ExportSystemC {

/// Collects the operation emission patterns of all supported dialects.
inline void registerAllOpEmitters(OpEmissionPatternSet &patterns,
                                  MLIRContext *context) {
  populateCombEmitters(patterns, context);
  populateHWEmitters(patterns, context);
  populateSCFEmitters(patterns, context);
  populateSystemCOpEmitters(patterns, context);
}

/// Collects the type emission patterns of all supported dialects.
inline void registerAllTypeEmitters(TypeEmissionPatternSet &patterns) {
  populateHWTypeEmitters(patterns);
  populateSystemCTypeEmitters(patterns);
}

} // namespace ExportSystemC
} // namespace circt

#endif // REGISTERALLEMITTERS_H
