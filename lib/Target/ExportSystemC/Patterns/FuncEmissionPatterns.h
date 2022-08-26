//===- FuncEmissionPatterns.h - Func Dialect Emission Patterns ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This exposes the emission patterns of the func dialect for registration.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_FUNCEMISSIONPATTERNS_H
#define CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_FUNCEMISSIONPATTERNS_H

#include "../EmissionPatternSupport.h"

namespace circt {
namespace ExportSystemC {

/// Register Func operation emission patterns.
void populateFuncOpEmitters(OpEmissionPatternSet &patterns,
                               MLIRContext *context);

/// Register Func type emission patterns.
void populateFuncTypeEmitters(TypeEmissionPatternSet &patterns);

} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_FUNCEMISSIONPATTERNS_H
