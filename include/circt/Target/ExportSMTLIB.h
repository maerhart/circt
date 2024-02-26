//===- ExportSMTLIB.h - SMT-LIB Exporter ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the SMT-LIB emitter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_EXPORTSMTLIB_H
#define CIRCT_TARGET_EXPORTSMTLIB_H

#include "circt/Support/LLVM.h"

namespace circt {
namespace ExportSMTLIB {

///
struct SMTEmissionOptions {
  bool printBitVectorsInHex = true;
};

///
LogicalResult
exportSMTLIB(Operation *module, llvm::raw_ostream &os,
             const SMTEmissionOptions &options = SMTEmissionOptions());

///
void registerExportSMTLIBTranslation();

} // namespace ExportSMTLIB
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSMTLIB_H
