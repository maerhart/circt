//===- LLHDTransformsUtil.h - Helper functions for LLHD passes --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A collection of helper functions for various LLHD transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLHDTRANSFORMUTIL
#define LLHDTRANSFORMUTIL

#include "circt/Support/LLVM.h"

namespace circt {
namespace llhd {

/// Explore all paths from the 'driveBlock' to the 'dominator' block and
/// construct a boolean expression at the current insertion point of 'builder'
/// to represent all those paths.
Value getBranchDecisionsFromDominatorToTarget(OpBuilder &builder,
                                              Block *driveBlock,
                                              Block *dominator,
                                              DenseMap<Block *, Value> &mem);

} // namespace llhd
} // namespace circt

#endif // LLHDTRANSFORMUTIL
