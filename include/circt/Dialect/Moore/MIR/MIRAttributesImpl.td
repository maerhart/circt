//===- MIRAttributesImpl.td - Moore MIR attributes impl ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Attribute implementations for the Moore MIR dialect.
//
//===----------------------------------------------------------------------===//

class MooreMIRAttribute<string name> : AttrDef<MIRDialect, name> { }

//===----------------------------------------------------------------------===//
// Integer atom types
//===----------------------------------------------------------------------===//

