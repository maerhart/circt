//===- MIRStatements.td - Moore MIR statements ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for Moore MIR statements.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Assignment Statements
//===----------------------------------------------------------------------===//

def BlockingAssignOp : MIROp<"blocking_assign", [TypesMatchWith<"", "src", "dest", "moore::LValueType::get($_self.cast<RValueType>().getNestedType())">]> {
  let summary = "Blocking assignment";
  let description = [{
    A SystemVerilog assignment statement 'x = y;'.
    These occur in module scope.  See SV Spec 10.3.2.
  }];
  let arguments = (ins MooreLValueType:$dest, MooreRValueType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `=` $src  attr-dict `:` custom<RValueType>(type($src))
  }];
}

def NonBlockingAssignOp : MIROp<"assign", [TypesMatchWith<"", "src", "dest", "moore::LValueType::get($_self.cast<RValueType>().getNestedType())">]> {
  let summary = "Continuous assignment";
  let description = [{
    A SystemVerilog assignment statement 'x = y;'.
    These occur in module scope.  See SV Spec 10.3.2.
  }];
  let arguments = (ins MooreLValueType:$dest, MooreRValueType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `=` $src  attr-dict `:` custom<RValueType>(type($src))
  }];
}

def NonBlockingDelayedAssignOp : MIROp<"delayed_assign", [TypesMatchWith<"", "src", "dest", "moore::LValueType::get($_self.cast<RValueType>().getNestedType())">]> {
  let summary = "Non-blocking delayed assignment";
  let description = [{
    A SystemVerilog assignment statement 'x = y;'.
    These occur in module scope.  See SV Spec 10.3.2.
  }];
  let arguments = (ins MooreLValueType:$dest, MooreRValueType:$src, MooreRValueType:$delay);
  let results = (outs);
  let assemblyFormat = [{
    $dest `=` $src `after` $delay attr-dict `:` custom<RValueType>(type($src)) `,` custom<RValueType>(type($delay))
  }];
}
