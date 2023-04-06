//===- ArcInterfaces.td - Interfaces used in Arc -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCINTERFACES_TD
#define CIRCT_DIALECT_ARC_ARCINTERFACES_TD

include "mlir/IR/OpBase.td"

def CallOpMutableInterface : OpInterface<"CallOpMutableInterface"> {
  let description = [{
    Can be implemented in addition to `CallOpInterface` to allow mutation of the
    call operation.
  }];
  let cppNamespace = "::circt::arc";

  let methods = [
    InterfaceMethod<[{
        Returns the operands within this call that are used as arguments to the
        callee as a mutable range.
      }],
      "::mlir::MutableOperandRange", "getArgOperandsMutable"
    >,
  ];
}

def ClockedOpMutableInterface : OpInterface<"ClockedOpMutableInterface"> {
  let description = [{
  }];
  let cppNamespace = "::circt::arc";

  let methods = [
    InterfaceMethod<[{
      }],
      "::mlir::Value", "getClock"
    >,
    InterfaceMethod<[{
      }],
      "::mlir::Value", "getNearestStrictlyDominatingClock", (ins), [{}], [{
        ClockedOpMutableInterface currClock = $_op;
        ClockedOpMutableInterface clockOp = currClock->getParentOfType<ClockedOpMutableInterface>();
        while (clockOp) {
          if (clockOp.getClock())
            return clockOp.getClock();
          clockOp = clockOp->getParentOfType<ClockedOpMutableInterface>();
        }
        return {};
      }]
    >,
    InterfaceMethod<[{
      }],
      "::mlir::MutableOperandRange", "getClockMutable"
    >,
    InterfaceMethod<[{
      }],
      "bool", "isClockOptional", (ins), [{}], [{ return false; }]
    >,
  ];

  let verify = [{
    auto clockOp = mlir::cast<ClockedOpMutableInterface>($_op);
    auto dominatingClock = clockOp.getNearestStrictlyDominatingClock();
    if (dominatingClock && clockOp.getClock() && clockOp.isClockOptional())
        return clockOp->emitOpError("must not have a clock when dominated by same clock");
    
    if (!dominatingClock && !clockOp.getClock())
        return clockOp->emitOpError("must have a clock");
      
    return mlir::success();
  }];
}

#endif // CIRCT_DIALECT_ARC_ARCINTERFACES_TD
