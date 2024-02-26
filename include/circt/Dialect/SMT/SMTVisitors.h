//===- SMTVisitors.h - SMT Dialect Visitors ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with the SMT IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTVISITORS_H
#define CIRCT_DIALECT_SMT_SMTVISITORS_H

#include "circt/Dialect/SMT/SMTOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace smt {

/// This helps visit SMT nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class SMTOpVisitor {
public:
  ResultType dispatchSMTOpVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // Constants
            BoolConstantOp, IntConstantOp, ConstantOp,
            // Bit-vector arithmetic
            NegOp, AddOp, SubOp, MulOp, URemOp, SRemOp, UModOp, SModOp, ShlOp,
            LShrOp, AShrOp, UDivOp, SDivOp,
            // Bit-vector bitwise
            BVNotOp, BVAndOp, BVOrOp, BVXOrOp, BVNAndOp, BVNOrOp, BVXNOrOp,
            // Other bit-vector ops
            ConcatOp, ExtractOp, RepeatOp, BVCmpOp,
            // Int arithmetic
            IntAddOp, IntMulOp, IntSubOp, IntDivOp, IntModOp, IntRemOp,
            IntPowOp, IntCmpOp,
            // Core Ops
            EqOp, DistinctOp, IteOp,
            // Variable/symbol declaration
            DeclareConstOp, DeclareFuncOp, ApplyFuncOp,
            // solver interaction
            SolverCreateOp, AssertOp, CheckSatOp,
            // Boolean logic
            NotOp, AndOp, OrOp, XOrOp, ImpliesOp,
            // Arrays
            ArrayStoreOp, ArraySelectOp, ArrayBroadcastOp, ArrayDefaultOp,
            // Quantifiers
            PatternCreateOp, ForallOp, ExistsOp>([&](auto expr) -> ResultType {
          return thisCast->visitSMTOp(expr, args...);
        })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidSMTOp(op, args...);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidSMTOp(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown SMT node");
    abort();
  }

  /// This callback is invoked on any SMT operations that are not
  /// handled by the concrete visitor.
  ResultType visitUnhandledSMTOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitSMTOp(OPTYPE op, ExtraArgs... args) {                        \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##SMTOp(op,         \
                                                                   args...);   \
  }

  // Constants
  HANDLE(BoolConstantOp, Unhandled);
  HANDLE(IntConstantOp, Unhandled);
  HANDLE(ConstantOp, Unhandled);

  // Bit-vector arithmetic
  HANDLE(NegOp, Unhandled);
  HANDLE(AddOp, Unhandled);
  HANDLE(SubOp, Unhandled);
  HANDLE(MulOp, Unhandled);
  HANDLE(URemOp, Unhandled);
  HANDLE(SRemOp, Unhandled);
  HANDLE(UModOp, Unhandled);
  HANDLE(SModOp, Unhandled);
  HANDLE(ShlOp, Unhandled);
  HANDLE(LShrOp, Unhandled);
  HANDLE(AShrOp, Unhandled);
  HANDLE(UDivOp, Unhandled);
  HANDLE(SDivOp, Unhandled);

  // Bit-vector bitwise operations
  HANDLE(BVNotOp, Unhandled);
  HANDLE(BVAndOp, Unhandled);
  HANDLE(BVOrOp, Unhandled);
  HANDLE(BVXOrOp, Unhandled);
  HANDLE(BVNAndOp, Unhandled);
  HANDLE(BVNOrOp, Unhandled);
  HANDLE(BVXNOrOp, Unhandled);

  // Other bit-vector operations
  HANDLE(ConcatOp, Unhandled);
  HANDLE(ExtractOp, Unhandled);
  HANDLE(RepeatOp, Unhandled);
  HANDLE(BVCmpOp, Unhandled);

  // Int arithmetic
  HANDLE(IntAddOp, Unhandled);
  HANDLE(IntMulOp, Unhandled);
  HANDLE(IntSubOp, Unhandled);
  HANDLE(IntDivOp, Unhandled);
  HANDLE(IntModOp, Unhandled);
  HANDLE(IntRemOp, Unhandled);
  HANDLE(IntPowOp, Unhandled);

  HANDLE(IntCmpOp, Unhandled);

  HANDLE(EqOp, Unhandled);
  HANDLE(DistinctOp, Unhandled);
  HANDLE(IteOp, Unhandled);

  HANDLE(DeclareConstOp, Unhandled);
  HANDLE(DeclareFuncOp, Unhandled);
  HANDLE(ApplyFuncOp, Unhandled);

  HANDLE(SolverCreateOp, Unhandled);
  HANDLE(AssertOp, Unhandled);
  HANDLE(CheckSatOp, Unhandled);

  // Boolean logic operations
  HANDLE(NotOp, Unhandled);
  HANDLE(AndOp, Unhandled);
  HANDLE(OrOp, Unhandled);
  HANDLE(XOrOp, Unhandled);
  HANDLE(ImpliesOp, Unhandled);

  // Array operations
  HANDLE(ArrayStoreOp, Unhandled);
  HANDLE(ArraySelectOp, Unhandled);
  HANDLE(ArrayBroadcastOp, Unhandled);
  HANDLE(ArrayDefaultOp, Unhandled);

  // Quantifier operations
  HANDLE(PatternCreateOp, Unhandled);
  HANDLE(ForallOp, Unhandled);
  HANDLE(ExistsOp, Unhandled);
  HANDLE(YieldOp, Unhandled);

#undef HANDLE
};

/// This helps visit SMT types.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class SMTTypeVisitor {
public:
  ResultType dispatchSMTTypeVisitor(Type type, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Type, ResultType>(type)
        .template Case<BoolType, IntegerType, PatternType, BitVectorType,
                       SolverType, ArrayType, SMTFunctionType, SortType>(
            [&](auto expr) -> ResultType {
              return thisCast->visitSMTType(expr, args...);
            })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidSMTType(type, args...);
        });
  }

  /// This callback is invoked on any non-expression types.
  ResultType visitInvalidSMTType(Type type, ExtraArgs... args) { abort(); }

  /// This callback is invoked on any SMT type that are not
  /// handled by the concrete visitor.
  ResultType visitUnhandledSMTType(Type type, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(TYPE, KIND)                                                     \
  ResultType visitSMTType(TYPE op, ExtraArgs... args) {                        \
    return static_cast<ConcreteType *>(this)->visit##KIND##SMTType(op,         \
                                                                   args...);   \
  }

  HANDLE(BoolType, Unhandled);
  HANDLE(IntegerType, Unhandled);
  HANDLE(PatternType, Unhandled);
  HANDLE(BitVectorType, Unhandled);
  HANDLE(SolverType, Unhandled);
  HANDLE(ArrayType, Unhandled);
  HANDLE(SMTFunctionType, Unhandled);
  HANDLE(SortType, Unhandled);

#undef HANDLE
};

} // namespace smt
} // namespace circt

#endif // CIRCT_DIALECT_SMT_SMTVISITORS_H
