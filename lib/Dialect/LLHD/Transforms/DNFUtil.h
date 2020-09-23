//===- PassDetails.h - LLHD pass class details ------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LLHD_TRANSFORMS_DNFUTIL_H
#define DIALECT_LLHD_TRANSFORMS_DNFUTIL_H

#include "mlir/IR/Builders.h"

namespace mlir {
namespace llhd {

/// Represents the type of a node in the DNF tree
enum class DnfNodeType {
  Const,
  Val,
  And,
  Or,
};

/// A tree representing a DNF formula
struct Dnf {
  Dnf() = delete;

  /// Create a copy of a DNF tree
  Dnf(const Dnf &e) : type(e.type), value(e.value), constant(e.constant), inv(e.inv) {
    for (auto &child : e.children) {
      children.push_back(std::make_unique<Dnf>(*child));
    }
  }

  /// Create a new Constant leaf node
  Dnf(bool constant, bool inv)
      : type(DnfNodeType::Const), constant(constant), inv(inv) {}

  /// Create a new value leaf node, this should either be a sample of a signal
  /// or an opaque value
  Dnf(Value val, bool inv) : type(DnfNodeType::Val), value(val), inv(inv) {}

  /// Create a new Dnf tree from two children connected bei either an AND or OR,
  /// given that the two children are both a DNF tree representing a valid DNF
  /// formula, the newly created DNF tree will also be a valid DNF formula
  Dnf(DnfNodeType ty, std::unique_ptr<Dnf> lhs, std::unique_ptr<Dnf> rhs);

  bool isConst() { return type == DnfNodeType::Const; }
  bool isVal() { return type == DnfNodeType::Val; }
  bool isLeaf() { return isConst() || isVal(); }
  bool isAnd() { return type == DnfNodeType::And; }
  bool isOr() { return type == DnfNodeType::Or; }
  bool isNegatedVal() { return isVal() && inv; }
  bool isProbedSignal();
  bool getConst() {
    assert(isConst() && "node has to be Const to return the constant");
    return constant != inv;
  }
  Value getProbedSignal();

  /// Simplify the top-level DNF (e.g. remove duplicates, constant true in an
  /// AND, etc.). Does not similify child expressions recursively.
  void simplifyExpression();

  /// Create the LLHD and standard dialect operations necessary to represent
  /// this DNF formula.
  Value buildOperations(OpBuilder &builder);

  DnfNodeType type;
  std::vector<std::unique_ptr<Dnf>> children;
  Value value;
  bool constant;
  bool inv;
};

/// This describes an algorithm to find a boolean expression which evaluates to
/// TRUE iff the control flow path will go through a given target basic block
/// starting at a given source basic block. The source basic block has to
/// dominate the target basic block.
/// Objective: create boolean expression from conditions of the conditional
/// branches in the set of basic blocks B containing all blocks b for which
///   (i)  b can be reached from the source block and
///   (ii) the target block can be reached from b
/// such that the target block will always be executed iff this boolean
/// expression evaluates to TRUE
/// Procedure:
///   (1) only consider blocks in B by starting at the target block and going up
///       in the CFG
///   (2) the condition for a block b is created by taking the disjuction of the
///       conjunction of the condition of all predecessor blocks and the
///       condition that has to hold to reach b from the predecessor
///   (3) The condition of the predecessor can be created by calling a recursive
///       function
///   (4) To reduce the computational complexity we memorize the conditions for
///       each basic block in order to receive it without calling the recursive
///       function again
std::unique_ptr<Dnf> getBooleanExprFromSourceToTarget(Block *source,
                                                      Block *target);

mlir::Value getBooleanExprFromSourceToTargetNonDnf(OpBuilder &builder, Block *source,
                                                   Block *target);

mlir::Value getBooleanExprFromSourceToTargetNonDnf(OpBuilder &builder, Block *source,
                                                   Block *target, DenseMap<Block *, Value> &mem);
} // namespace llhd
} // namespace mlir

#endif // DIALECT_LLHD_TRANSFORMS_DNFUTIL_H
