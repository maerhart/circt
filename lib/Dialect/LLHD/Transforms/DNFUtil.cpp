//===- DNFUtil.cpp - Implement DNF utility class --------------------------===//
//
//===----------------------------------------------------------------------===//

#include "DNFUtil.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::llhd;

mlir::llhd::Dnf::Dnf(DnfNodeType ty, std::unique_ptr<Dnf> lhs,
                     std::unique_ptr<Dnf> rhs)
    : type(ty) {
  assert(lhs && rhs && "Passed expressions should not be a nullptr!");

  // Perform the necessary tree rotations, etc. to merge the two DNF formulas
  // into one with the given operation type
  switch (ty) {
  case DnfNodeType::And: {
    if (lhs->isOr() && rhs->isOr()) {
      // (A v B) ^ (C v D) => (A ^ C) v (B ^ C) v (A ^ D) v (B ^ D)
      type = DnfNodeType::Or;
      for (auto &&lhsChild : lhs->children) {
        for (auto &&rhsChild : rhs->children) {
          children.push_back(std::make_unique<Dnf>(
              DnfNodeType::And, std::make_unique<Dnf>(*lhsChild),
              std::make_unique<Dnf>(*rhsChild)));
        }
      }
    } else if (lhs->isOr() && rhs->isAnd()) {
      // (A v B) ^ (C ^ D) => (A ^ C ^ D) v (B ^ C ^ D)
      type = DnfNodeType::Or;
      for (auto &&lhsChild : lhs->children) {
        children.push_back(std::make_unique<Dnf>(DnfNodeType::And,
                                                 std::move(lhsChild),
                                                 std::make_unique<Dnf>(*rhs)));
      }
    } else if (lhs->isAnd() && rhs->isOr()) {
      // (A ^ B) ^ (C v D) => (A ^ B ^ C) v (A ^ B ^ D)
      type = DnfNodeType::Or;
      for (auto &&rhsChild : rhs->children) {
        children.push_back(std::make_unique<Dnf>(DnfNodeType::And,
                                                 std::make_unique<Dnf>(*lhs),
                                                 std::move(rhsChild)));
      }
    } else if (lhs->isAnd() && rhs->isAnd()) {
      // (A ^ B) ^ (C ^ D) => A ^ B ^ C ^ D
      std::move(begin(lhs->children), end(lhs->children),
                std::back_inserter(children));
      std::move(begin(rhs->children), end(rhs->children),
                std::back_inserter(children));
    } else if (lhs->isLeaf() && rhs->isAnd()) {
      // A ^ (B ^ C) => A ^ B ^ C
      children.push_back(std::move(lhs));
      std::move(begin(rhs->children), end(rhs->children),
                std::back_inserter(children));
    } else if (lhs->isAnd() && rhs->isLeaf()) {
      // (A ^ B) ^ C => A ^ B ^ C
      std::move(begin(lhs->children), end(lhs->children),
                std::back_inserter(children));
      children.push_back(std::move(rhs));
    } else if (lhs->isLeaf() && rhs->isLeaf()) {
      // A ^ B
      children.push_back(std::move(lhs));
      children.push_back(std::move(rhs));
    } else if (lhs->isOr() && rhs->isLeaf()) {
      // (A v B) ^ C => (A ^ C) v (B ^ C)
      type = DnfNodeType::Or;
      // std::move(begin(lhs->children), end(lhs->children),
      // std::back_inserter(children));
      for (auto &&lhsChild : lhs->children) {
        children.push_back(std::make_unique<Dnf>(DnfNodeType::And,
                                                 std::move(lhsChild),
                                                 std::make_unique<Dnf>(*rhs)));
      }
    } else if (lhs->isLeaf() && rhs->isOr()) {
      // A v (B ^ C) => (A ^ B) v (A ^ C)
      type = DnfNodeType::Or;
      // std::move(begin(rhs->children), end(rhs->children),
      // std::back_inserter(children));
      for (auto &&rhsChild : rhs->children) {
        children.push_back(std::make_unique<Dnf>(DnfNodeType::And,
                                                 std::make_unique<Dnf>(*lhs),
                                                 std::move(rhsChild)));
      }
    } else {
      assert(false && "Unreachable!");
    }
    break;
  }
  case DnfNodeType::Or: {
    if ((lhs->isAnd() && rhs->isAnd()) || (lhs->isLeaf() && rhs->isLeaf()) ||
        (lhs->isLeaf() && rhs->isAnd()) || (lhs->isAnd() && rhs->isLeaf())) {
      // (A ^ B) v (C ^ D)
      // or
      // A v B
      children.push_back(std::move(lhs));
      children.push_back(std::move(rhs));
    } else if (lhs->isOr() && (rhs->isAnd() || rhs->isLeaf())) {
      // (A v B) v (C ^ D) => A v B v (C ^ D)
      // or
      // (A v B) v C => A v B v C
      std::move(begin(lhs->children), end(lhs->children),
                std::back_inserter(children));
      children.push_back(std::move(rhs));
    } else if ((lhs->isAnd() || lhs->isLeaf()) && rhs->isOr()) {
      // (A ^ B) v (C v D) => (A ^ B) v C v D
      // or
      // A v (B v C) => A v B v C
      children.push_back(std::move(lhs));
      std::move(begin(rhs->children), end(rhs->children),
                std::back_inserter(children));
    } else if (lhs->isOr() && rhs->isOr()) {
      // (A v B) v (C v D) => A v B v C v D
      std::move(begin(rhs->children), end(rhs->children),
                std::back_inserter(children));
      std::move(begin(lhs->children), end(lhs->children),
                std::back_inserter(children));
    } else {
      assert(false && "Unreachable!");
    }
    break;
  }
  default: {
    assert(false &&
           "To create a Const or Val node, use the other constructors!");
    break;
  }
  }
  simplifyExpression();
}

bool mlir::llhd::Dnf::isProbedSignal() {
  return isVal() && isa<llhd::PrbOp>(value.getDefiningOp());
}

Value mlir::llhd::Dnf::getProbedSignal() {
  assert(isProbedSignal() && "Can only return probed signal if the value "
                             "actually got probed from a signal!");
  return cast<llhd::PrbOp>(value.getDefiningOp()).signal();
}

void mlir::llhd::Dnf::simplifyExpression() {
  for (auto it1 = children.begin(); it1 != children.end(); ++it1) {
    auto &c1 = *it1;
    if (!c1)
      continue;

    if (c1->isConst()) {
      if (c1->getConst() == true) {
        if (isAnd()) {
          // Remove a constant TRUE in an AND node, because it doesn't change
          // anything about the formula, except if it is the only child
          if (children.size() > 1) {
            children.erase(it1--);
            continue;
          }
        } else if (isOr()) {
          // If there is a constant TRUE in this OR node, remove all other
          // nodes, and replace this node with a constant TRUE, because it
          // will always evaluate to TRUE
          children.clear();
          type = DnfNodeType::Const;
          constant = true;
          inv = false;
          return;
        }
      } else {
        if (isAnd()) {
          // If there is a constant FALSE in this AND node, remove all
          // children and become a constant FALSE node
          children.clear();
          type = DnfNodeType::Const;
          constant = false;
          inv = false;
          return;
        } else if (isOr()) {
          // If there is a constant FALSE in an OR node, remove it because the
          // other nodes alone determine what the OR evaluates to, except if
          // it is the only child
          if (children.size() > 1) {
            children.erase(it1--);
            continue;
          }
        }
      }
    }

    for (auto it = std::next(it1); it != children.end(); ++it) {
      auto &c2 = *it;
      if (c1->isVal() && c2->isVal()) {
        if ((c1->value == c2->value)) {
          if (c1->inv == c2->inv) {
            // Remove duplicate Val children, if this node will end up with
            // only one child, it will become this child at the end of this
            // method
            children.erase(it--);
          } else {
            if (isAnd()) {
              // If this node is an AND node and it has two children which are
              // the opposite (one is the negation of the other), delete all
              // children of this node and replace this node with a constant
              // FALSE because this node will always be FALSE
              children.clear();
              type = DnfNodeType::Const;
              constant = false;
              inv = false;
              return;
            } else if (isOr()) {
              // If this node is an OR node and it has two children which are
              // the opposite (one is the negation of the other), delete all
              // children of this node and replace this node with a constant
              // TRUE because this node will always be TRUE
              children.clear();
              type = DnfNodeType::Const;
              constant = true;
              inv = false;
              return;
            }
          }
        }
      }
    }
  }

  // Erase AND and OR children which do not have children anymore
  for (auto it = children.begin(); it != children.end(); ++it) {
    auto &child = *it;
    if (child->isAnd() || child->isOr()) {
      if (child->children.empty()) {
        children.erase(it--);
      }
    }
  }

  // If you have one child only, become the child
  // if (children.size() == 1) {
  //   auto &child = *children.begin();
  //   type = child->type;
  //   inv = child->inv;
  //   value = child->value;
  //   std::vector<std::unique_ptr<Dnf>> c = std::move(child->children);
  //   children.clear();
  //   children = std::move(c);
  // }
}

static mlir::Value createConstOrValueDNF(mlir::OpBuilder &builder,
                                         mlir::llhd::Dnf &dnf) {
  mlir::Location loc = builder.getInsertionPoint()->getLoc();

  if (dnf.isConst()) {
    return builder.create<mlir::llhd::ConstOp>(
        loc, builder.getI1Type(), builder.getBoolAttr(dnf.getConst()));
  }
  if (dnf.isNegatedVal()) {
    return builder.create<mlir::llhd::NotOp>(loc, dnf.value);
  }
  if (dnf.isVal()) {
    return dnf.value;
  }
  return mlir::Value();
}

static mlir::Value createLeafOrAndDNF(mlir::OpBuilder &builder,
                                      mlir::llhd::Dnf &dnf) {
  mlir::Location loc = builder.getInsertionPoint()->getLoc();

  if (dnf.isLeaf())
    return createConstOrValueDNF(builder, dnf);

  if (dnf.isAnd()) {
    mlir::Value runner = mlir::Value();
    for (std::unique_ptr<mlir::llhd::Dnf> &term : dnf.children) {
      if (!runner) {
        runner = createConstOrValueDNF(builder, *term);
        continue;
      }
      runner = builder.create<mlir::llhd::AndOp>(
          loc, createConstOrValueDNF(builder, *term), runner);
    }
    return runner;
  }
  return mlir::Value();
}

static std::unique_ptr<mlir::llhd::Dnf> &
recursiveHelper(Block *curr, Block *source,
                DenseMap<Block *, std::unique_ptr<llhd::Dnf>> &mem,
                DenseMap<Block *, bool> &visited) {
  if (mem.count(curr))
    return mem[curr];

  if (curr == source) {
    mem.try_emplace(curr, std::make_unique<llhd::Dnf>(true, false));
    return mem[curr];
  }

  assert(!curr->getPredecessors().empty() && "Something went wrong");
  std::unique_ptr<llhd::Dnf> dnf =
      nullptr; // std::make_unique<llhd::Dnf>(llhd::Dnf(*recursiveHelper(*curr->pred_begin(),
               // source, mem, visited)));
  for (auto iter = curr->pred_begin(); iter != curr->pred_end(); ++iter) {
    if ((*iter)->getTerminator()->getNumSuccessors() == 1) {
      if (!dnf)
        dnf = std::make_unique<llhd::Dnf>(
            llhd::Dnf(*recursiveHelper(*iter, source, mem, visited)));
      else
        dnf = std::make_unique<llhd::Dnf>(
            llhd::Dnf(llhd::DnfNodeType::Or, std::move(dnf),
                      std::make_unique<llhd::Dnf>(llhd::Dnf(
                          *recursiveHelper(*iter, source, mem, visited)))));
    } else {
      auto condBr = cast<CondBranchOp>((*iter)->getTerminator());
      llhd::Dnf tmp(llhd::DnfNodeType::And,
                    std::make_unique<llhd::Dnf>(llhd::Dnf(
                        *recursiveHelper(*iter, source, mem, visited))),
                    std::make_unique<llhd::Dnf>(condBr.condition(),
                                                condBr.falseDest() == curr));
      if (!dnf)
        dnf = std::make_unique<llhd::Dnf>(tmp);
      else
        dnf = std::make_unique<llhd::Dnf>(llhd::DnfNodeType::Or, std::move(dnf),
                                          std::make_unique<llhd::Dnf>(tmp));
    }
  }
  mem.try_emplace(curr, std::move(dnf));
  return mem[curr];
}

Value mlir::llhd::Dnf::buildOperations(OpBuilder &builder) {
  Location loc = builder.getInsertionPoint()->getLoc();

  if (isLeaf() || isAnd())
    return createLeafOrAndDNF(builder, *this);

  if (isOr()) {
    mlir::Value runner = mlir::Value();
    for (std::unique_ptr<mlir::llhd::Dnf> &term : children) {
      if (!runner) {
        runner = createLeafOrAndDNF(builder, *term);
        continue;
      }
      runner = builder.create<mlir::llhd::OrOp>(
          loc, createLeafOrAndDNF(builder, *term), runner);
    }
    return runner;
  }
  return Value();
}

std::unique_ptr<llhd::Dnf>
mlir::llhd::getBooleanExprFromSourceToTarget(Block *source, Block *target) {
  assert(source->getParent() == target->getParent() &&
         "Blocks are required to be in the same region!");
  DenseMap<Block *, std::unique_ptr<llhd::Dnf>> memorization;
  DenseMap<Block *, bool> visited;

  return std::move(recursiveHelper(target, source, memorization, visited));
}
