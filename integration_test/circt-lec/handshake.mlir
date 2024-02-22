//  RUN: circt-lec %s -c1=passthrough -c2=sinkSource | FileCheck %s --check-prefix=CHECK_SOURCE_SINK

// Those two circuits are not equivalent because 'passthrough' always produces
// the same amount of tokens as it gets in, while 'sinkSource' can produce more
// or less than it gets as input.

handshake.func @passthrough2to1(%arg0: none, %arg1: none) -> none {
  return %arg0 : none
}

handshake.func @passthrough1to1(%arg0: none) -> none {
  return %arg0 : none
}

handshake.func @buffer(%arg0: i32) -> i32 {
  %0 = handshake.buffer [1000] fifo %arg0 : i32
  return %0 : i32
}

handshake.func @passthrough1to1int(%arg0: i32) -> i32 {
  return %arg0 : i32
}

handshake.func @const0(%arg0: none) -> i32 {
  %0 = handshake.constant %arg0 {value = 0 : i32} : i32
  return %0 : i32
}

handshake.func @const1(%arg0: none) -> i32 {
  %0 = handshake.constant %arg0 {value = 1 : i32} : i32
  return %0 : i32
}

handshake.func @const0FromSource(%arg0: none) -> i32 {
  handshake.sink %arg0 : none
  %0 = handshake.source
  %1 = handshake.constant %0 {value = 1 : i32} : i32
  return %1 : i32
}

handshake.func @sinkSource(%arg0: none) -> none {
  handshake.sink %arg0 : none
  %0 = handshake.source
  handshake.return %0 : none
}

// // -----
// //  RUN: circt-lec %s -c1=forkLhs -c2=forkRhs | FileCheck %s --check-prefix=CHECK_FORK_COMMUTATIVE

// // Those two circuits are equivalent because the number of output tokens and
// // their values always match, only the exact delay differs (which we don't care
// // about in latency-insensitive proofs).

// Join has very bad performance, can only prove this function for token
// sequences of up to 2 in reasonable time.
handshake.func @forkJoin(%arg0: none) -> none {
  %0:2 = handshake.fork [2] %arg0 : none
  %1 = handshake.join %0#0, %0#1 : none, none
  return %1 : none
}

handshake.func @forkRhs(%arg0: i32) -> (i32, i32, i32) {
  %0:2 = handshake.fork [2] %arg0 : i32
  %1:2 = handshake.fork [2] %0#1 : i32
  return %0#0, %1#0, %1#1 : i32, i32, i32
}

// -----
//  RUN: circt-lec %s -c1=joinLhs -c2=joinRhs | FileCheck %s --check-prefix=CHECK_JOIN_TREE

handshake.func @join1(%arg0: none, %arg1: none) -> none {
  %0 = handshake.join %arg0, %arg1 : none, none
  return %0 : none
}

handshake.func @join2(%arg0: none, %arg1: none) -> none {
  %0 = handshake.join %arg0, %arg1 : none, none
  return %0 : none
}

// Those are equivalent.

// handshake.func @joinLhs(%arg0: i32, %arg1: i32, %arg2: i32) -> none {
//   // Values: %1 = constant none thus holds trivially
//   // Length:
//   // len(%0) == min(len(%arg0), len(%arg1))
//   // len(%1) == min(len(%0), len(%arg2))
//   %0 = handshake.join %arg0, %arg1 : i32, i32
//   %1 = handshake.join %0, %arg2 : none, i32
//   return %1 : none
// }

// handshake.func @joinRhs(%arg0: i32, %arg1: i32, %arg2: i32) -> none {
//   // Values: %1 = constant none  thus holds trivially
//   // Length:
//   // len(%0) == min(len(%arg1), len(%arg2))
//   // len(%1) == min(len(%arg0), len(%0))
//   %0 = handshake.join %arg1, %arg2 : i32, i32
//   %1 = handshake.join %arg0, %0 : i32, none
//   return %1 : none
// }

// -----
//  RUN: circt-lec %s -c1=joinLhs -c2=joinRhs | FileCheck %s --check-prefix=CHECK_JOIN_TREE

handshake.func @branchFork(%cond: i1, %arg1: i32) -> (i32, i32, i32) {
  %a, %b = handshake.cond_br %cond, %arg1 : i32
  %c, %d = handshake.fork [2] %b : i32
  return %a, %c, %d : i32, i32, i32
}

handshake.func @doubleBranchFork(%cond: i1, %arg1: i32) -> (i32, i32, i32) {
  %c1, %c2 = handshake.fork [2] %cond : i1
  %a1, %a2 = handshake.fork [2] %arg1 : i32
  %a, %b = handshake.cond_br %c1, %a1 : i32
  %c, %d = handshake.cond_br %c2, %a2 : i32
  handshake.sink %c : i32
  return %a, %b, %d : i32, i32, i32
}

handshake.func @doubleBranchForkWrong(%cond: i1, %arg1: i32) -> (i32, i32, i32) {
  %c1, %c2 = handshake.fork [2] %cond : i1
  %a1, %a2 = handshake.fork [2] %arg1 : i32
  %a, %b = handshake.cond_br %c1, %a1 : i32
  %c, %d = handshake.cond_br %c2, %a2 : i32
  handshake.sink %d : i32
  return %a, %b, %c : i32, i32, i32
}

handshake.func @branch(%cond: i1, %arg1: i32) -> (i32, i32) {
  %a, %b = handshake.cond_br %cond, %arg1 : i32
  return %a, %b : i32, i32
}

handshake.func @negBranch(%cond: i1, %arg1: i32) -> (i32, i32) {
  %0 = handshake.source
  %1 = handshake.join %cond, %0 : i1, none
  %true = handshake.constant %1 {value = 1 : i1} : i1
  %neg_cond = comb.xor %cond, %true : i1
  %a, %b = handshake.cond_br %neg_cond, %arg1 : i32
  return %b, %a : i32, i32
}

// This is ~16x slower to prove equivalence with @branch than @negBranch is!
handshake.func @negBranch2(%cond: i1, %arg1: i32) -> (i32, i32) {
  %1 = handshake.join %cond, %cond : i1, i1
  %true = handshake.constant %1 {value = 1 : i1} : i1
  %neg_cond = comb.xor %cond, %true : i1
  %a, %b = handshake.cond_br %neg_cond, %arg1 : i32
  return %b, %a : i32, i32
}

// This is ~86x slower to prove equivalence with @branch than @negBranch is!
handshake.func @negBranch3(%cond: i1, %arg1: i32) -> (i32, i32) {
  %0 = handshake.source
  %true = handshake.constant %0 {value = 1 : i1} : i1
  %neg_cond = comb.xor %cond, %true : i1
  %a, %b = handshake.cond_br %neg_cond, %arg1 : i32
  return %b, %a : i32, i32
}

handshake.func @negNeg(%arg0: i32) -> i32 {
  %0 = handshake.join %arg0, %arg0 : i32, i32
  // %0 = handshake.source
  %c-1 = handshake.constant %0 {value = -1 : i32} : i32
  %neg = comb.xor %arg0, %c-1 : i32
  %res = comb.xor %neg, %c-1 : i32
  return %res : i32
}

handshake.func @divAndMulBy2(%arg0: i32) -> i32 {
  %ctrl = handshake.join %arg0, %arg0 : i32, i32
  %c1 = handshake.constant %ctrl {value = 1 : i32} : i32
  %c2 = handshake.constant %ctrl {value = 2 : i32} : i32
  %0 = comb.shrs %arg0, %c1 : i32
  %1 = comb.mul %0, %c2 : i32
  return %1 : i32
}

handshake.func @mulAndDivBy2(%arg0: i32) -> i32 {
  %ctrl = handshake.join %arg0, %arg0 : i32, i32
  %c1 = handshake.constant %ctrl {value = 1 : i32} : i32
  %c2 = handshake.constant %ctrl {value = 2 : i32} : i32
  %0 = comb.mul %arg0, %c2 : i32
  %1 = comb.shrs %0, %c1 : i32
  return %1 : i32
}
