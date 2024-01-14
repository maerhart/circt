// RUN: circt-opt %s --lower-smt-to-z3-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void --shared-libs=/usr/lib/x86_64-linux-gnu/libz3.so | \
// RUN: FileCheck %s

// COM: res=1 indicates satisfiability of "a == 0" which is obviously true by
// COM: assigning 0 to a
// CHECK: res=1
func.func @simple() {
  %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  %0 = smt.declare_const "a" : !smt.bv<32>
  %1 = smt.eq %c0_bv32, %0 : !smt.bv<32>
  %s = smt.solver_create "s"
  smt.assert %s, %1
  smt.check_sat %s
  return
}

llvm.func @entry() {
  %0 = llvm.call @Z3_mk_config() :  () -> !llvm.ptr
  %1 = llvm.call @Z3_mk_context(%0) : (!llvm.ptr) -> !llvm.ptr
  %2 = llvm.mlir.addressof @ctx : !llvm.ptr
  llvm.store %1, %2 : !llvm.ptr, !llvm.ptr
  llvm.call @Z3_del_config(%0) :  (!llvm.ptr) -> ()
  func.call @simple() : () -> ()
  llvm.return
}

llvm.mlir.global internal @ctx() {alignment = 8 : i64} : !llvm.ptr {
  %0 = llvm.mlir.zero : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func @Z3_mk_config() -> !llvm.ptr
llvm.func @Z3_mk_context(!llvm.ptr) -> !llvm.ptr
llvm.func @Z3_del_config(!llvm.ptr) -> ()
