// RUN: circt-translate %s --export-systemc | FileCheck %s

// CHECK-LABEL: // stdout.h
// CHECK-NEXT: #ifndef STDOUT_H
// CHECK-NEXT: #define STDOUT_H

// CHECK: uint32_t funcDeclaration(uint32_t funcArg0, uint32_t funcArg1);
func.func private @funcDeclaration (%a: i32, %b: i32) -> i32 attributes {argNames=["funcArg0", "funcArg1"]}
// CHECK-EMPTY:
// CHECK-NEXT: void voidFunc() {
func.func @voidFunc () {
  // CHECK-NEXT: return;
  return
// CHECK-NEXT: }
}
// CHECK-EMPTY: 
// CHECK-NEXT: uint32_t testFunc(uint64_t a, uint32_t b, std::function<uint32_t(uint8_t, bool)> c, std::function<void()> d) {
func.func @testFunc (%a: i64, %b: i32, %c: (i8, i1) -> i32, %d: () -> ()) -> i32 attributes {argNames=["a", "b", "c", "d"]} {
  %0 = func.call @funcDeclaration(%b, %b) : (i32, i32) -> i32
  // CHECK-NEXT: voidFunc();
  func.call @voidFunc() : () -> ()
  // CHECK-NEXT: voidFunc();
  %1 = func.constant @voidFunc : () -> ()
  func.call_indirect %1 () : () -> ()
  // CHECK-NEXT: uint32_t v = funcDeclaration(b, b);
  %2 = func.call @funcDeclaration(%b, %b) : (i32, i32) -> i32
  %v = systemc.cpp.variable %2 : i32
  // CHECK-NEXT: return funcDeclaration(b, b);
  return %0 : i32
// CHECK-NEXT: }
}

// CHECK: #endif // STDOUT_H
