// RUN: circt-translate %s --export-systemc | FileCheck %s

// CHECK-LABEL: // stdout.h
// CHECK-NEXT: #ifndef STDOUT_H
// CHECK-NEXT: #define STDOUT_H

func.func private @funcDeclaration (%a: i32, %b: i32) -> i32
func.func private @voidFunc (%a: i32)

func.func @testFunc (%a: i64, %b: i32) -> i32 {
  %0 = func.call @funcDeclaration(%b, %b) : (i32, i32) -> i32
  %1 = func.constant @voidFunc : (i32) -> ()
  func.call_indirect %1 (%b) : (i32) -> ()
  return %0 : i32
}

// CHECK: #endif // STDOUT_H
