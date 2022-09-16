// REQUIRES: clang-tidy
// RUN: circt-translate %s --export-systemc > %t.cpp
// RUN: clang-tidy --extra-arg=-frtti %t.cpp

emitc.include <"stdint.h">
emitc.include <"functional">

func.func private @funcDeclaration (%a: i32, %b: i32) -> i32 attributes {argNames=["funcArg0", "funcArg1"]}

func.func @voidFunc () {
  return
}

func.func @testFunc (%a: i64, %b: i32, %c: (i8, i1) -> i32, %d: () -> ()) -> i32 attributes {argNames=["a", "b", "c", "d"]} {
  %0 = func.call @funcDeclaration(%b, %b) : (i32, i32) -> i32
  func.call @voidFunc() : () -> ()
  %1 = func.constant @voidFunc : () -> ()
  func.call_indirect %1 () : () -> ()
  %2 = func.call @funcDeclaration(%b, %b) : (i32, i32) -> i32
  %v = systemc.cpp.variable %2 : i32
  return %0 : i32
}
