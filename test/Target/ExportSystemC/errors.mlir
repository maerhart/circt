// RUN: circt-translate %s --export-systemc --verify-diagnostics --split-input-file | FileCheck %s

// CHECK: <<UNSUPPORTED OPERATION (hw.module)>>
// expected-error @+1 {{no emission pattern found for 'hw.module'}}
hw.module @notSupported () -> () { }

// -----

// CHECK: <<UNSUPPORTED TYPE (!hw.inout<i2>)>>
// expected-error @+1 {{no emission pattern found for type '!hw.inout<i2>'}}
systemc.module @invalidType (%port0: !systemc.in<!hw.inout<i2>>) {}

// -----

// expected-error @+1 {{no emission pattern found for type '() -> (i32, i32)'}}
func.func private @functionTypeMultipleResults (%a: () -> (i32, i32)) attributes {argNames=["a"]}

// -----

// expected-error @+1 {{no emission pattern found for 'func.func'}}
func.func private @argNamesWrongSize (%a: i32) attributes {argNames=[]}

// -----

// expected-error @+1 {{no emission pattern found for 'func.func'}}
func.func private @argNamesMissing (%a: i32)

// -----

// expected-error @+1 {{no emission pattern found for 'func.func'}}
func.func private @argNamesWrongElementType (%a: i32) attributes {argNames=[4 : i32]}

// -----

// expected-error @+1 {{no emission pattern found for 'func.func'}}
func.func private @multipleResults () -> (i32, i32)

func.func private @callOpMultipleResults () -> i32 {
  // expected-error @+1 {{inlining not supported for value '%0:2 = "func.call"() {callee = @multipleResults} : () -> (i32, i32)'}}
  %0, %1 = func.call @multipleResults() : () -> (i32, i32)
  // expected-note @+1 {{requested to be inlined here}}
  return %0 : i32
}

// -----

// expected-error @+1 {{no emission pattern found for 'func.func'}}
func.func private @multipleResults () -> (i32, i32)

func.func private @callIndirectOpMultipleResults () -> i32 {
  %f = func.constant @multipleResults : () -> (i32, i32)
  // expected-error @+1 {{inlining not supported for value '%1:2 = "func.call_indirect"(%0) : (() -> (i32, i32)) -> (i32, i32)'}}
  %0, %1 = func.call_indirect %f() : () -> (i32, i32)
  // expected-note @+1 {{requested to be inlined here}}
  return %0 : i32
}
