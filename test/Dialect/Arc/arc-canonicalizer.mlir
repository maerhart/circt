// RUN: circt-opt %s --arc-canonicalizer | FileCheck %s

//===----------------------------------------------------------------------===//
// Remove Passthrough calls
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @passthoughChecks
hw.module @passthoughChecks(%in0: i1, %in1: i1) -> (out0: i1, out1: i1, out2: i1, out3: i1, out4: i1, out5: i1, out6: i1, out7: i1, out8: i1, out9: i1) {
  %0:2 = arc.call @passthrough(%in0, %in1) : (i1, i1) -> (i1, i1)
  %1:2 = arc.call @noPassthrough(%in0, %in1) : (i1, i1) -> (i1, i1)
  %2:2 = arc.state @passthrough(%in0, %in1) lat 0 : (i1, i1) -> (i1, i1)
  %3:2 = arc.state @noPassthrough(%in0, %in1) lat 0 : (i1, i1) -> (i1, i1)
  %4:2 = arc.state @passthrough(%in0, %in1) clock %in0 lat 1 : (i1, i1) -> (i1, i1)
  hw.output %0#0, %0#1, %1#0, %1#1, %2#0, %2#1, %3#0, %3#1, %4#0, %4#1 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  // CHECK-NEXT: [[V0:%.+]]:2 = arc.call @noPassthrough(%in0, %in1) :
  // CHECK-NEXT: [[V1:%.+]]:2 = arc.state @noPassthrough(%in0, %in1) lat 0 :
  // CHECK-NEXT: [[V2:%.+]]:2 = arc.state @passthrough(%in0, %in1) clock %in0 lat 1 :
  // CHECK-NEXT: hw.output %in0, %in1, [[V0]]#0, [[V0]]#1, %in0, %in1, [[V1]]#0, [[V1]]#1, [[V2]]#0, [[V2]]#1 :
}
arc.define @passthrough(%arg0: i1, %arg1: i1) -> (i1, i1) {
  arc.output %arg0, %arg1 : i1, i1
}
arc.define @noPassthrough(%arg0: i1, %arg1: i1) -> (i1, i1) {
  arc.output %arg1, %arg0 : i1, i1
}

//===----------------------------------------------------------------------===//
// MemoryWritePortOp canonicalizer
//===----------------------------------------------------------------------===//

arc.define @memArcFalse(%arg0: i1, %arg1: i32) -> (i1, i32, i1) {
  %false = hw.constant false
  arc.output %arg0, %arg1, %false : i1, i32, i1
}
arc.define @memArcTrue(%arg0: i1, %arg1: i32) -> (i1, i32, i1) {
  %true = hw.constant true
  arc.output %arg0, %arg1, %true : i1, i32, i1
}

// CHECK-LABEL: hw.module @memoryWritePortCanonicalizations
hw.module @memoryWritePortCanonicalizations(%clk: i1, %addr: i1, %data: i32) {
  // CHECK-NEXT: [[MEM:%.+]] = arc.memory <2 x i32, i1>
  %mem = arc.memory <2 x i32, i1>
  arc.memory_write_port %mem, @memArcFalse(%addr, %data) clock %clk enable lat 1 : <2 x i32, i1>, i1, i32
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @memArcTrue_0(%addr, %data) clock %clk lat 1 :
  arc.memory_write_port %mem, @memArcTrue(%addr, %data) clock %clk enable lat 1 : <2 x i32, i1>, i1, i32
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @memArcTrue_0(%addr, %data) clock %clk lat 1 :
  arc.memory_write_port %mem, @memArcTrue(%addr, %data) clock %clk enable lat 1 : <2 x i32, i1>, i1, i32
  %0:3 = arc.state @memArcTrue(%addr, %data) lat 0 : (i1, i32) -> (i1, i32, i1)
  // CHECK-NEXT: hw.output
  hw.output
}

//===----------------------------------------------------------------------===//
// RemoveUnusedArcs
//===----------------------------------------------------------------------===//

// CHECK-NOT: arc.define @unusedArcIsDeleted
arc.define @unusedArcIsDeleted(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arc.call @nestedUnused(%arg0, %arg1) : (i32, i32) -> i32
  arc.output %0 : i32
}
// CHECK-NOT: arc.define @nestedUnused
arc.define @nestedUnused(%arg0: i32, %arg1: i32) -> i32 {
  %0 = comb.add %arg0, %arg1 : i32
  arc.output %0 : i32
}

//===----------------------------------------------------------------------===//
// ICMPCanonicalizer
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @icmpEqCanonicalizer
hw.module @icmpEqCanonicalizer(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i4, %arg5: i4, %arg6: i4, %arg7: i4) -> (out0: i1, out1: i1, out2: i1, out3: i1) {
  // CHECK: [[V0:%.+]] = comb.and bin %arg0, %arg1, %arg2, %arg3 : i1
  %c-1_i4 = hw.constant -1 : i4
  %0 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %1 = comb.icmp bin eq %0, %c-1_i4 : i4

  // CHECK-NEXT: [[V1:%.+]] = comb.or bin %arg0, %arg1, %arg2, %arg3 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.xor bin [[V1]], %true : i1
  %c0_i4 = hw.constant 0 : i4
  %2 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %3 = comb.icmp bin eq %2, %c0_i4 : i4

  // CHECK-NEXT: [[V3:%.+]] = comb.and bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V4:%.+]] = comb.icmp bin eq [[V3]], %c-1_i4 : i4
  %c-1_i16 = hw.constant -1 : i16
  %4 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %5 = comb.icmp bin eq %4, %c-1_i16 : i16

  // CHECK-NEXT: [[V5:%.+]] = comb.or bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V6:%.+]] = comb.icmp bin eq [[V5]], %c0_i4 : i4
  %c0_i16 = hw.constant 0 : i16
  %6 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %7 = comb.icmp bin eq %6, %c0_i16 : i16

  // CHECK-NEXT: hw.output [[V0]], [[V2]], [[V4]], [[V6]] :
  hw.output %1, %3, %5, %7 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @icmpNeCanonicalizer
hw.module @icmpNeCanonicalizer(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i4, %arg5: i4, %arg6: i4, %arg7: i4) -> (out0: i1, out1: i1, out2: i1, out3: i1) {
  // CHECK: [[V0:%.+]] = comb.or bin %arg0, %arg1, %arg2, %arg3 : i1
  %c0_i4 = hw.constant 0 : i4
  %0 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %1 = comb.icmp bin ne %0, %c0_i4 : i4

  // CHECK-NEXT: [[V1:%.+]] = comb.and bin %arg0, %arg1, %arg2, %arg3 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.xor bin [[V1]], %true : i1
  %c-1_i4 = hw.constant -1 : i4
  %2 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %3 = comb.icmp bin ne %2, %c-1_i4 : i4

  // CHECK-NEXT: [[V3:%.+]] = comb.or bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V4:%.+]] = comb.icmp bin ne [[V3]], %c0_i4 : i4
  %c0_i16 = hw.constant 0 : i16
  %4 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %5 = comb.icmp bin ne %4, %c0_i16 : i16

  // CHECK-NEXT: [[V5:%.+]] = comb.and bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V6:%.+]] = comb.icmp bin ne [[V5]], %c-1_i4 : i4
  %c-1_i16 = hw.constant -1 : i16
  %6 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %7 = comb.icmp bin ne %6, %c-1_i16 : i16

  // CHECK-NEXT: hw.output [[V0]], [[V2]], [[V4]], [[V6]] :
  hw.output %1, %3, %5, %7 : i1, i1, i1, i1
}

//===----------------------------------------------------------------------===//
// RemoveUnusedArcArguments
//===----------------------------------------------------------------------===//

// COM: this has to be before @OneOfThreeUsed to check that arguments that
// COM: become unused during the process are removed as well.
// CHECK: arc.define @NestedCall(%arg0: i1) -> i1 {
arc.define @NestedCall(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
  // CHECK: arc.call @OneOfThreeUsed(%arg0) : (i1) -> i1
  %0 = arc.call @OneOfThreeUsed(%arg0, %arg1, %arg2) : (i1, i1, i1) -> i1
  arc.output %0 : i1
}

// CHECK-LABEL: arc.define @OneOfThreeUsed(%arg0: i1)
arc.define @OneOfThreeUsed(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
  %true = hw.constant true
  %0 = comb.xor %arg1, %true : i1
  // CHECK: arc.output {{%[0-9]+}} :
  arc.output %0 : i1
}

// CHECK: @test1
hw.module @test1 (%arg0: i1, %arg1: i1, %arg2: i1, %clock: i1) -> (out0: i1, out1: i1) {
  // CHECK-NEXT: arc.state @OneOfThreeUsed(%arg1) clock %clock lat 1 : (i1) -> i1
  %0 = arc.state @OneOfThreeUsed(%arg0, %arg1, %arg2) clock %clock lat 1 : (i1, i1, i1) -> i1
  // CHECK-NEXT: arc.state @NestedCall(%arg1)
  %1 = arc.state @NestedCall(%arg0, %arg1, %arg2) clock %clock lat 1 : (i1, i1, i1) -> i1
  hw.output %0, %1 : i1, i1
}

// CHECK-LABEL: arc.define @NoArgsToRemove()
arc.define @NoArgsToRemove() -> i1 {
  %0 = hw.constant 0 : i1
  arc.output %0 : i1
}

// CHECK: @test2
hw.module @test2 () -> (out: i1) {
  // CHECK-NEXT: arc.state @NoArgsToRemove() lat 0 : () -> i1
  %0 = arc.state @NoArgsToRemove() lat 0 : () -> i1
  hw.output %0 : i1
}

//===----------------------------------------------------------------------===//
// SinkArcInputs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @SinkSameConstantsArc(%arg0: i4)
arc.define @SinkSameConstantsArc(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: %c2_i4 = hw.constant 2
  // CHECK-NEXT: [[TMP:%.+]] = comb.add %arg0, %c2_i4
  // CHECK-NEXT: arc.output [[TMP]]
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK: arc.define @Foo
arc.define @Foo(%arg0: i4) -> i4 {
  // CHECK-NOT: hw.constant
  %k1 = hw.constant 2 : i4
  // CHECK: {{%.+}} = arc.call @SinkSameConstantsArc(%arg0)
  %0 = arc.call @SinkSameConstantsArc(%arg0, %k1) : (i4, i4) -> i4
  arc.output %0 : i4
}

// CHECK: hw.module @SinkSameConstants
hw.module @SinkSameConstants(%x: i4) -> (out0: i4, out1: i4, out2: i4) {
  // CHECK-NOT: hw.constant
  // CHECK-NEXT: %0 = arc.state @SinkSameConstantsArc(%x)
  // CHECK-NEXT: %1 = arc.state @SinkSameConstantsArc(%x)
  // CHECK-NEXT: arc.call
  // CHECK-NEXT: hw.output
  %k1 = hw.constant 2 : i4
  %k2 = hw.constant 2 : i4
  %0 = arc.state @SinkSameConstantsArc(%x, %k1) lat 0 : (i4, i4) -> i4
  %1 = arc.state @SinkSameConstantsArc(%x, %k2) lat 0 : (i4, i4) -> i4
  %2 = arc.call @Foo(%x) : (i4) -> i4
  hw.output %0, %1, %2 : i4, i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @DontSinkDifferentConstantsArc(%arg0: i4, %arg1: i4)
arc.define @DontSinkDifferentConstantsArc(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: comb.add %arg0, %arg1
  // CHECK-NEXT: arc.output
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-LABEL: hw.module @DontSinkDifferentConstants
hw.module @DontSinkDifferentConstants(%x: i4) -> (out0: i4, out1: i4) {
  // CHECK-NEXT: %c2_i4 = hw.constant 2 : i4
  // CHECK-NEXT: %c3_i4 = hw.constant 3 : i4
  // CHECK-NEXT: %0 = arc.state @DontSinkDifferentConstantsArc(%x, %c2_i4)
  // CHECK-NEXT: %1 = arc.state @DontSinkDifferentConstantsArc(%x, %c3_i4)
  // CHECK-NEXT: hw.output
  %c2_i4 = hw.constant 2 : i4
  %c3_i4 = hw.constant 3 : i4
  %0 = arc.state @DontSinkDifferentConstantsArc(%x, %c2_i4) lat 0 : (i4, i4) -> i4
  %1 = arc.state @DontSinkDifferentConstantsArc(%x, %c3_i4) lat 0 : (i4, i4) -> i4
  hw.output %0, %1 : i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @DontSinkDifferentConstantsArc1(%arg0: i4, %arg1: i4)
arc.define @DontSinkDifferentConstantsArc1(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: [[TMP:%.+]] = comb.add %arg0, %arg1
  // CHECK-NEXT: arc.output [[TMP]]
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK: arc.define @Bar
arc.define @Bar(%arg0: i4) -> i4 {
  // CHECK: %c1_i4 = hw.constant 1
  %k1 = hw.constant 1 : i4
  // CHECK: {{%.+}} = arc.call @DontSinkDifferentConstantsArc1(%arg0, %c1_i4)
  %0 = arc.call @DontSinkDifferentConstantsArc1(%arg0, %k1) : (i4, i4) -> i4
  arc.output %0 : i4
}

// CHECK: hw.module @DontSinkDifferentConstants1
hw.module @DontSinkDifferentConstants1(%x: i4) -> (out0: i4, out1: i4, out2: i4) {
  // CHECK-NEXT: %c2_i4 = hw.constant 2 : i4
  // CHECK-NEXT: %0 = arc.state @DontSinkDifferentConstantsArc1(%x, %c2_i4)
  // CHECK-NEXT: %1 = arc.state @DontSinkDifferentConstantsArc1(%x, %c2_i4)
  // CHECK-NEXT: arc.call
  // CHECK-NEXT: hw.output
  %k1 = hw.constant 2 : i4
  %k2 = hw.constant 2 : i4
  %0 = arc.state @DontSinkDifferentConstantsArc1(%x, %k1) lat 0 : (i4, i4) -> i4
  %1 = arc.state @DontSinkDifferentConstantsArc1(%x, %k2) lat 0 : (i4, i4) -> i4
  %2 = arc.call @Bar(%x) : (i4) -> i4
  hw.output %0, %1, %2 : i4, i4, i4
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//
// Canonicalize mux sequences
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @muxSequenceImplementingLZC
func.func @muxSequenceImplementingLZC(%arg0: i16) -> i4 {
  %true = hw.constant true
  %c-1_i3 = hw.constant -1 : i3
  %c-3_i4 = hw.constant -3 : i4
  %c-4_i4 = hw.constant -4 : i4
  %c-5_i4 = hw.constant -5 : i4
  %c-6_i4 = hw.constant -6 : i4
  %c-7_i4 = hw.constant -7 : i4
  %c-8_i4 = hw.constant -8 : i4
  %c7_i4 = hw.constant 7 : i4
  %c6_i4 = hw.constant 6 : i4
  %c5_i4 = hw.constant 5 : i4
  %c4_i4 = hw.constant 4 : i4
  %c3_i4 = hw.constant 3 : i4
  %c2_i4 = hw.constant 2 : i4
  %c1_i4 = hw.constant 1 : i4
  %c0_i4 = hw.constant 0 : i4
  %0 = comb.extract %arg0 from 15 : (i16) -> i1
  %1 = comb.extract %arg0 from 14 : (i16) -> i1
  %2 = comb.extract %arg0 from 13 : (i16) -> i1
  %3 = comb.extract %arg0 from 12 : (i16) -> i1
  %4 = comb.extract %arg0 from 11 : (i16) -> i1
  %5 = comb.extract %arg0 from 10 : (i16) -> i1
  %6 = comb.extract %arg0 from 9 : (i16) -> i1
  %7 = comb.extract %arg0 from 8 : (i16) -> i1
  %8 = comb.extract %arg0 from 7 : (i16) -> i1
  %9 = comb.extract %arg0 from 6 : (i16) -> i1
  %10 = comb.extract %arg0 from 5 : (i16) -> i1
  %11 = comb.extract %arg0 from 4 : (i16) -> i1
  %12 = comb.extract %arg0 from 3 : (i16) -> i1
  %13 = comb.extract %arg0 from 2 : (i16) -> i1
  %14 = comb.extract %arg0 from 1 : (i16) -> i1
  %15 = comb.xor %14, %true : i1
  %16 = comb.concat %c-1_i3, %15 : i3, i1
  %18 = comb.mux bin %13, %c-3_i4, %16 : i4
  %19 = comb.mux bin %12, %c-4_i4, %18 : i4
  %20 = comb.mux bin %11, %c-5_i4, %19 : i4
  %21 = comb.mux bin %10, %c-6_i4, %20 : i4
  %22 = comb.mux bin %9, %c-7_i4, %21 : i4
  %23 = comb.mux bin %8, %c-8_i4, %22 : i4
  %24 = comb.mux bin %7, %c7_i4, %23 : i4
  %25 = comb.mux bin %6, %c6_i4, %24 : i4
  %26 = comb.mux bin %5, %c5_i4, %25 : i4
  %27 = comb.mux bin %4, %c4_i4, %26 : i4
  %28 = comb.mux bin %3, %c3_i4, %27 : i4
  %29 = comb.mux bin %2, %c2_i4, %28 : i4
  %30 = comb.mux bin %1, %c1_i4, %29 : i4
  %31 = comb.mux bin %0, %c0_i4, %30 : i4
  return %31 : i4

  // CHECK-NEXT: [[EXT1:%.+]] = comb.extract %arg0 from 1 : (i16) -> i15
  // CHECK-NEXT: [[LZC:%.+]] = arc.zero_count leading [[EXT1]] : i15
  // CHECK-NEXT: [[EXT2:%.+]] = comb.extract [[LZC]] from 0 : (i15) -> i4
  // CHECK-NEXT: return [[EXT2]] : i4
}

// CHECK-LABEL: @muxSequenceImplementingShift
func.func @muxSequenceImplementingShift(%arg0: i100, %arg1: i100) -> i100 {
  %c-1_i100 = hw.constant -1 : i100
  %c0_i99 = hw.constant 0 : i99
  %c316912650057057350374175801344_i100 = hw.constant 316912650057057350374175801344 : i100
  %c158456325028528675187087900672_i100 = hw.constant 158456325028528675187087900672 : i100
  %c79228162514264337593543950336_i100 = hw.constant 79228162514264337593543950336 : i100
  %c39614081257132168796771975168_i100 = hw.constant 39614081257132168796771975168 : i100
  %c19807040628566084398385987584_i100 = hw.constant 19807040628566084398385987584 : i100
  %c9903520314283042199192993792_i100 = hw.constant 9903520314283042199192993792 : i100
  %c4951760157141521099596496896_i100 = hw.constant 4951760157141521099596496896 : i100
  %c2475880078570760549798248448_i100 = hw.constant 2475880078570760549798248448 : i100
  %c1237940039285380274899124224_i100 = hw.constant 1237940039285380274899124224 : i100
  %c618970019642690137449562112_i100 = hw.constant 618970019642690137449562112 : i100
  %c309485009821345068724781056_i100 = hw.constant 309485009821345068724781056 : i100
  %c154742504910672534362390528_i100 = hw.constant 154742504910672534362390528 : i100
  %c77371252455336267181195264_i100 = hw.constant 77371252455336267181195264 : i100
  %c38685626227668133590597632_i100 = hw.constant 38685626227668133590597632 : i100
  %c19342813113834066795298816_i100 = hw.constant 19342813113834066795298816 : i100
  %c9671406556917033397649408_i100 = hw.constant 9671406556917033397649408 : i100
  %c4835703278458516698824704_i100 = hw.constant 4835703278458516698824704 : i100
  %c2417851639229258349412352_i100 = hw.constant 2417851639229258349412352 : i100
  %c1208925819614629174706176_i100 = hw.constant 1208925819614629174706176 : i100
  %c604462909807314587353088_i100 = hw.constant 604462909807314587353088 : i100
  %c302231454903657293676544_i100 = hw.constant 302231454903657293676544 : i100
  %c151115727451828646838272_i100 = hw.constant 151115727451828646838272 : i100
  %c75557863725914323419136_i100 = hw.constant 75557863725914323419136 : i100
  %c37778931862957161709568_i100 = hw.constant 37778931862957161709568 : i100
  %c18889465931478580854784_i100 = hw.constant 18889465931478580854784 : i100
  %c9444732965739290427392_i100 = hw.constant 9444732965739290427392 : i100
  %c4722366482869645213696_i100 = hw.constant 4722366482869645213696 : i100
  %c2361183241434822606848_i100 = hw.constant 2361183241434822606848 : i100
  %c1180591620717411303424_i100 = hw.constant 1180591620717411303424 : i100
  %c590295810358705651712_i100 = hw.constant 590295810358705651712 : i100
  %c295147905179352825856_i100 = hw.constant 295147905179352825856 : i100
  %c147573952589676412928_i100 = hw.constant 147573952589676412928 : i100
  %c73786976294838206464_i100 = hw.constant 73786976294838206464 : i100
  %c36893488147419103232_i100 = hw.constant 36893488147419103232 : i100
  %c18446744073709551616_i100 = hw.constant 18446744073709551616 : i100
  %c9223372036854775808_i100 = hw.constant 9223372036854775808 : i100
  %c4611686018427387904_i100 = hw.constant 4611686018427387904 : i100
  %c2305843009213693952_i100 = hw.constant 2305843009213693952 : i100
  %c1152921504606846976_i100 = hw.constant 1152921504606846976 : i100
  %c576460752303423488_i100 = hw.constant 576460752303423488 : i100
  %c288230376151711744_i100 = hw.constant 288230376151711744 : i100
  %c144115188075855872_i100 = hw.constant 144115188075855872 : i100
  %c72057594037927936_i100 = hw.constant 72057594037927936 : i100
  %c36028797018963968_i100 = hw.constant 36028797018963968 : i100
  %c18014398509481984_i100 = hw.constant 18014398509481984 : i100
  %c9007199254740992_i100 = hw.constant 9007199254740992 : i100
  %c4503599627370496_i100 = hw.constant 4503599627370496 : i100
  %c2251799813685248_i100 = hw.constant 2251799813685248 : i100
  %c1125899906842624_i100 = hw.constant 1125899906842624 : i100
  %c562949953421312_i100 = hw.constant 562949953421312 : i100
  %c281474976710656_i100 = hw.constant 281474976710656 : i100
  %c140737488355328_i100 = hw.constant 140737488355328 : i100
  %c70368744177664_i100 = hw.constant 70368744177664 : i100
  %c35184372088832_i100 = hw.constant 35184372088832 : i100
  %c17592186044416_i100 = hw.constant 17592186044416 : i100
  %c8796093022208_i100 = hw.constant 8796093022208 : i100
  %c4398046511104_i100 = hw.constant 4398046511104 : i100
  %c2199023255552_i100 = hw.constant 2199023255552 : i100
  %c1099511627776_i100 = hw.constant 1099511627776 : i100
  %c549755813888_i100 = hw.constant 549755813888 : i100
  %c274877906944_i100 = hw.constant 274877906944 : i100
  %c137438953472_i100 = hw.constant 137438953472 : i100
  %c68719476736_i100 = hw.constant 68719476736 : i100
  %c34359738368_i100 = hw.constant 34359738368 : i100
  %c17179869184_i100 = hw.constant 17179869184 : i100
  %c8589934592_i100 = hw.constant 8589934592 : i100
  %c4294967296_i100 = hw.constant 4294967296 : i100
  %c2147483648_i100 = hw.constant 2147483648 : i100
  %c1073741824_i100 = hw.constant 1073741824 : i100
  %c536870912_i100 = hw.constant 536870912 : i100
  %c268435456_i100 = hw.constant 268435456 : i100
  %c134217728_i100 = hw.constant 134217728 : i100
  %c67108864_i100 = hw.constant 67108864 : i100
  %c33554432_i100 = hw.constant 33554432 : i100
  %c16777216_i100 = hw.constant 16777216 : i100
  %c8388608_i100 = hw.constant 8388608 : i100
  %c4194304_i100 = hw.constant 4194304 : i100
  %c2097152_i100 = hw.constant 2097152 : i100
  %c1048576_i100 = hw.constant 1048576 : i100
  %c524288_i100 = hw.constant 524288 : i100
  %c262144_i100 = hw.constant 262144 : i100
  %c131072_i100 = hw.constant 131072 : i100
  %c65536_i100 = hw.constant 65536 : i100
  %c32768_i100 = hw.constant 32768 : i100
  %c16384_i100 = hw.constant 16384 : i100
  %c8192_i100 = hw.constant 8192 : i100
  %c4096_i100 = hw.constant 4096 : i100
  %c2048_i100 = hw.constant 2048 : i100
  %c1024_i100 = hw.constant 1024 : i100
  %c512_i100 = hw.constant 512 : i100
  %c256_i100 = hw.constant 256 : i100
  %c128_i100 = hw.constant 128 : i100
  %c64_i100 = hw.constant 64 : i100
  %c32_i100 = hw.constant 32 : i100
  %c16_i100 = hw.constant 16 : i100
  %c8_i100 = hw.constant 8 : i100
  %c4_i100 = hw.constant 4 : i100
  %c2_i100 = hw.constant 2 : i100
  %c1_i100 = hw.constant 1 : i100
  %2 = comb.extract %arg0 from 0 : (i100) -> i1
  %3 = comb.extract %arg0 from 1 : (i100) -> i1
  %4 = comb.extract %arg0 from 2 : (i100) -> i1
  %5 = comb.extract %arg0 from 3 : (i100) -> i1
  %6 = comb.extract %arg0 from 4 : (i100) -> i1
  %7 = comb.extract %arg0 from 5 : (i100) -> i1
  %8 = comb.extract %arg0 from 6 : (i100) -> i1
  %9 = comb.extract %arg0 from 7 : (i100) -> i1
  %10 = comb.extract %arg0 from 8 : (i100) -> i1
  %11 = comb.extract %arg0 from 9 : (i100) -> i1
  %12 = comb.extract %arg0 from 10 : (i100) -> i1
  %13 = comb.extract %arg0 from 11 : (i100) -> i1
  %14 = comb.extract %arg0 from 12 : (i100) -> i1
  %15 = comb.extract %arg0 from 13 : (i100) -> i1
  %16 = comb.extract %arg0 from 14 : (i100) -> i1
  %17 = comb.extract %arg0 from 15 : (i100) -> i1
  %18 = comb.extract %arg0 from 16 : (i100) -> i1
  %19 = comb.extract %arg0 from 17 : (i100) -> i1
  %20 = comb.extract %arg0 from 18 : (i100) -> i1
  %21 = comb.extract %arg0 from 19 : (i100) -> i1
  %22 = comb.extract %arg0 from 20 : (i100) -> i1
  %23 = comb.extract %arg0 from 21 : (i100) -> i1
  %24 = comb.extract %arg0 from 22 : (i100) -> i1
  %25 = comb.extract %arg0 from 23 : (i100) -> i1
  %26 = comb.extract %arg0 from 24 : (i100) -> i1
  %27 = comb.extract %arg0 from 25 : (i100) -> i1
  %28 = comb.extract %arg0 from 26 : (i100) -> i1
  %29 = comb.extract %arg0 from 27 : (i100) -> i1
  %30 = comb.extract %arg0 from 28 : (i100) -> i1
  %31 = comb.extract %arg0 from 29 : (i100) -> i1
  %32 = comb.extract %arg0 from 30 : (i100) -> i1
  %33 = comb.extract %arg0 from 31 : (i100) -> i1
  %34 = comb.extract %arg0 from 32 : (i100) -> i1
  %35 = comb.extract %arg0 from 33 : (i100) -> i1
  %36 = comb.extract %arg0 from 34 : (i100) -> i1
  %37 = comb.extract %arg0 from 35 : (i100) -> i1
  %38 = comb.extract %arg0 from 36 : (i100) -> i1
  %39 = comb.extract %arg0 from 37 : (i100) -> i1
  %40 = comb.extract %arg0 from 38 : (i100) -> i1
  %41 = comb.extract %arg0 from 39 : (i100) -> i1
  %42 = comb.extract %arg0 from 40 : (i100) -> i1
  %43 = comb.extract %arg0 from 41 : (i100) -> i1
  %44 = comb.extract %arg0 from 42 : (i100) -> i1
  %45 = comb.extract %arg0 from 43 : (i100) -> i1
  %46 = comb.extract %arg0 from 44 : (i100) -> i1
  %47 = comb.extract %arg0 from 45 : (i100) -> i1
  %48 = comb.extract %arg0 from 46 : (i100) -> i1
  %49 = comb.extract %arg0 from 47 : (i100) -> i1
  %50 = comb.extract %arg0 from 48 : (i100) -> i1
  %51 = comb.extract %arg0 from 49 : (i100) -> i1
  %52 = comb.extract %arg0 from 50 : (i100) -> i1
  %53 = comb.extract %arg0 from 51 : (i100) -> i1
  %54 = comb.extract %arg0 from 52 : (i100) -> i1
  %55 = comb.extract %arg0 from 53 : (i100) -> i1
  %56 = comb.extract %arg0 from 54 : (i100) -> i1
  %57 = comb.extract %arg0 from 55 : (i100) -> i1
  %58 = comb.extract %arg0 from 56 : (i100) -> i1
  %59 = comb.extract %arg0 from 57 : (i100) -> i1
  %60 = comb.extract %arg0 from 58 : (i100) -> i1
  %61 = comb.extract %arg0 from 59 : (i100) -> i1
  %62 = comb.extract %arg0 from 60 : (i100) -> i1
  %63 = comb.extract %arg0 from 61 : (i100) -> i1
  %64 = comb.extract %arg0 from 62 : (i100) -> i1
  %65 = comb.extract %arg0 from 63 : (i100) -> i1
  %66 = comb.extract %arg0 from 64 : (i100) -> i1
  %67 = comb.extract %arg0 from 65 : (i100) -> i1
  %68 = comb.extract %arg0 from 66 : (i100) -> i1
  %69 = comb.extract %arg0 from 67 : (i100) -> i1
  %70 = comb.extract %arg0 from 68 : (i100) -> i1
  %71 = comb.extract %arg0 from 69 : (i100) -> i1
  %72 = comb.extract %arg0 from 70 : (i100) -> i1
  %73 = comb.extract %arg0 from 71 : (i100) -> i1
  %74 = comb.extract %arg0 from 72 : (i100) -> i1
  %75 = comb.extract %arg0 from 73 : (i100) -> i1
  %76 = comb.extract %arg0 from 74 : (i100) -> i1
  %77 = comb.extract %arg0 from 75 : (i100) -> i1
  %78 = comb.extract %arg0 from 76 : (i100) -> i1
  %79 = comb.extract %arg0 from 77 : (i100) -> i1
  %80 = comb.extract %arg0 from 78 : (i100) -> i1
  %81 = comb.extract %arg0 from 79 : (i100) -> i1
  %82 = comb.extract %arg0 from 80 : (i100) -> i1
  %83 = comb.extract %arg0 from 81 : (i100) -> i1
  %84 = comb.extract %arg0 from 82 : (i100) -> i1
  %85 = comb.extract %arg0 from 83 : (i100) -> i1
  %86 = comb.extract %arg0 from 84 : (i100) -> i1
  %87 = comb.extract %arg0 from 85 : (i100) -> i1
  %88 = comb.extract %arg0 from 86 : (i100) -> i1
  %89 = comb.extract %arg0 from 87 : (i100) -> i1
  %90 = comb.extract %arg0 from 88 : (i100) -> i1
  %91 = comb.extract %arg0 from 89 : (i100) -> i1
  %92 = comb.extract %arg0 from 90 : (i100) -> i1
  %93 = comb.extract %arg0 from 91 : (i100) -> i1
  %94 = comb.extract %arg0 from 92 : (i100) -> i1
  %95 = comb.extract %arg0 from 93 : (i100) -> i1
  %96 = comb.extract %arg0 from 94 : (i100) -> i1
  %97 = comb.extract %arg0 from 95 : (i100) -> i1
  %98 = comb.extract %arg0 from 96 : (i100) -> i1
  %99 = comb.extract %arg0 from 97 : (i100) -> i1
  %100 = comb.extract %arg0 from 98 : (i100) -> i1
  %101 = comb.extract %arg0 from 99 : (i100) -> i1
  %102 = comb.concat %101, %c0_i99 : i1, i99
  %103 = comb.mux bin %100, %c316912650057057350374175801344_i100, %102 : i100
  %104 = comb.mux bin %99, %c158456325028528675187087900672_i100, %103 : i100
  %105 = comb.mux bin %98, %c79228162514264337593543950336_i100, %104 : i100
  %106 = comb.mux bin %97, %c39614081257132168796771975168_i100, %105 : i100
  %107 = comb.mux bin %96, %c19807040628566084398385987584_i100, %106 : i100
  %108 = comb.mux bin %95, %c9903520314283042199192993792_i100, %107 : i100
  %109 = comb.mux bin %94, %c4951760157141521099596496896_i100, %108 : i100
  %110 = comb.mux bin %93, %c2475880078570760549798248448_i100, %109 : i100
  %111 = comb.mux bin %92, %c1237940039285380274899124224_i100, %110 : i100
  %112 = comb.mux bin %91, %c618970019642690137449562112_i100, %111 : i100
  %113 = comb.mux bin %90, %c309485009821345068724781056_i100, %112 : i100
  %114 = comb.mux bin %89, %c154742504910672534362390528_i100, %113 : i100
  %115 = comb.mux bin %88, %c77371252455336267181195264_i100, %114 : i100
  %116 = comb.mux bin %87, %c38685626227668133590597632_i100, %115 : i100
  %117 = comb.mux bin %86, %c19342813113834066795298816_i100, %116 : i100
  %118 = comb.mux bin %85, %c9671406556917033397649408_i100, %117 : i100
  %119 = comb.mux bin %84, %c4835703278458516698824704_i100, %118 : i100
  %120 = comb.mux bin %83, %c2417851639229258349412352_i100, %119 : i100
  %121 = comb.mux bin %82, %c1208925819614629174706176_i100, %120 : i100
  %122 = comb.mux bin %81, %c604462909807314587353088_i100, %121 : i100
  %123 = comb.mux bin %80, %c302231454903657293676544_i100, %122 : i100
  %124 = comb.mux bin %79, %c151115727451828646838272_i100, %123 : i100
  %125 = comb.mux bin %78, %c75557863725914323419136_i100, %124 : i100
  %126 = comb.mux bin %77, %c37778931862957161709568_i100, %125 : i100
  %127 = comb.mux bin %76, %c18889465931478580854784_i100, %126 : i100
  %128 = comb.mux bin %75, %c9444732965739290427392_i100, %127 : i100
  %129 = comb.mux bin %74, %c4722366482869645213696_i100, %128 : i100
  %130 = comb.mux bin %73, %c2361183241434822606848_i100, %129 : i100
  %131 = comb.mux bin %72, %c1180591620717411303424_i100, %130 : i100
  %132 = comb.mux bin %71, %c590295810358705651712_i100, %131 : i100
  %133 = comb.mux bin %70, %c295147905179352825856_i100, %132 : i100
  %134 = comb.mux bin %69, %c147573952589676412928_i100, %133 : i100
  %135 = comb.mux bin %68, %c73786976294838206464_i100, %134 : i100
  %136 = comb.mux bin %67, %c36893488147419103232_i100, %135 : i100
  %137 = comb.mux bin %66, %c18446744073709551616_i100, %136 : i100
  %138 = comb.mux bin %65, %c9223372036854775808_i100, %137 : i100
  %139 = comb.mux bin %64, %c4611686018427387904_i100, %138 : i100
  %140 = comb.mux bin %63, %c2305843009213693952_i100, %139 : i100
  %141 = comb.mux bin %62, %c1152921504606846976_i100, %140 : i100
  %142 = comb.mux bin %61, %c576460752303423488_i100, %141 : i100
  %143 = comb.mux bin %60, %c288230376151711744_i100, %142 : i100
  %144 = comb.mux bin %59, %c144115188075855872_i100, %143 : i100
  %145 = comb.mux bin %58, %c72057594037927936_i100, %144 : i100
  %146 = comb.mux bin %57, %c36028797018963968_i100, %145 : i100
  %147 = comb.mux bin %56, %c18014398509481984_i100, %146 : i100
  %148 = comb.mux bin %55, %c9007199254740992_i100, %147 : i100
  %149 = comb.mux bin %54, %c4503599627370496_i100, %148 : i100
  %150 = comb.mux bin %53, %c2251799813685248_i100, %149 : i100
  %151 = comb.mux bin %52, %c1125899906842624_i100, %150 : i100
  %152 = comb.mux bin %51, %c562949953421312_i100, %151 : i100
  %153 = comb.mux bin %50, %c281474976710656_i100, %152 : i100
  %154 = comb.mux bin %49, %c140737488355328_i100, %153 : i100
  %155 = comb.mux bin %48, %c70368744177664_i100, %154 : i100
  %156 = comb.mux bin %47, %c35184372088832_i100, %155 : i100
  %157 = comb.mux bin %46, %c17592186044416_i100, %156 : i100
  %158 = comb.mux bin %45, %c8796093022208_i100, %157 : i100
  %159 = comb.mux bin %44, %c4398046511104_i100, %158 : i100
  %160 = comb.mux bin %43, %c2199023255552_i100, %159 : i100
  %161 = comb.mux bin %42, %c1099511627776_i100, %160 : i100
  %162 = comb.mux bin %41, %c549755813888_i100, %161 : i100
  %163 = comb.mux bin %40, %c274877906944_i100, %162 : i100
  %164 = comb.mux bin %39, %c137438953472_i100, %163 : i100
  %165 = comb.mux bin %38, %c68719476736_i100, %164 : i100
  %166 = comb.mux bin %37, %c34359738368_i100, %165 : i100
  %167 = comb.mux bin %36, %c17179869184_i100, %166 : i100
  %168 = comb.mux bin %35, %c8589934592_i100, %167 : i100
  %169 = comb.mux bin %34, %c4294967296_i100, %168 : i100
  %170 = comb.mux bin %33, %c2147483648_i100, %169 : i100
  %171 = comb.mux bin %32, %c1073741824_i100, %170 : i100
  %172 = comb.mux bin %31, %c536870912_i100, %171 : i100
  %173 = comb.mux bin %30, %c268435456_i100, %172 : i100
  %174 = comb.mux bin %29, %c134217728_i100, %173 : i100
  %175 = comb.mux bin %28, %c67108864_i100, %174 : i100
  %176 = comb.mux bin %27, %c33554432_i100, %175 : i100
  %177 = comb.mux bin %26, %c16777216_i100, %176 : i100
  %178 = comb.mux bin %25, %c8388608_i100, %177 : i100
  %179 = comb.mux bin %24, %c4194304_i100, %178 : i100
  %180 = comb.mux bin %23, %c2097152_i100, %179 : i100
  %181 = comb.mux bin %22, %c1048576_i100, %180 : i100
  %182 = comb.mux bin %21, %c524288_i100, %181 : i100
  %183 = comb.mux bin %20, %c262144_i100, %182 : i100
  %184 = comb.mux bin %19, %c131072_i100, %183 : i100
  %185 = comb.mux bin %18, %c65536_i100, %184 : i100
  %186 = comb.mux bin %17, %c32768_i100, %185 : i100
  %187 = comb.mux bin %16, %c16384_i100, %186 : i100
  %188 = comb.mux bin %15, %c8192_i100, %187 : i100
  %189 = comb.mux bin %14, %c4096_i100, %188 : i100
  %190 = comb.mux bin %13, %c2048_i100, %189 : i100
  %191 = comb.mux bin %12, %c1024_i100, %190 : i100
  %192 = comb.mux bin %11, %c512_i100, %191 : i100
  %193 = comb.mux bin %10, %c256_i100, %192 : i100
  %194 = comb.mux bin %9, %c128_i100, %193 : i100
  %195 = comb.mux bin %8, %c64_i100, %194 : i100
  %196 = comb.mux bin %7, %c32_i100, %195 : i100
  %197 = comb.mux bin %6, %c16_i100, %196 : i100
  %198 = comb.mux bin %5, %c8_i100, %197 : i100
  %199 = comb.mux bin %4, %c4_i100, %198 : i100
  %200 = comb.mux bin %3, %c2_i100, %199 : i100
  %201 = comb.mux bin %2, %c1_i100, %200 : i100
  return %201 : i100

  // CHECK-NEXT: %c0_i93 = hw.constant 0 : i93
  // CHECK-NEXT: %c1_i100 = hw.constant 1 : i100
  // CHECK-NEXT: [[LZC:%.+]] = arc.zero_count trailing %arg0 : i100
  // CHECK-NEXT: [[EXT:%.+]] = comb.extract [[LZC]] from 0 : (i100) -> i7
  // CHECK-NEXT: [[ZEXT:%.+]] = comb.concat %c0_i93, [[EXT]] : i93, i7
  // CHECK-NEXT: [[SHL:%.+]] = comb.shl %c1_i100, [[ZEXT]] : i100
  // CHECK-NEXT: return [[SHL]] : i100
}

// CHECK-LABEL: @arrayGetOfConstAggregate
func.func @arrayGetOfConstAggregate(%arg0: i3) -> (i3, i32, i3, i32) {
  %0 = hw.aggregate_constant [0 : i3, 1 : i3, 2 : i3, 3 : i3, 4 : i3, 5 : i3, 6 : i3, 7 : i3] : !hw.array<8xi3>
  %1 = hw.aggregate_constant [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32] : !hw.array<8xi32>
  %2 = hw.aggregate_constant [0 : i3, 1 : i3, 2 : i3, 4 : i3, 4 : i3, 5 : i3, 6 : i3, 7 : i3] : !hw.array<8xi3>

  %3 = hw.array_get %0[%arg0] : !hw.array<8xi3>, i3
  %4 = hw.array_get %1[%arg0] : !hw.array<8xi32>, i3
  %5 = hw.array_get %2[%arg0] : !hw.array<8xi3>, i3

  %6 = hw.aggregate_constant [1 : i32, 2 : i32, 4 : i32, 8 : i32, 16 : i32, 32 : i32, 64 : i32, 128 : i32] : !hw.array<8xi32>
  %7 = hw.array_get %6[%arg0] : !hw.array<8xi32>, i3

  return %3, %4, %5, %7 : i3, i32, i3, i32

  // CHECK-NEXT: %c0_i29 = hw.constant 0 : i29
  // CHECK-NEXT: %c1_i32 = hw.constant 1 : i32
  // CHECK-NEXT: [[ARR:%.+]] = hw.aggregate_constant [0 : i3, 1 : i3, 2 : i3, -4 : i3, -4 : i3, -3 : i3, -2 : i3, -1 : i3] : !hw.array<8xi3>
  // CHECK-NEXT: [[ZEXT:%.+]] = comb.concat %c0_i29, %arg0 : i29, i3
  // CHECK-NEXT: [[INVALID:%.+]] = hw.array_get [[ARR]][%arg0] : !hw.array<8xi3>, i3
  // CHECK-NEXT: [[ZEXT2:%.+]] = comb.concat %c0_i29, %arg0 : i29, i3
  // CHECK-NEXT: [[SHL:%.+]] = comb.shl %c1_i32, [[ZEXT2]] : i32
  // CHECK-NEXT: return %arg0, [[ZEXT]], [[INVALID]], [[SHL]] : i3, i32, i3, i32
}

hw.module @BreakpointUnit(%arg0: i2, %arg1: i1) -> (out0: i1) {
  %0 = comb.extract %arg0 from 1 : (i2) -> i1
  %2 = comb.extract %arg0 from 0 : (i2) -> i1
  %3 = comb.mux bin %0, %arg1, %2 : i1
  hw.output %3 : i1
}

hw.module @MulAddRecFNToRaw_postMul_arc_split_6(%arg0: i51) -> (out0: i5) {
  %c0_i2 = hw.constant 0 : i2
  %c-4_i4 = hw.constant -4 : i4
  %c-9_i5 = hw.constant -9 : i5
  %c-10_i5 = hw.constant -10 : i5
  %c-11_i5 = hw.constant -11 : i5
  %c-12_i5 = hw.constant -12 : i5
  %c-13_i5 = hw.constant -13 : i5
  %c-14_i5 = hw.constant -14 : i5
  %c-15_i5 = hw.constant -15 : i5
  %c-16_i5 = hw.constant -16 : i5
  %c15_i5 = hw.constant 15 : i5
  %c14_i5 = hw.constant 14 : i5
  %c13_i5 = hw.constant 13 : i5
  %c12_i5 = hw.constant 12 : i5
  %c11_i5 = hw.constant 11 : i5
  %c10_i5 = hw.constant 10 : i5
  %c9_i5 = hw.constant 9 : i5
  %c8_i5 = hw.constant 8 : i5
  %c7_i5 = hw.constant 7 : i5
  %c6_i5 = hw.constant 6 : i5
  %c5_i5 = hw.constant 5 : i5
  %c4_i5 = hw.constant 4 : i5
  %c3_i5 = hw.constant 3 : i5
  %c2_i5 = hw.constant 2 : i5
  %c1_i5 = hw.constant 1 : i5
  %c0_i5 = hw.constant 0 : i5
  %0 = comb.extract %arg0 from 50 : (i51) -> i1
  %1 = comb.extract %arg0 from 48 : (i51) -> i2
  %2 = comb.icmp bin ne %1, %c0_i2 : i2
  %3 = comb.extract %arg0 from 46 : (i51) -> i2
  %4 = comb.icmp bin ne %3, %c0_i2 : i2
  %5 = comb.extract %arg0 from 44 : (i51) -> i2
  %6 = comb.icmp bin ne %5, %c0_i2 : i2
  %7 = comb.extract %arg0 from 42 : (i51) -> i2
  %8 = comb.icmp bin ne %7, %c0_i2 : i2
  %9 = comb.extract %arg0 from 40 : (i51) -> i2
  %10 = comb.icmp bin ne %9, %c0_i2 : i2
  %11 = comb.extract %arg0 from 38 : (i51) -> i2
  %12 = comb.icmp bin ne %11, %c0_i2 : i2
  %13 = comb.extract %arg0 from 36 : (i51) -> i2
  %14 = comb.icmp bin ne %13, %c0_i2 : i2
  %15 = comb.extract %arg0 from 34 : (i51) -> i2
  %16 = comb.icmp bin ne %15, %c0_i2 : i2
  %17 = comb.extract %arg0 from 32 : (i51) -> i2
  %18 = comb.icmp bin ne %17, %c0_i2 : i2
  %19 = comb.extract %arg0 from 30 : (i51) -> i2
  %20 = comb.icmp bin ne %19, %c0_i2 : i2
  %21 = comb.extract %arg0 from 28 : (i51) -> i2
  %22 = comb.icmp bin ne %21, %c0_i2 : i2
  %23 = comb.extract %arg0 from 26 : (i51) -> i2
  %24 = comb.icmp bin ne %23, %c0_i2 : i2
  %25 = comb.extract %arg0 from 24 : (i51) -> i2
  %26 = comb.icmp bin ne %25, %c0_i2 : i2
  %27 = comb.extract %arg0 from 22 : (i51) -> i2
  %28 = comb.icmp bin ne %27, %c0_i2 : i2
  %29 = comb.extract %arg0 from 20 : (i51) -> i2
  %30 = comb.icmp bin ne %29, %c0_i2 : i2
  %31 = comb.extract %arg0 from 18 : (i51) -> i2
  %32 = comb.icmp bin ne %31, %c0_i2 : i2
  %33 = comb.extract %arg0 from 16 : (i51) -> i2
  %34 = comb.icmp bin ne %33, %c0_i2 : i2
  %35 = comb.extract %arg0 from 14 : (i51) -> i2
  %36 = comb.icmp bin ne %35, %c0_i2 : i2
  %37 = comb.extract %arg0 from 12 : (i51) -> i2
  %38 = comb.icmp bin ne %37, %c0_i2 : i2
  %39 = comb.extract %arg0 from 10 : (i51) -> i2
  %40 = comb.icmp bin ne %39, %c0_i2 : i2
  %41 = comb.extract %arg0 from 8 : (i51) -> i2
  %42 = comb.icmp bin ne %41, %c0_i2 : i2
  %43 = comb.extract %arg0 from 6 : (i51) -> i2
  %44 = comb.icmp bin ne %43, %c0_i2 : i2
  %45 = comb.extract %arg0 from 4 : (i51) -> i2
  %46 = comb.icmp bin ne %45, %c0_i2 : i2
  %47 = comb.extract %arg0 from 2 : (i51) -> i2
  %48 = comb.icmp bin eq %47, %c0_i2 : i2
  %49 = comb.concat %c-4_i4, %48 : i4, i1
  %50 = comb.mux bin %46, %c-9_i5, %49 : i5
  %51 = comb.mux bin %44, %c-10_i5, %50 : i5
  %52 = comb.mux bin %42, %c-11_i5, %51 : i5
  %53 = comb.mux bin %40, %c-12_i5, %52 : i5
  %54 = comb.mux bin %38, %c-13_i5, %53 : i5
  %55 = comb.mux bin %36, %c-14_i5, %54 : i5
  %56 = comb.mux bin %34, %c-15_i5, %55 : i5
  %57 = comb.mux bin %32, %c-16_i5, %56 : i5
  %58 = comb.mux bin %30, %c15_i5, %57 : i5
  %59 = comb.mux bin %28, %c14_i5, %58 : i5
  %60 = comb.mux bin %26, %c13_i5, %59 : i5
  %61 = comb.mux bin %24, %c12_i5, %60 : i5
  %62 = comb.mux bin %22, %c11_i5, %61 : i5
  %63 = comb.mux bin %20, %c10_i5, %62 : i5
  %64 = comb.mux bin %18, %c9_i5, %63 : i5
  %65 = comb.mux bin %16, %c8_i5, %64 : i5
  %66 = comb.mux bin %14, %c7_i5, %65 : i5
  %67 = comb.mux bin %12, %c6_i5, %66 : i5
  %68 = comb.mux bin %10, %c5_i5, %67 : i5
  %69 = comb.mux bin %8, %c4_i5, %68 : i5
  %70 = comb.mux bin %6, %c3_i5, %69 : i5
  %71 = comb.mux bin %4, %c2_i5, %70 : i5
  %72 = comb.mux bin %2, %c1_i5, %71 : i5
  %73 = comb.mux bin %0, %c0_i5, %72 : i5
  hw.output %73 : i5
}
