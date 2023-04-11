// RUN: circt-opt %s --arc-lower-state | FileCheck %s

// CHECK-LABEL: arc.model "Empty" {
// CHECK-NEXT:  }
hw.module @Empty() {
}

// CHECK-LABEL: arc.model "InputsAndOutputs" {
hw.module @InputsAndOutputs(%a: i42, %b: i17) -> (c: i42, d: i17) {
  %0 = comb.add %a, %a : i42
  %1 = comb.add %b, %b : i17
  hw.output %0, %1 : i42, i17
  // CHECK-NEXT: [[INA:%.+]] = arc.root_input "a" <i42>
  // CHECK-NEXT: [[INB:%.+]] = arc.root_input "b" <i17>
  // CHECK-NEXT: arc.passthrough {
  // CHECK-NEXT:   [[A:%.+]] = arc.state_read [[INA]] : <i42>
  // CHECK-NEXT:   [[TMP:%.+]] = comb.add [[A]], [[A]] : i42
  // CHECK-NEXT:   arc.state_write [[OUTA:%.+]] = [[TMP]] : <i42>
  // CHECK-NEXT:   [[B:%.+]] = arc.state_read [[INB]] : <i17>
  // CHECK-NEXT:   [[TMP:%.+]] = comb.add [[B]], [[B]] : i17
  // CHECK-NEXT:   arc.state_write [[OUTB:%.+]] = [[TMP]] : <i17>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[OUTA]] = arc.root_output "c" <i42>
  // CHECK-NEXT: [[OUTB]] = arc.root_output "d" <i17>
}

// CHECK-LABEL: arc.model "State" {
hw.module @State(%clk: i1, %en: i1) {
  %gclk = arc.clock_gate %clk, %en
  %3 = arc.state @DummyArc(%6) clock %clk lat 1 : (i42) -> i42
  %4 = arc.state @DummyArc(%5) clock %gclk lat 1 : (i42) -> i42
  %5 = comb.add %3, %3 : i42
  %6 = comb.add %4, %4 : i42
  // CHECK-NEXT: [[INCLK:%.+]] = arc.root_input "clk" <i1>
  // CHECK-NEXT: [[INEN:%.+]] = arc.root_input "en" <i1>
  // CHECK-NEXT: [[CLK:%.+]] = arc.state_read [[INCLK]] : <i1>
  // CHECK-NEXT: arc.clock_tree [[CLK]] {
  // CHECK-NEXT:   [[TMP0:%.+]] = arc.state_read [[S1:%.+]] : <i42>
  // CHECK-NEXT:   [[TMP1:%.+]] = comb.add [[TMP0]], [[TMP0]]
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.state @DummyArc([[TMP1]]) lat 0 : (i42) -> i42
  // CHECK-NEXT:   arc.state_write [[S0:%.+]] = [[TMP2]] : <i42>
  // CHECK-NEXT:   [[TMP0:%.+]] = arc.state_read [[S0]] : <i42>
  // CHECK-NEXT:   [[TMP1:%.+]] = comb.add [[TMP0]], [[TMP0]]
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.state @DummyArc([[TMP1]]) lat 0 : (i42) -> i42
  // CHECK-NEXT:   [[EN:%.+]] = arc.state_read [[INEN]] : <i1>
  // CHECK-NEXT:   arc.state_write [[S1]] = [[TMP2]] if [[EN]] : <i42>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[S0]] = arc.alloc_state <i42>
  // CHECK-NEXT: [[S1]] = arc.alloc_state <i42>
}

// CHECK-LABEL: arc.model "State2" {
hw.module @State2(%clk: i1) {
  %3 = arc.state @DummyArc(%3) clock %clk lat 1 : (i42) -> i42
  %4 = arc.state @DummyArc(%4) clock %clk lat 1 : (i42) -> i42
  // CHECK-NEXT: %in_clk = arc.root_input "clk" <i1>
  // CHECK-NEXT: [[CLK:%.+]] = arc.state_read %in_clk : <i1>
  // CHECK-NEXT: arc.clock_tree [[CLK]] {
  // CHECK-NEXT:   [[TMP0:%.+]] = arc.state_read [[S0:%.+]] : <i42>
  // CHECK-NEXT:   [[TMP1:%.+]] = arc.state @DummyArc([[TMP0]]) lat 0 : (i42) -> i42
  // CHECK-NEXT:   arc.state_write [[S0]] = [[TMP1]] : <i42>
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.state_read [[S1:%.+]] : <i42>
  // CHECK-NEXT:   [[TMP3:%.+]] = arc.state @DummyArc([[TMP2]]) lat 0 : (i42) -> i42
  // CHECK-NEXT:   arc.state_write [[S1]] = [[TMP3]] : <i42>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[S0]] = arc.alloc_state <i42>
  // CHECK-NEXT: [[S1]] = arc.alloc_state <i42>
}

arc.define @DummyArc(%arg0: i42) -> i42 {
  arc.output %arg0 : i42
}

// CHECK-LABEL: arc.model "MemoryReadAndNonMaskedWrite"
hw.module @MemoryReadAndNonMaskedWrite(%clk0: i1) {
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c9001_i42 = hw.constant 9001 : i42
  %mem = arc.memory <4 x i42>
  arc.memory_write_port %mem[%c0_i2], %true, %c9001_i42 clock %clk0 : <4 x i42>, i2
  // COM: also checks that the read is properly reordered to be before the write
  %read0 = arc.memory_read_port %mem[%c0_i2], %true clock %clk0 : <4 x i42>, i2

  // CHECK-NEXT: [[INCLK0:%.+]] = arc.root_input "clk0" <i1>
  // CHECK-NEXT: [[CLK0:%.+]] = arc.state_read [[INCLK0]] : <i1>
  // CHECK-NEXT: arc.clock_tree [[CLK0]] {
  // CHECK:        %2 = scf.if %true -> (i42) {
  // CHECK-NEXT:     [[TMP:%.+]] = arc.memory_read [[MEM:%.+]][%c0_i2] : <4 x i42>, i2
  // CHECK-NEXT:     scf.yield [[TMP]] : i42
  // CHECK-NEXT:   } else {
  // CHECK:          scf.yield %c0_i42 : i42
  // CHECK-NEXT:   }
  // CHECK:        arc.memory_write [[MEM]][%c0_i2], %true, %c9001_i42 : <4 x i42>, i2
  // CHECK-NEXT: }
  // CHECK-NEXT: [[MEM]] = arc.alloc_memory <4 x i42>
}

// CHECK-LABEL:  arc.model "maskedMemoryWrite"
hw.module @maskedMemoryWrite(%clk: i1) {
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c9001_i42 = hw.constant 9001 : i42
  %c1010_i42 = hw.constant 1010 : i42
  %mem = arc.memory <4 x i42>
  arc.memory_write_port %mem[%c0_i2], %true, %c9001_i42 mask (%c1010_i42 : i42) clock %clk : <4 x i42>, i2
}
// CHECK:      %c9001_i42 = hw.constant 9001 : i42
// CHECK:      %c1010_i42 = hw.constant 1010 : i42
// CHECK:      [[RD:%.+]] = arc.memory_read [[MEM:%.+]][%c0_i2] : <4 x i42>, i2
// CHECK:      %c-1_i42 = hw.constant -1 : i42
// CHECK:      [[NEG_MASK:%.+]] = comb.xor bin %c1010_i42, %c-1_i42 : i42
// CHECK:      [[OLD_MASKED:%.+]] = comb.and bin [[NEG_MASK]], [[RD]] : i42
// CHECK:      [[NEW_MASKED:%.+]] = comb.and bin %c1010_i42, %c9001_i42 : i42
// CHECK:      [[DATA:%.+]] = comb.or bin [[OLD_MASKED]], [[NEW_MASKED]] : i42
// CHECK:      arc.memory_write [[MEM]][%c0_i2], %true, [[DATA]] : <4 x i42>, i2

// CHECK-LABEL: arc.model "Taps"
hw.module @Taps() {
  // CHECK-NOT: arc.tap
  // CHECK-DAG: [[VALUE:%.+]] = hw.constant 0 : i42
  // CHECK-DAG: [[STATE:%.+]] = arc.alloc_state tap {name = "myTap"}
  // CHECK-DAG: arc.state_write [[STATE]] = [[VALUE]]
  %c0_i42 = hw.constant 0 : i42
  arc.tap %c0_i42 {name = "myTap"} : i42
}

// CHECK-LABEL: arc.model "MaterializeOpsWithRegions"
hw.module @MaterializeOpsWithRegions(%clk0: i1, %clk1: i1) -> (z: i42) {
  %true = hw.constant true
  %c19_i42 = hw.constant 19 : i42
  %0 = scf.if %true -> (i42) {
    scf.yield %c19_i42 : i42
  } else {
    %c42_i42 = hw.constant 42 : i42
    scf.yield %c42_i42 : i42
  }
  // CHECK:      arc.passthrough {
  // CHECK-NEXT:   %true = hw.constant true
  // CHECK-NEXT:   %c19_i42 = hw.constant 19
  // CHECK-NEXT:   [[TMP:%.+]] = scf.if %true -> (i42) {
  // CHECK-NEXT:     scf.yield %c19_i42
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %c42_i42 = hw.constant 42
  // CHECK-NEXT:     scf.yield %c42_i42
  // CHECK-NEXT:   }
  // CHECK-NEXT:   arc.state_write
  // CHECK-NEXT: }

  // CHECK:      [[CLK0:%.+]] = arc.state_read %in_clk0
  // CHECK-NEXT: arc.clock_tree [[CLK0]] {
  // CHECK-NEXT:   %true = hw.constant true
  // CHECK-NEXT:   %c19_i42 = hw.constant 19
  // CHECK-NEXT:   [[TMP:%.+]] = scf.if %true -> (i42) {
  // CHECK-NEXT:     scf.yield %c19_i42
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %c42_i42 = hw.constant 42
  // CHECK-NEXT:     scf.yield %c42_i42
  // CHECK-NEXT:   }
  // CHECK-NEXT:   arc.state @DummyArc([[TMP]]) lat 0
  // CHECK-NEXT:   arc.state_write
  // CHECK-NEXT: }

  // CHECK:      [[CLK1:%.+]] = arc.state_read %in_clk1
  // CHECK-NEXT: arc.clock_tree [[CLK1]] {
  // CHECK-NEXT:   %true = hw.constant true
  // CHECK-NEXT:   %c19_i42 = hw.constant 19
  // CHECK-NEXT:   [[TMP:%.+]] = scf.if %true -> (i42) {
  // CHECK-NEXT:     scf.yield %c19_i42
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %c42_i42 = hw.constant 42
  // CHECK-NEXT:     scf.yield %c42_i42
  // CHECK-NEXT:   }
  // CHECK-NEXT:   arc.state @DummyArc([[TMP]]) lat 0
  // CHECK-NEXT:   arc.state_write
  // CHECK-NEXT: }

  %1 = arc.state @DummyArc(%0) clock %clk0 lat 1 : (i42) -> i42
  %2 = arc.state @DummyArc(%0) clock %clk1 lat 1 : (i42) -> i42
  hw.output %0 : i42
}

arc.define @i1Identity(%arg0: i1) -> i1 {
  arc.output %arg0 : i1
}

arc.define @DummyArc2(%arg0: i42) -> (i42, i42) {
  arc.output %arg0, %arg0 : i42, i42
}

hw.module @stateReset(%clk: i1, %arg0: i42, %rst: i1) -> (out0: i42, out1: i42) {
  %0 = arc.state @i1Identity(%rst) lat 0 : (i1) -> (i1)
  %1 = arc.state @i1Identity(%rst) lat 0 : (i1) -> (i1)
  %2, %3 = arc.state @DummyArc2(%arg0) clock %clk enable %0 reset %1 lat 1 : (i42) -> (i42, i42)
  hw.output %2, %3 : i42, i42
}
// CHECK-LABEL: arc.model "stateReset"
// CHECK: arc.clock_tree %{{.*}} {
// CHECK:   [[IN_RST:%.+]] = arc.state_read %in_rst : <i1>
// CHECK:   [[EN:%.+]] = arc.state @i1Identity([[IN_RST]]) lat 0 : (i1) -> i1
// CHECK:   [[RST:%.+]] = arc.state @i1Identity([[IN_RST]]) lat 0 : (i1) -> i1
// CHECK:   scf.if [[RST]] {
// CHECK:     arc.state_write [[ALLOC1:%.+]] = %c0_i42{{.*}} : <i42>
// CHECK:     arc.state_write [[ALLOC2:%.+]] = %c0_i42{{.*}} : <i42>
// CHECK:   } else {
// CHECK:     [[ARG:%.+]] = arc.state_read %in_arg0 : <i42>
// CHECK:     [[STATE:%.+]]:2 = arc.state @DummyArc2([[ARG]]) lat 0 : (i42) -> (i42, i42)
// CHECK:     arc.state_write [[ALLOC1]] = [[STATE]]#0 if [[EN]] : <i42>
// CHECK:     arc.state_write [[ALLOC2]] = [[STATE]]#1 if [[EN]] : <i42>
// CHECK:   }
// CHECK: }
// CHECK: [[ALLOC1]] = arc.alloc_state <i42>
// CHECK: [[ALLOC2]] = arc.alloc_state <i42>
