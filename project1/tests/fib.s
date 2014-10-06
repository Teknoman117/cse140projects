# Initialize fibinocci sequence
init: 
    xor $t0, $t0, $t0
    xor $t1, $t1, $t1
    ori $t1, $t1, 1

    # use s0 as the current memory address
    xor $s0, $s0, $s0
    ori $s0, $s0, 128
    sw $t0, 0($s0)
    sw $t1, 4($s0)
    addiu $s0, $s0, 8

fib:
    addu $t2, $t0, $t1
    addiu $t1, $t0, 0    # mov
    addiu $t0, $t2, 0    # mov
    sw $t2, 0($s0)
    addiu $s0, $s0, 4
    j fib

