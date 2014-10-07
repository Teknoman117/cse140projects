xor $s0, $s0, $s0    # int total = 0
xor $s1, $s1, $s1    # char* array = 0x00000000

# for(int x = 0; x < 64; x++) array[x] = x
_lp1_begin:
    xor $s2, $s2, $s2    # x = 0

_lp1:
    # loop logic
    sltiu $t2, $s2, 64
    beq $t2, $0, _lp2_begin 

    # compute index
    addu $t0, $s1, $s2
    sb $s2, 0($t0)
    addiu $s2, $s2, 1
    j _lp1


# for(int x = 0; x < 64; x++) total += array[x]
_lp2_begin:
    xor $s2, $s2, $s2    # x = 0

_lp2:
    # loop logic
    sltiu $t2, $s2, 64
    beq $t2, $0, _lp3_begin 

    # compute index
    addu $t0, $s1, $s2
    lb $t1, 0($t0)
    addu $s0, $s0, $t1
    addiu $s2, $s2, 1
    j _lp2

# for(int x = 0; x < 64; x++) total += array[x]
_lp3_begin:
    xor $s2, $s2, $s2    # x = 0

_lp3:
    # loop logic
    sltiu $t2, $s2, 64
    beq $t2, $0, end 

    # compute index
    addu $t0, $s1, $s2
    lb $t1, 0($t0)
    addu $s0, $s0, $t1
    addiu $s2, $s2, 1
    j _lp3

end:
    j end

