count:
    sw $t0, 0($s0)
    addiu $s0, $s0, 4
    addiu $t0, $t0, 1
    j count

