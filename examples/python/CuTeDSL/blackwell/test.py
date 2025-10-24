import cutlass.cute as cute

@cute.jit
def test():
    layoutA = cute.make_layout((4, 4), stride=(4, 1))
    layoutA_inv = cute.left_inverse(layoutA) # stride is meaningless, index to coord is always column major
    layoutA_inv_with_shape = layoutA_inv.with_shape((8, 2))
    print(layoutA)
    print(layoutA_inv)
    print(layoutA_inv_with_shape)

test()