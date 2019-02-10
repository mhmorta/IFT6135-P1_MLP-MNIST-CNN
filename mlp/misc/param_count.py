def l(input, n_count):
    ret = (input+1) * n_count
    return ret


params = [700, 500]
h1 = l(784, params[0])
print('h1', h1)
h2 = l(params[0], params[1])
print('h2', h2)
o = l(params[1], 10)
print('o', o)

print(h1 + h2 + o)

# [500, 600] -> 699110
# [500, 500] -> 648010
# [600, 700] -> 898710
# [700, 500] -> 905010