def l(input, n_count):
    ret = (input+1) * n_count
    return ret


params = [500, 600]
h1 = l(784, params[0])
print('h1', h1)
h2 = l(params[0], params[1])
print('h2', h2)
o = l(params[1], 10)
print('o', o)

print(h1 + h2 + o)