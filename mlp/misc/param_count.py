def l(input, n_count):
    ret = input * n_count + n_count
    return ret


h1 = l(784, 500)
print('h1', h1)
h2 = l(500, 500)
print('h2', h2)
o = l(500, 10)
print('o', o)

print(h1 + h2 + o)