def func(b, **kwargs):
    if 'a' in kwargs.keys():
        print(kwargs['a'])
    else:
        print('not exist')

def foo(b, *args):
    if len(args) > 0:
        print(args[0])
    else:
        print('not exist')

foo(1)