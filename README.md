## AnML
* Simple ML inspired language
* User friendly syntax
* Hindley-Milner type system with type inference

```ruby
$ python anml.py # recommended: rlwrap python anml.py
# blocks use '->' for a single expression
>>> def sqr(x) -> x * x
def sqr : (a) -> a = <Function>
>>> sqr(5) + 9
34
>>> sqr(5) + 9 > 33
true
>>> sqr(5) + 9 > 33 and sqr(5) + 9 < 35
true
>>> x = 42
x : int = 42

>>> x == true
Type Mismatch: int != bool
# To declare local bindings, use block form for function definition:
# def fun():
#   <decl>[0]
#   ...
#   <decl>[n]
#   return <expr>
>>> def double(x):
        y = 2
        return x * y

def double : (int) -> int = <Function>
>>> double(5)
10

# Recursive functions:
>>> def pow(x, e) ->
      if e == 0 -> 1
      else -> x * pow(x, e - 1)

def pow : (int, int) -> int = <Function>
>>> pow(2, 32)
4294967296

# Variables in scope are captured by value:
>>> y = 3
y : int = 3
>>> def triple(x) -> x * y
def triple : (int) -> int = <Function>
>>> triple(5)
15

# Variables are immutable, so this will rebind y, shadowing previous definition:
>>> y = 2
y : int = 2

# y was captured with lexical scope when triple was defined, so does not change:
>>> triple(5)
15

# function are first class citizens:
>>> def call_func(f) -> f(4)
def call_func : ((int) -> a) -> a = <Function>
>>> call_func(sqr)
16
>>> sq = sqr
sq : (int) -> int = def sqr : (int) -> int = <Function>
>>> test(sq)
16

# More examples:
>>> def min(a,b) ->
      if a < b -> a
      else -> b

def min : (a, a) -> a = <Function>
>>> min(10, 2)
2
>>> min(10, 20)
10
>>> def max(a,b):
      return if (a > b) -> a
             else -> b

def max : (a, a) -> a = <Function>
>>> max(10,2)
10
>>> max(10,20)
20
>>> x = 123456789
x : int = 123456789
>>> y = x * x
y : int = 15241578750190521
```
