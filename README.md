## AnML
* Simple ML inspired language
* User friendly syntax
* Hindley-Milner type system with type inference

```python
$ python anml.py # recommended: rlwrap python anml.py
# blocks use '->' for a single expression, and ':' ... 'end' for multiple declarations + expression
>>> def sqr(x) -> x * x
>>> sqr(5) + 9
34.0
>>> sqr(5) + 9 > 33
True
>>> sqr(5) + 9 > 33 and sqr(5) + 9 < 35
True
>>> x = 42
42.0
>>> x == true
Type Mismatch: float != bool
>>> def double(x):
       y = 2
       x * y
    end
>>> double(5)
10.0
>>> y = 3
>>> def triple(x) -> x * y
>>> triple(5)
15.0
>>> y = 2
>>> triple(5)
15.0
# if/else is an expression
>>> def min(a,b) ->
      if a < b -> a
      else -> b

>>> min(10, 2)
2.0
>>> min(10, 20)
10.0
>>> def max(a,b) ->
      if (a > b) -> a
      else -> b

>>> max(10, 2)
10.0
>>> max(10, 20)
20.0
```
