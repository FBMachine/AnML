## AnML
* Simple ML inspired language
* Pythonic syntax
* Hindley-Milner type system with type inference

```python
$ python anml.py # recommended: rlwrap python anml.py
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
...    y = 2
...    x * y
... end
>>> double(5)
10.0
>>> y = 3
>>> def triple(x) -> x * y
>>> triple(5)
15.0
>>> y = 2
>>> triple(5)
15.0
```
