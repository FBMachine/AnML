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
```
