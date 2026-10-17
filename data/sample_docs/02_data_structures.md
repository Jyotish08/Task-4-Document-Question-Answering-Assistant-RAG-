---
title: "Python Data Structures: Lists, Tuples, Sets, and Dictionaries"
source_url: "https://docs.python.org/3/tutorial/datastructures.html"
retrieval_date: "2026-10-17"
license: "Python Software Foundation License (PSFL)"
category: "Programming / Python Fundamentals"
---

# Python Data Structures: Lists, Tuples, Sets, and Dictionaries

## 1. Lists (`list`)
A list is an ordered, mutable sequence of arbitrary objects. Lists are declared using square brackets `[]` or the `list()` constructor.

### Key List Operations and Methods
- `list.append(x)`: Adds item `x` to the end of the list in O(1) amortized time.
- `list.extend(iterable)`: Extends the list by appending all items from the iterable.
- `list.insert(i, x)`: Inserts item `x` at position `i`. All subsequent elements are shifted.
- `list.remove(x)`: Removes the first item from the list whose value is equal to `x`. Raises `ValueError` if not found.
- `list.pop([i])`: Removes and returns the item at given index `i`. If no index is specified, `pop()` removes and returns the last item in O(1) time.
- `list.sort(key=None, reverse=False)`: Sorts the items of the list in place using the Timsort algorithm.

### List Comprehensions
List comprehensions provide a concise way to create lists without traditional loops:
```python
squares = [x**2 for x in range(10) if x % 2 == 0]
```
This produces `[0, 4, 16, 36, 64]`.

## 2. Tuples (`tuple`)
A tuple is an ordered, immutable sequence of elements. Tuples are written with parentheses `()` or comma-separated values:
```python
t = (1, "apple", 3.14)
singleton = (42,)  # Note the trailing comma required for single-element tuples
```
Because tuples are immutable, their contents cannot be modified, added to, or removed after instantiation. Tuples are hashable if all their elements are hashable, meaning they can serve as dictionary keys or set elements, whereas lists cannot.

Tuples support sequence unpacking:
```python
x, y, z = t
```

## 3. Sets (`set` and `frozenset`)
A set is an unordered collection of distinct, hashable objects. Common uses include membership testing, eliminating duplicates from a sequence, and computing mathematical operations such as union, intersection, difference, and symmetric difference.

Sets are written with curly braces `{}` containing elements (e.g., `{1, 2, 3}`) or instantiated via `set()`. Note that an empty set must be created with `set()`, because `{}` creates an empty dictionary.

### Set Operations
Given sets `a = {1, 2, 3}` and `b = {3, 4, 5}`:
- **Union (`a | b`)**: Returns `{1, 2, 3, 4, 5}` containing elements in either set.
- **Intersection (`a & b`)**: Returns `{3}` containing elements common to both sets.
- **Difference (`a - b`)**: Returns `{1, 2}` containing elements in `a` but not in `b`.
- **Symmetric Difference (`a ^ b`)**: Returns `{1, 2, 4, 5}` containing elements in either set, but not both.

Average time complexity for membership testing (`x in a`) is O(1). An immutable variant called `frozenset` is also available and is hashable.

## 4. Dictionaries (`dict`)
A dictionary is an associative array (or hash map) of key-value pairs. In Python 3.7+, dictionaries are guaranteed to maintain insertion order. Dictionaries are constructed with curly braces `{key: value}` or `dict()`.

### Dictionary Keys and Methods
Dictionary keys must be hashable and immutable (e.g., strings, numbers, or tuples containing only immutable objects). Values can be any arbitrary Python object.
- `d[key]`: Accesses the value associated with `key`. Raises `KeyError` if the key does not exist.
- `d.get(key, default=None)`: Safely returns the value for `key` if present; returns `default` if the key is missing.
- `d.keys()`: Returns a dynamic view of all dictionary keys.
- `d.values()`: Returns a dynamic view of all values.
- `d.items()`: Returns a dynamic view of `(key, value)` tuples.
- `d.pop(key)`: Removes the specified key and returns its corresponding value.
