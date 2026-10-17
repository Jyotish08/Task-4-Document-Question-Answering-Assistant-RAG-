---
title: "Python Syntax, Variables, and Primitive Types"
source_url: "https://docs.python.org/3/tutorial/introduction.html"
retrieval_date: "2026-10-17"
license: "Python Software Foundation License (PSFL)"
category: "Programming / Python Fundamentals"
---

# Python Syntax, Variables, and Primitive Types

## 1. Syntax and Indentation
Python relies on whitespace indentation rather than curly brackets `{}` or keywords like `begin`/`end` to delimit code blocks. A standard Python block uses 4 spaces per indentation level. Mixing tabs and spaces in Python 3 raises an `IndentationError`. Python statements end at the end of a line, although backslashes `\` or parentheses `()` can be used for explicit or implicit line continuation across multiple lines.

Comments in Python start with the hash character `#` and extend to the physical end of the line. Multi-line string literals enclosed in triple quotes `'''` or `"""` can also serve as documentation strings (docstrings) when placed immediately after function, class, or module definitions.

## 2. Variables and Dynamic Typing
Python is a dynamically and strongly typed language:
- **Dynamically typed**: Variables do not require explicit type declarations. A variable name is merely a reference to an object in memory. Assigning a value binds the name to that object: `x = 42`.
- **Strongly typed**: Python does not automatically coerce incompatible types in operations. For example, evaluating `"Score: " + 10` raises a `TypeError` instead of implicitly converting the integer to a string. The programmer must explicitly convert the integer using `str(10)`.

Variable identifiers in Python are case-sensitive, must begin with a letter (a-z, A-Z) or underscore `_`, and may contain digits (0-9). Reserved keywords like `def`, `class`, `if`, and `return` cannot be used as variable names.

## 3. Numeric Types
Python has three built-in numeric types:
1. **Integers (`int`)**: Integers in Python 3 have arbitrary precision (unlimited length), limited only by the available machine memory. Examples: `42`, `-100`, `1_000_000`.
2. **Floating-point numbers (`float`)**: Represented using standard C double-precision 64-bit IEEE 754 floating-point values. Floating-point division using the `/` operator always returns a `float` (e.g., `7 / 2` evaluates to `3.5`). Integer division (floor division) is performed using the `//` operator (e.g., `7 // 2` evaluates to `3`).
3. **Complex numbers (`complex`)**: Written in the form `real + imag*j`, where `j` represents the imaginary unit. Example: `z = 3 + 4j`.

Numeric objects in Python are immutable; modifying an integer or float variable creates a new object rather than altering the existing one in memory.

## 4. Text Sequence Type: Strings (`str`)
Strings in Python are immutable sequences of Unicode characters. They can be enclosed in single quotes `'...'` or double quotes `"..."`.

### String Indexing and Slicing
Strings support 0-based indexing from the left and negative indexing from the right:
- Given `s = "Python"`:
  - `s[0]` is `'P'`
  - `s[-1]` is `'n'`
- Slicing uses the syntax `s[start:stop:step]`:
  - `s[0:2]` returns `'Py'` (character at index `stop` is excluded).
  - `s[::-1]` returns the reversed string `'nohtyP'`.

Because strings are immutable, attempting to assign to an index (e.g., `s[0] = 'J'`) raises a `TypeError`. String formatting is commonly performed using f-strings: `f"Hello, {name}"`.

## 5. Booleans and Truth Value Testing
The boolean type `bool` has exactly two constant instances: `True` and `False`. In Python, `bool` is a subclass of `int`, where `True` has the numerical value `1` and `False` has the numerical value `0`.

The following values are considered false in boolean contexts:
- Constants: `None` and `False`
- Zero of any numeric type: `0`, `0.0`, `0j`
- Empty sequences and collections: `""`, `()`, `[]`, `{}`, `set()`

All other values are evaluated as true. The `None` object represents the absence of a value or a null value and is the sole instance of `NoneType`.
