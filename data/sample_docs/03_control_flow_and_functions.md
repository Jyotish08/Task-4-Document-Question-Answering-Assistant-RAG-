---
title: "Python Control Flow and Functions"
source_url: "https://docs.python.org/3/tutorial/controlflow.html"
retrieval_date: "2026-10-17"
license: "Python Software Foundation License (PSFL)"
category: "Programming / Python Fundamentals"
---

# Python Control Flow and Functions

## 1. Conditional Statements
Python uses `if`, `elif`, and `else` clauses to perform conditional execution:
```python
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"
```
The `elif` keyword is short for "else if" and avoids excessive nesting. Python also supports ternary conditional expressions in the form `x if condition else y`.

## 2. Iteration Statements: For and While Loops
### The `for` Statement
The `for` loop in Python iterates over the items of any iterable (such as a list, tuple, string, or range), in the order that they appear:
```python
for item in ["apple", "banana", "cherry"]:
    print(item)
```
The built-in function `range(start, stop, step)` generates arithmetic progressions. `range(5)` yields `0, 1, 2, 3, 4`.

### The `while` Statement
The `while` loop executes as long as its condition remains true:
```python
count = 0
while count < 5:
    count += 1
```

### Loop Control: `break`, `continue`, and `else` Clauses
- `break`: Terminates the innermost loop immediately.
- `continue`: Skips the remainder of the current iteration and advances to the next iteration.
- Loop `else` clause: Python loops have an optional `else` block. The `else` clause executes when the loop terminates through exhaustion of the iterable or when the condition becomes false, but NOT when the loop is terminated by a `break` statement.

## 3. Function Definitions and Parameters
Functions are defined using the `def` keyword followed by the function name, a parenthesized list of parameters, and a colon:
```python
def greet(name, greeting="Hello"):
    """Return a personalized greeting message."""
    return f"{greeting}, {name}!"
```

### Return Values
The `return` statement exits a function and optionally passes back an expression to the caller. A function without an explicit `return` statement, or with a plain `return` lacking an argument, returns `None`.

### Default Argument Values
Parameters can have default values. When a caller omits the argument, the default is used. Important caveat: default argument expressions are evaluated once when the function is defined, not each time the function is called. Therefore, mutable default arguments (such as `def append_to(item, target=[])`) retain modifications across successive invocations. The standard idiomatic solution is to use `None` as the default value:
```python
def append_to(item, target=None):
    if target is None:
        target = []
    target.append(item)
    return target
```

### Variable-Length Arguments (`*args` and `**kwargs`)
- `*args`: Collects extra positional arguments into a tuple.
- `**kwargs`: Collects extra keyword arguments into a dictionary.
```python
def log_event(event_type, *args, **kwargs):
    print(f"Type: {event_type}, Details: {args}, Metadata: {kwargs}")
```

### Anonymous Functions: Lambda Expressions
Small anonymous functions can be created using the `lambda` keyword:
```python
square = lambda x: x * x
```
Lambda expressions are syntactically restricted to a single expression and cannot contain complex statements or annotations.
