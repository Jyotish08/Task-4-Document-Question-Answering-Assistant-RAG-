---
title: "Python Errors and Exception Handling"
source_url: "https://docs.python.org/3/tutorial/errors.html"
retrieval_date: "2026-10-17"
license: "Python Software Foundation License (PSFL)"
category: "Programming / Python Fundamentals"
---

# Python Errors and Exception Handling

## 1. Syntax Errors vs Exceptions
Python distinguishes between two primary categories of errors:
1. **Syntax Errors (Parsing Errors)**: Detected by the Python parser before code execution begins when code violates grammatical rules. Example: missing a colon after an `if` statement raises `SyntaxError`.
2. **Exceptions**: Errors detected during execution. Even if a statement is syntactically correct, it may trigger an error when evaluated. Common built-in exceptions include `ZeroDivisionError`, `NameError`, `TypeError`, and `IndexError`.

## 2. Handling Exceptions (`try ... except`)
The `try ... except` block intercepts and handles exceptions gracefully without terminating the program:
```python
try:
    result = 10 / divisor
except ZeroDivisionError as err:
    print(f"Handling division by zero: {err}")
    result = 0
```

### Catching Multiple Exceptions
A single `try` clause may have multiple `except` handlers to address different exception types:
```python
try:
    f = open("data.txt")
    val = int(f.readline().strip())
except FileNotFoundError:
    print("File not found")
except ValueError:
    print("Could not parse integer from line")
except (IOError, RuntimeError) as e:
    print(f"System error: {e}")
```
Multiple exceptions can also be grouped in a parenthesized tuple, as shown in `(IOError, RuntimeError)`.

## 3. The `else` and `finally` Clauses
The `try` statement has two optional supplemental clauses:
- **`else` clause**: Must follow all `except` clauses. The code inside `else` executes only if the code in the `try` block did NOT raise an exception. It is useful for code that should run only if the protected operation succeeded:
```python
try:
    f = open("results.csv", "r")
except OSError:
    print("Cannot open file")
else:
    print(f"File has {len(f.readlines())} lines")
    f.close()
```
- **`finally` clause**: Placed at the very end of the construct. The `finally` block executes under all circumstances, whether an exception occurred, was handled, was re-raised, or if a `return`, `break`, or `continue` statement was encountered. It is intended for releasing external resources (closing database connections or file descriptors).

## 4. Raising Exceptions (`raise`)
The `raise` statement allows a programmer to force a specified exception to occur:
```python
if amount < 0:
    raise ValueError("Amount cannot be negative")
```
Calling `raise` without arguments inside an `except` block re-raises the active exception.

## 5. User-Defined Custom Exceptions
Programs may name their own exceptions by creating a new class that derives directly or indirectly from the built-in `Exception` class:
```python
class InsufficientFundsError(Exception):
    """Raised when an account withdrawal exceeds available balance."""
    def __init__(self, balance, amount):
        super().__init__(f"Attempted to withdraw {amount} with only {balance} available")
        self.balance = balance
        self.amount = amount
```
Custom exceptions should always inherit from `Exception`, not `BaseException`, to ensure they can be caught by standard `except Exception:` blocks without intercepting system-exiting exceptions like `KeyboardInterrupt` or `SystemExit`.
