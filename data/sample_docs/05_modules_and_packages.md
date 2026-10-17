---
title: "Python Modules, Packages, and Environments"
source_url: "https://docs.python.org/3/tutorial/modules.html"
retrieval_date: "2026-10-17"
license: "Python Software Foundation License (PSFL)"
category: "Programming / Python Fundamentals"
---

# Python Modules, Packages, and Environments

## 1. Modules
A module is a file containing Python definitions and statements. The file name is the module name with the suffix `.py` appended. Within a module, the global variable `__name__` holds the module's name as a string.

### Importing Modules
Modules are loaded using the `import` statement:
- `import math`: Imports the entire module; functions are accessed via `math.sqrt(16)`.
- `from math import pi, sqrt`: Imports specific names directly into the local namespace.
- `import numpy as np`: Imports a module under an alias.

When a module is imported for the first time, its top-level code is executed from top to bottom. Subsequent imports in the same process reuse the cached module object stored in `sys.modules`.

### Executing Modules as Scripts
A module can determine whether it is being run as a standalone script or imported by inspecting `__name__`:
```python
if __name__ == "__main__":
    # Code executed only when script is run directly
    print("Running as main program")
```
When a file is run via `python myscript.py`, Python sets `__name__` to `"__main__"`.

## 2. Packages
Packages are a way of structuring Python's module namespace using dotted module names (e.g., `package.submodule`). A directory is treated as a package if it contains a file named `__init__.py`. In Python 3.3+, namespace packages allow packages without `__init__.py`, but regular packages require `__init__.py` to initialize package-level state or define `__all__`.

The special variable `__all__` in `__init__.py` is a list of string module names that specifies which symbols are exported when `from package import *` is executed.

## 3. The Module Search Path (`sys.path`)
When a module named `spam` is imported, the interpreter searches for a file named `spam.py` in the directories listed in `sys.path`. `sys.path` is initialized from:
1. The directory containing the input script (or the current working directory).
2. The `PYTHONPATH` environment variable (a list of directory names).
3. The installation-dependent default directory list (standard library and site-packages).

## 4. Standard Library Modules
Python includes a comprehensive "batteries included" standard library:
- `math`: Mathematical functions (`math.ceil`, `math.floor`, `math.log`, `math.sin`).
- `sys`: Interpreter-level variables and functions (`sys.argv`, `sys.exit`, `sys.version`, `sys.path`).
- `os`: Operating system interaction (`os.environ`, `os.path.join`, `os.mkdir`).
- `json`: JSON encoding and decoding (`json.dumps`, `json.loads`).
- `pathlib`: Object-oriented filesystem paths (`pathlib.Path`).

## 5. Virtual Environments (`venv`)
Python applications often require different versions of packages. The standard `venv` module creates lightweight, isolated virtual environments with their own site-packages directory and Python interpreter:
```bash
python -m venv myenv
```
Activating a virtual environment updates the shell's `PATH` to prioritize the environment's Python binary and site-packages, preventing conflicts between project dependencies.
