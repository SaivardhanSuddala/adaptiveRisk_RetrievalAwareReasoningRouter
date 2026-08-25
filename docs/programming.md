# Programming Reference Notes

## Python List Methods
`append(x)` adds a single element to the end of a list. `extend(iterable)` adds
each element of an iterable to the end of a list. There is no `push` method on
Python lists; code that calls `list.push()` will raise an `AttributeError`.
`pop()` removes and returns the last element, or the element at a given index.

## Recursion and Base Cases
A recursive function must have a base case that does not call itself, otherwise
it recurses until the call stack is exhausted and a `RecursionError` (or stack
overflow) occurs. A common bug is a base case that returns correctly but a
recursive case that fails to combine the base case's result with the current
call's contribution, silently dropping part of the computation.

## References vs. Pointers
In C++, a reference (`int&`) is an alias for an existing variable; modifying the
reference modifies the original variable directly, and no new storage is
allocated. This differs from a pointer, which stores an address and must be
dereferenced explicitly.

## Object-Oriented Method Resolution
When a subclass overrides a method and calls `super().method()` inside the
override, the parent class's implementation runs first, and its return value
is available to be combined with the subclass's own logic before the subclass
method returns.

## Common Complexity Classes in Practice
List indexing and dictionary key lookup are O(1) on average in Python.
Searching an unsorted list is O(n). Sorting with Python's built-in `sorted()`
(Timsort) is O(n log n) in the average and worst case.
