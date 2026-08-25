# Data Structures and Algorithms Reference Notes

## Binary Search
Binary search runs in O(log n) time on a sorted array by repeatedly halving
the search interval: compare the target to the middle element, then recurse
into the left or right half. It requires the input to already be sorted.

## Sorting Complexity
Comparison-based sorts (merge sort, quicksort, heapsort) cannot beat O(n log n)
in the worst case. Quicksort's average case is O(n log n) but its worst case is
O(n^2) when the pivot choice repeatedly produces unbalanced partitions, for
example on an already-sorted array with a naive first-element pivot.

## Amortized Analysis
A dynamic array that doubles its capacity when full has O(1) amortized
insertion time, even though any individual doubling operation costs O(n). The
potential method assigns a "potential" to the data structure and charges each
operation its actual cost plus the change in potential, spreading the
occasional expensive resize across many cheap insertions.

## Hash Tables
A hash table offers average O(1) insertion, lookup, and deletion by mapping
keys to array indices via a hash function. Collisions are handled by chaining
(a linked list per bucket) or open addressing (probing for the next free
slot). Worst-case lookup degrades to O(n) if many keys collide.

## LRU Cache Design
A least-recently-used cache supporting O(1) get and put typically combines a
hash map (key to node) with a doubly linked list that tracks recency order,
so the least-recently-used entry can be evicted from the list's tail in O(1).
