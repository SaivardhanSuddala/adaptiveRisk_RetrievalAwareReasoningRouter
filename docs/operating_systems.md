# Operating Systems Reference Notes

## Process Scheduling
Round-robin scheduling assigns each process a fixed time quantum and cycles
through the ready queue, giving reasonable fairness at the cost of context-
switch overhead when the quantum is small. Shortest-Job-First minimizes
average waiting time but can starve long jobs. Priority scheduling can also
starve low-priority processes unless it uses aging to gradually raise their
priority.

## Deadlock Conditions
A deadlock requires four conditions simultaneously: mutual exclusion, hold-and-
wait, no preemption, and circular wait. Breaking any one of the four prevents
deadlock. A common concurrency bug is two threads acquiring the same two locks
in opposite order, creating a circular wait between them.

## Virtual Memory and Paging
Virtual memory maps a process's logical address space to physical frames via a
page table, allowing a process to use more memory than is physically present
and isolating processes from each other. A page fault occurs when a referenced
page is not currently in physical memory and must be loaded from disk.

## Threads vs. Processes
Threads within the same process share the same address space and file
descriptors, making context switches between threads cheaper than between
processes, but also making threads vulnerable to each other's memory errors. A
mutex is used to ensure only one thread accesses a shared resource at a time.

## System Calls
A system call is the mechanism by which a user-space program requests a
service from the kernel, such as opening a file or allocating memory. It
involves a controlled transition from user mode to kernel mode.
