# Primitives

## All-Reduce

Neural networks today are often trained across multiple GPUs or even multiple systems, each with multiple GPUs. There are two main categories of techniques for doing this: synchronous and asynchronous. Synchronous techniques rely on keeping the parameters on all instances of the model synchronized, usually by making sure all instances of the model have the same copy of the gradients before taking an optimization step. The primitive usually used to perform this operation is called All-Reduce.

All-Reduce based on the number of ranks and the size of the data.
