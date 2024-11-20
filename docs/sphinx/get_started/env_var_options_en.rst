Environment Variable Usage
--------------------------


This section describes the definition of the Allspark environment variables and their function.

Memory Mangament
================

.. list-table:: Environment Var: Memory
   :widths: 10 15 5 5 25
   :header-rows: 1

   * - EnvVar Name
     - Describe
     - Type
     - Default
     - Options

   * - ``BFC_ALLOCATOR``
     - Use BFC Allocator or raw cudaMalloc API

       for management of CUDA Device memory.

     - bool
     - ``ON``
     - ON - Enable BFC Allocator

       OFF - Disable BFC Allocator

   * - ``BFC_MEM_RATIO``
     - The max ratio of device memory that will be managemented by BFC Allocator.
     - float
     - ``0.975``
     - float value between (0.0,1.0]

   * - ``BFC_LEFTOVER_MB``
     - The amount of GPU memory that cannot be used by the BFC allocator,

       typically the sum of GPU memory occupied by PyTorch, CUDA driver,

       default context, etc.
     - int
     - ``350``
     - Formula for the actual GPU memory allocated

       by the BFC Allocator on each GPU:

       ``(Total Physical Memory - BFC_LEFTOVER_MB) * BFC_MEM_RATIO``

Logging
=======


.. list-table:: Environment Var: Logging
   :widths: 10 15 5 5 25
   :header-rows: 1

   * - EnvVar Name
     - Describe
     - Type
     - Default
     - Options

   * - ``ALLSPARK_TIME_LOG``
     - Whether logging the generation and context step detailed time in different phase.
     - int
     - ``0``
     - ``0`` - not print;  ``1`` - print log.

   * - ``ALLSPARK_DUMP_OUTPUT_TOKEN``
     - Whether print output token in log.
     - int
     - ``0``
     - ``0`` - not print;  ``1`` - print log.

   * - ``HIE_LOG_SATAUS_INTERVAL``
     - The threshold control for printing statistical log when text generation.
     - int
     - ``5``
     - In second, should be greater than 0.


Engine Behavior
===============

.. list-table:: Environment Var: Engine Behavior
   :widths: 10 15 5 5 25
   :header-rows: 1

   * - EnvVar Name
     - Describe
     - Type
     - Default
     - Options

   * - ``ALLSPARK_USE_TORCH_SAMPLE``
     - Use the same sampler as vllm and PyTorch.

       The generation speed may decrease by 5%-10%.
     - int
     - ``1``
     - ``0`` - use torch sampler

       ``1`` - use allspark native sampler,

       which provides the same distribution, 
       
       but not exactly the same value.

   * - ``AS_WEIGHT_LOAD_FROM_MMAP``
     - Enable loading weight from mmap, or file.
     - string
     - ``off``
     - ``on`` - load weight by ``mmap(3)`` API, 
     
       will be faster on servers with SSD-like storage.

       ``off`` - use fopen API to read API.

   * - ``AS_FLASH_THRESH``
     - Threshold for enable Flash Attention do context attention calculation.

       Flash Attention will be used if context length is greater than this threshold.
     - int
     - ``1024``
     - int value between (0, int64_max).

   * - ``ALLSPARK_DISABLE_WARMUP``
     - Disable warm up step when model is start up.
     - int
     - ``0``
     - ``1``: disable warm up

       ``0``: not disable warm up
