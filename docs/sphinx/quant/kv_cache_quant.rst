========================
KV Cache Quantization
========================

Overview
--------
Key-Value (KV) cache quantization is an important aspect of efficient large language model (LLM) inference. The importance of KV cache quantization lies in its potential to reduce memory consumption and improve runtime performance, especially for larger sequence lengths and batch sizes. We use the same quantization method as IQ Weight quantization, and we employ per-channel configuration for the KV-cache.

Config and Usage
----------------
The KV cache quantization feature is controlled by the ``kv_cache_mode`` function in the :doc:`../llm/runtime_config`:

``kv_cache_mode(cache_mode: AsCacheMode)``: Sets the cache mode for the key-value cache. The `AsCacheMode` enum provides three options: `AsCacheDefault`, `AsCacheQuantI8`, and `AsCacheQuantU4`.

  1. **AsCacheDefault**: Keeps the same data format as the model's inference, usually meaning a BF16/FP16 stored KV-Cache, this is default setting.
  2. **AsCacheQuantI8**: Quantizes the KV-cache into int8 format, reducing the KV-cache memory footprint by half compared to BF16.
  3. **AsCacheQuantU4**: Quantizes the KV-cache into uint8 format, reducing the KV-cache memory footprint by a quarter compared to BF16.


Example
-------

You can modify one line to enable this feature in :doc:`../get_started/quick_start_api_py_en` :

.. code-block:: python

  # insert this code in runtime cfg builder part.
  runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantI8) # for int8
  # runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantU4) # for int4

