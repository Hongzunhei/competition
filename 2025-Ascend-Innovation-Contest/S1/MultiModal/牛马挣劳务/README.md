# 队伍：牛马挣劳务

## 一、Qwen2-VL / Janus Pro 多模态推理优化说明

本仓库主要针对 **MindNLP + MindSpore** 下的多模态大模型（以 **Qwen2-VL / Janus Pro** 为主）进行推理侧优化，目标是：

- 在 **不改变模型行为与精度** 的前提下，  
- **降低端到端推理延迟**、**提升吞吐**，  
- 并尽量 **降低显存占用、清理冗余实现**，为后续维护和扩展打基础。

所有改动集中在：

- `mindnlp/transformers/models/qwen2_vl/modeling_qwen2_vl.py`
- `mindnlp/transformers/models/llama/modeling_llama.py`
- `llm/inference/janus_pro/janus/models/*` 及相关预处理逻辑

等文件中。

---

## 二、整体优化思路概览

主要针对 **Prefill 时延、Decode 时延、峰值显存占用** 三方面的优化展开：

1. **Prefill 时延**

   Prefill 时延主要是输入到产生第一个 token 的时间，可以看作「预处理 + 图像前向 + 文本编码」整体耗时。
   
    deepseek-ai/Janus-Pro-7B：
   - 在deepseek-ai/Janus-Pro-7B当中，Prefill时延瓶颈主要来源于预处理部分。详细参考llm/inference/janus_pro/janus/models/processing_vlm.py。
     本人采用了一种一种笨笨的办法，通过[JIT装饰器](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.jit.html)加速操作。JIT装饰器主要作用是可以针对相同shape下的多个算子一起下发，节约时间。一定要是相同shape，否则将会重新编译，编译时长可能大于JIT能节约的时间。通过填充无用数据构造成相同shape后，再将无用数据剔除，还原原本shape的操作，可以将编译时间丢到warm-up里，后面正式进行推理时则节约了时间。具体收益不记得了，但是收益较大。

     Qwen/Qwen2-VL-2B-Instruct：
   - 在Qwen/Qwen2-VL-2B-Instruct当中，Prefill时延瓶颈主要来源于视觉部分。详细参考mindnlp/transformers/models/qwen2_vl/modeling_qwen2_vl.py。
     通过Profiler工具可以观察到，有一个Conv3D的算子相当的耗时，将nn.Conv3d替换为mindspore.mint.nn.Conv3d能大大减少耗时，收益较大。
   - 另外一部分瓶颈来源于**VisionAttention**的计算。通过**flash_attention_score**融合算子可以大大减少Prefill时延，收益较大。
   ```python
   output = mops.flash_attention_score(query, key, value, head_num=head_num, input_layout='BSND',
                             real_shift=None, padding_mask=None, attn_mask=attn_mask,
                             scalar_value=scale_factor, keep_prob=1 - dropout_p, pre_tokens=2147483647,
                             next_tokens=2147483647, inner_precise=0,
                             drop_mask=None, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None,
                             sparse_mode=sparse_mode)
   ```
     - 对于VisionAttention的attn_mask加一个判断，如果要构造一个没有掩码的mask，则直接将mask设为None。收益较小。

2. **Decode 时延**

   Decode 时延主要指生成阶段「每步新增一个 token」的平均耗时，瓶颈通常在注意力计算、KV Cache 管理及 mask 构造。

   deepseek-ai/Janus-Pro-7B：
   - 在deepseek-ai/Janus-Pro-7B当中，Decode时延瓶颈主要来源于apply_rotary_pos_emb。但可惜在这里用rotary_position_embedding融合算子会导致mismatch，所以deepseek-ai/Janus-Pro-7B的decode优化较少。详细参考mindnlp/transformers/models/llama/modeling_llama.py。
   - 针对apply_rotary_pos_emb,将q和k一起进行rotary_pos_emb有一丢丢收益，不大。
    ```python
    qk = mindspore.mint.cat((q, k), dim=0)
    qk_embed = mindspore.mint.mul(qk, cos) + mindspore.mint.mul(rotate_half(qk), sin)
    q_embed, k_embed = mindspore.mint.split(qk_embed, 1, 0)
    return q_embed, k_embed
    ```
   - 将repeat_kv通过repeat_interleave融合算子替代。
   - 有很多可以使用mint替换ops，有一些有收益。很难估计，但收益都比较小，积少成多。

   Qwen/Qwen2-VL-2B-Instruct：
   - 在Qwen/Qwen2-VL-2B-Instruct当中，Decode时延瓶颈主要来源于Qwen2RMSNorm、apply_rotary_pos_emb_vision等部分。详细参考mindnlp/transformers/models/qwen2_vl/modeling_qwen2_vl.py。
   - 用融合算子rotary_position_embedding替代原有rotary_pos_emb逻辑。
   - 将repeat_kv通过repeat_interleave融合算子替代。
   - 将apply_mrope_mask操作从Qwen2VLAttention.forward丢到Qwen2VLModel.forward进行，避免做N层重复运算。
   - 在Qwen2VLModel里增加对Prefill和decode阶段的判断，避免在decode阶段进行_prepare_4d_causal_attention_mask_with_cache_position操作。
   - 有很多可以使用mint替换ops，有一些有收益。很难估计，但收益都比较小，积少成多。
3. **峰值显存占用**

   峰值显存主要由 KV Cache、注意力中间结果以及多模态 embedding 组成。
   为降低峰值显存，本次优化仅在Qwen/Qwen2-VL-2B-Instruct优化成功：
   - **将VisionAttention当中的softmax去掉提升精度**:通过**flash_attention_score**融合算子也可以。
## 三、总结
改动较多，主要列举了收益较大的改动，其余改动见文件所示。未一一列举请多多包涵。

