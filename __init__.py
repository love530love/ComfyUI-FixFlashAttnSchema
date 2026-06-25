"""
fix_flash_attn_schema
----------------------
绕过 ComfyUI comfy/ldm/modules/attention.py 中
flash_attn_wrapper 的 torch.library.custom_op schema 注册 bug。

现象:
    Flash Attention failed, using default SDPA:
    schema_.has_value() INTERNAL ASSERT FAILED at "...OperatorEntry.h":84
    Tried to access the schema for  which doesn't have a schema registered yet

根因:
    comfy/ldm/modules/attention.py 用
    @torch.library.custom_op("flash_attention::flash_attn", ...)
    包了一层 flash_attn_func，这个自定义算子的 schema
    注册在某些情况下没完成，dispatcher 查 schema 时触发断言失败，
    自动回退到 SDPA（结果仍正确，但没有真正用上 flash-attn 的加速）。

做法:
    在自定义节点加载阶段（晚于 comfy.ldm.modules.attention 的
    模块级 try/except 执行），把 flash_attn_wrapper 替换成一个
    不经过 torch.library.custom_op 的普通函数，直接调用同一个
    flash_attn_func 引用。

对应版本（务必核对你本机的行号/代码是否一致）:
    H:\\PythonProjects3\\Win_ComfyUI\\comfy\\ldm\\modules\\attention.py
    第 680~697 行附近

确认方式（PowerShell）:
    Select-String -Path "...\\attention.py" -Pattern "flash_attn_wrapper|custom_op|register_fake"
"""

import comfy.ldm.modules.attention as comfy_attn

PATCHED = False

if hasattr(comfy_attn, "flash_attn_wrapper") and hasattr(comfy_attn, "flash_attn_func"):
    def _flash_attn_wrapper_patched(q, k, v, dropout_p: float = 0.0, causal: bool = False):
        # 直接复用 comfy.ldm.modules.attention 里已经导入好的 flash_attn_func，
        # 不再经过 torch.library.custom_op 包装，从而绕开 schema 注册 bug。
        return comfy_attn.flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)

    comfy_attn.flash_attn_wrapper = _flash_attn_wrapper_patched
    PATCHED = True
    print("[fix_flash_attn_schema] OK: flash_attn_wrapper 已切换为直连模式，绕过 torch.library.custom_op schema bug")
else:
    missing = []
    if not hasattr(comfy_attn, "flash_attn_wrapper"):
        missing.append("flash_attn_wrapper")
    if not hasattr(comfy_attn, "flash_attn_func"):
        missing.append("flash_attn_func")
    print(
        "[fix_flash_attn_schema] 警告: 在 comfy.ldm.modules.attention 中未找到 "
        + ", ".join(missing)
        + "，补丁未生效。可能是 ComfyUI 版本更新后这部分代码改了名字/位置，"
          "需要重新用 Select-String 核对 attention.py 的内容。"
    )

# 该模块不注册任何节点，只是在加载时打补丁
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
