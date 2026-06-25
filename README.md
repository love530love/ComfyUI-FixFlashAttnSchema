# ComfyUI-FixFlashAttnSchema

Bypass ComfyUI's `torch.library.custom_op` schema-registration bug that causes
Flash Attention to silently fall back to SDPA.

## The problem

If your ComfyUI console is being spammed with:

```
Flash Attention failed, using default SDPA: schema_.has_value() INTERNAL ASSERT FAILED
at "...OperatorEntry.h":84, please report a bug to PyTorch.
Tried to access the schema for  which doesn't have a schema registered yet
```

...your sampling still completes correctly, but Flash Attention is silently
falling back to SDPA on every step — you're not getting the speed/VRAM benefit
you think you are.

## Root cause

`comfy/ldm/modules/attention.py` wraps the third-party `flash_attn_func` with
`torch.library.custom_op("flash_attention::flash_attn", ...)` for FakeTensor /
`torch.compile` tracing support. In certain conditions this custom op's schema
isn't fully registered before the dispatcher looks it up, triggering the
`OperatorEntry.h:84` internal assert. This has been reproduced across multiple
PyTorch versions (2.6–2.11.dev) and on both Windows/CUDA and Linux/ROCm, so
it's neither a PyTorch version regression nor a Windows-wheel limitation — it's
specific to this wrapper.

See the references below for the community investigation this fix is based on.

## What this does

This node patches `comfy.ldm.modules.attention.flash_attn_wrapper` at load time
to call `flash_attn_func` directly, skipping the broken `custom_op` wrapper.
No core ComfyUI files are modified, so the patch survives ComfyUI updates.

It adds no new nodes — it's a load-time fix only. `NODE_CLASS_MAPPINGS` is
intentionally empty.

## Install

Via ComfyUI-Manager, or manually:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/love530love/ComfyUI-FixFlashAttnSchema.git
```

Restart ComfyUI. You should see in the startup log:

```
[fix_flash_attn_schema] OK: flash_attn_wrapper 已切换为直连模式，绕过 torch.library.custom_op schema bug
```

If you instead see a warning, your ComfyUI version's `attention.py` may have
changed the function name/signature — please open an issue with your ComfyUI
version and the relevant lines from `comfy/ldm/modules/attention.py`.

## Caveats

- This is a community workaround, not an upstream fix. The PyTorch issue is
  still open as of this writing.
- Removing the `custom_op` wrapper means this attention path loses FakeTensor
  shape inference / `torch.compile` graph tracing support. If your workflow
  relies on `torch.compile` tracing through this specific attention call,
  test carefully.
- Function name/signature may change in future ComfyUI versions; this patch
  uses defensive `hasattr` checks and will warn (not crash) if it can't find
  what it expects.

## References

- [pytorch/pytorch#172944](https://github.com/pytorch/pytorch/issues/172944)
- [Comfy-Org/ComfyUI Discussion #13024](https://github.com/Comfy-Org/ComfyUI/discussions/13024)
- [Dao-AILab/flash-attention#1929](https://github.com/Dao-AILab/flash-attention/issues/1929)
- [Dao-AILab/flash-attention#2132](https://github.com/Dao-AILab/flash-attention/issues/2132)
- [ClownsharkBatwing/RES4LYF#225](https://github.com/ClownsharkBatwing/RES4LYF/issues/225)

---

## 中文说明

### 问题现象

ComfyUI 跑图时控制台反复刷屏 `schema_.has_value() INTERNAL ASSERT FAILED`，出图结果正确，但 Flash Attention 每次都被悄悄回退到 SDPA，没吃到应有的加速/省显存效果。

### 根因

`comfy/ldm/modules/attention.py` 用 `torch.library.custom_op` 包了一层第三方 `flash_attn_func`，这个自定义算子的 schema 在某些情况下没注册完整，dispatcher 查询时触发断言失败。已在 PyTorch 2.6~2.11.dev 多个版本、Windows/CUDA 和 Linux/ROCm 多个平台上复现，跟 PyTorch 版本回归、跟 Windows wheel 阉割算子都没有必然关系，问题specific 在这层包装代码本身。详细排查过程见下方参考链接。

### 这个插件做了什么

在 ComfyUI 加载自定义节点阶段，把 `flash_attn_wrapper` 替换成不经过 `custom_op` 的普通函数，直连 `flash_attn_func`。不改任何 ComfyUI 核心文件，升级 ComfyUI 不会冲掉这个补丁。不提供任何新节点，纯粹是加载期修复。

### 安装

通过 ComfyUI-Manager 搜索安装，或手动 clone 到 `custom_nodes` 目录下，重启 ComfyUI。

### 注意事项

这是社区层面的 workaround，不是官方修复；PyTorch 官方 issue 截至目前仍是 open 状态。去掉 `custom_op` 包装会让这条路径失去 FakeTensor 推断 / `torch.compile` 图追踪支持，如果你的工作流依赖这点请额外测试。
