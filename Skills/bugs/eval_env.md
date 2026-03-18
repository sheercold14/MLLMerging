# VLMEvalKit 评测环境 Bug 记录

## Bug 1: nvidia-cusparse-cu12 版本不兼容

**错误信息**:
```
ImportError: libcusparse.so.12: undefined symbol: __nvJitLinkCreate_12_8, version libnvJitLink.so.12
```

**原因**:
torch 2.10.0+cu128 依赖 nvidia-cusparse-cu12==12.5.8.93，而 cusparse 12.5 需要 nvjitlink 12.8。但系统 CUDA Driver 只支持到 12.4，nvjitlink 12.8 中的某些符号在运行时找不到。

**解决方案**:
```bash
pip install nvidia-cusparse-cu12==12.3.1.170 --no-deps
```

**注意**: 每次 `pip install` 其他包（如 torchvision, accelerate）可能把 cusparse 升回 12.5，需要重新降级。

---

## Bug 2: transformers 5.x 不兼容 VLMEvalKit

**错误信息**:
```
ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'
```

**原因**: transformers 5.x 移除了 `AutoModelForVision2Seq`，VLMEvalKit 中 `idefics.py` 还在引用。

**解决方案**:
```bash
pip install 'transformers>=4.45,<5.0' --no-deps
```

---

## Bug 3: huggingface-hub 1.x 不兼容 transformers 4.x

**错误信息**:
```
ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this module, but found huggingface-hub==1.7.1
```

**解决方案**:
```bash
pip install 'huggingface-hub>=0.34,<1.0'
```

---

## Bug 4: GPU 数量不匹配

**错误信息**:
```
torch.AcceleratorError: CUDA error: invalid device ordinal
```

**原因**: `eval.sh` 默认 `--nproc-per-node=8`，但实际只有 4 张 GPU（或 CUDA_VISIBLE_DEVICES 限制了可见 GPU）。

**解决方案**: 根据实际可用 GPU 数量调整:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2
torchrun --nproc-per-node=3 run.py ...
```

---

## Bug 5: 多次运行 pkl 文件冲突

**错误信息**:
```
[Errno 2] No such file or directory: './outputs/Model/T.../13_Dataset.pkl'
```

**原因**: 先用 4 卡跑了部分结果（生成 `04_`, `14_`, `24_`, `34_` 前缀的 pkl），再用 3 卡跑（生成 `03_`, `13_`, `23_` 前缀），合并时找不到对应的分片文件。

**解决方案**: 换 GPU 数量前先清理旧结果:
```bash
rm -rf ./outputs/Model_Name/
```

---

## Bug 6: OOM (CUDA out of memory)

**错误信息**:
```
CUDA out of memory. Tried to allocate 1.47 GiB. GPU 0 has a total capacity of 23.64 GiB
```

**场景**: Qwen2-VL-7B 在 MathVision_MINI 和 TextVQA_VAL 的某些高分辨率图片上 OOM。

**缓解方案**:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun ...
```

也可在 config.py 注册模型时降低 `max_pixels`:
```python
partial(Qwen2VLChat, model_path='...', min_pixels=256*28*28, max_pixels=512*28*28)
```

---

## Bug 7: VizWiz 评测阶段失败

**现象**: 推理完成但 evaluate 报错（空错误信息）。

**可能原因**:
- 我们用的 val split，官方用 test split，answer 格式可能不同
- 待排查

---

## Bug 8: MathVista/MathVision 评测需要 OpenAI API

**错误信息**:
```
Illegal openai_key. Please set the environment variable OPENAI_API_KEY to your openai key.
```

**原因**: 这两个数据集需要 GPT 作为 judge 来评测 free-form 答案。

**解决方案**: 设置 API key:
```bash
# 在 VLMEvalKit/.env 文件中添加:
OPENAI_API_KEY=sk-xxx
# 或直接设环境变量:
export OPENAI_API_KEY=sk-xxx
```

也可通过 `--judge` 参数指定其他 judge 模型。
