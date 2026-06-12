# LLaVA Inference Patches and Problem Solutions

This document records the runtime patches and workarounds that enable **LLaVA-OneVision-1.5-8B-Instruct** to run locally, along with the errors each patch addresses. All patches are applied **in notebook/script code** — no edits to the model checkpoint files are required.

**Primary references in this repo:**

| File | Role |
|------|------|
| `LLaVA-RLHF/test_ov.ipynb` | Single-image OneVision inference test |
| `mywork/test_on_images/llava_ov_multi.ipynb` | 16-probe cross-image benchmark (OneVision) |
| `mywork/test_on_images/SFTvsRLHF_multi.ipynb` | Same benchmark on LLaVA-RLHF SFT+ / RLHF (no transformers 5.x patches needed) |

**Model path:** `LLaVA-RLHF/models/LLaVA-OneVision-1.5-8B-Instruct`  
**Checkpoint build version:** `transformers==4.53.0` (recorded in `config.json` and `generation_config.json`)

---

## Summary

| Problem | Symptom | Solution |
|---------|---------|----------|
| Wrong `transformers` version | Load errors or garbage outputs | Pin `transformers==4.53.0` |
| Missing `SlidingWindowCache` (transformers 5.x) | `ImportError` / `AttributeError` on model import | Alias to `StaticCache` at runtime |
| Missing `layer_type_validation` (transformers 5.x) | Config validation crash | Inject no-op validator |
| Missing `PreTrainedModel.set_submodule` (transformers 5.x) | Weight-loading crash | Inject shim method |
| Missing `ROPE_INIT_FUNCTIONS["default"]` (transformers 5.x) | RoPE init `KeyError` | Register default RoPE function |
| Missing `pad_token_id` in config (transformers 5.x) | Generation fails or behaves incorrectly | Read from `generation_config.json` |
| `torch_dtype` renamed to `dtype` (transformers 5.x) | Unexpected kwarg / wrong dtype | Version-gated kwarg name |
| GPU OOM on full-precision load | `OutOfMemoryError` or weights on `meta` device | 4-bit NF4 fallback via BitsAndBytes |
| Jupyter + `argparse` | `unrecognized arguments: --f=...` | Use plain variables; no CLI parsing in notebooks |
| Partial CPU offload | Model loads but outputs nonsense | Require full GPU load (bf16/fp16 or 4-bit on GPU) |

---

## 1. Required environment: `transformers==4.53.0`

### Problem

The OneVision checkpoint was built and exported with **transformers 4.53.0**. Running with **transformers 5.0.0** causes a chain of API mismatches. Even after applying all runtime compatibility patches below, **inference on transformers 5.x still produces invalid logits** (garbage text). Loading may succeed, but outputs are not usable.

### Solution

Use a dedicated conda env (e.g. `llava-ov`) with the pinned version:

```bash
pip install 'transformers==4.53.0'
```

The inference entry points **hard-block** transformers ≥ 5 for inference and raise:

```
RuntimeError: transformers>=5.0 is not supported for inference.
Run: pip install 'transformers==4.53.0'
```

On **transformers 4.53.0**, no transformers 5.x compatibility patches are needed; the model loads and generates correctly.

### Verified hardware

- **GPU:** NVIDIA RTX 4000 Ada (20 GB VRAM)
- **Precision:** bf16 (~17 GB) when GPU is free; 4-bit NF4 fallback if bf16 OOMs

---

## 2. Transformers 5.x compatibility patches (load-only)

These patches live in `apply_transformers_compat_patches()` and are applied **before** `AutoModelForCausalLM.from_pretrained()`. They monkey-patch missing symbols in the installed `transformers` package so that custom model code (`modeling_llavaonevision1_5.py`) can be imported.

> **Important:** These patches only unblock **loading**. They do **not** make transformers 5.x safe for inference on this checkpoint.

### 2.1 `SlidingWindowCache` → `StaticCache`

**Problem:** `modeling_llavaonevision1_5.py` imports `SlidingWindowCache` from `transformers.cache_utils`. This class exists in transformers 4.53 but is absent in transformers 5.x, causing:

```
ImportError: cannot import name 'SlidingWindowCache' from 'transformers.cache_utils'
```

**Patch:**

```python
import transformers.cache_utils as cache_utils

if not hasattr(cache_utils, "SlidingWindowCache"):
    cache_utils.SlidingWindowCache = cache_utils.StaticCache
```

**Rationale:** The model checks `isinstance(past_key_values, SlidingWindowCache)` for sliding-window attention. Aliasing to `StaticCache` satisfies the import and type checks enough to construct the model. Correct sliding-window cache behavior is not guaranteed under transformers 5.x — another reason to use 4.53.0.

---

### 2.2 `layer_type_validation` stub

**Problem:** transformers 5.x expects `configuration_utils.layer_type_validation` when validating layer-type lists in composite configs. Without it, config loading can fail when the OneVision config is parsed.

**Patch:**

```python
import transformers.configuration_utils as configuration_utils

if not hasattr(configuration_utils, "layer_type_validation"):

    def _layer_type_validation(layer_types):
        return layer_types

    configuration_utils.layer_type_validation = _layer_type_validation
```

**Rationale:** A pass-through validator accepts the checkpoint config as-is without enforcing new 5.x constraints.

---

### 2.3 `PreTrainedModel.set_submodule` shim

**Problem:** Weight loading in transformers 5.x calls `model.set_submodule(path, module)` to place submodules by dotted name. This method does not exist on older `PreTrainedModel` bases used with 4.x-style loading paths.

**Patch:**

```python
from transformers.modeling_utils import PreTrainedModel

if not hasattr(PreTrainedModel, "set_submodule"):

    def _set_submodule(self, target: str, module: nn.Module) -> None:
        atoms = target.split(".")
        if len(atoms) == 1:
            setattr(self, atoms[0], module)
            return
        parent = self.get_submodule(".".join(atoms[:-1]))
        setattr(parent, atoms[-1], module)

    PreTrainedModel.set_submodule = _set_submodule
```

**Rationale:** Mirrors the transformers 5.x API so `from_pretrained` can assign vision tower and language-model submodules by name.

---

### 2.4 `ROPE_INIT_FUNCTIONS["default"]` backfill

**Problem:** The OneVision text backbone initializes RoPE via `ROPE_INIT_FUNCTIONS[self.rope_type]`. Under transformers 5.x the `"default"` entry may be missing, causing:

```
KeyError: 'default'
```

**Patch:**

```python
import transformers.modeling_rope_utils as rope_utils

if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:

    def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2, dtype=torch.int64)
                .to(device=device, dtype=torch.float)
                / dim
            )
        )
        return inv_freq, 1.0

    rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
```

**Rationale:** Standard inverse-frequency RoPE computation matching the pre-5.x `"default"` initializer.

---

## 3. Config and model-loading patches

### 3.1 `pad_token_id` from `generation_config.json`

**Problem:** Under transformers 5.x, `config.json` → `text_config` may omit `pad_token_id`. Without it, `model.generate()` can fail or pad incorrectly because the embedding layer cannot resolve the padding index.

**Patch** (`load_config`):

```python
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

if not hasattr(config.text_config, "pad_token_id") or config.text_config.pad_token_id is None:
    with open(os.path.join(model_path, "generation_config.json")) as f:
        config.text_config.pad_token_id = json.load(f)["pad_token_id"]
```

**Rationale:** `generation_config.json` in the checkpoint still carries the correct `pad_token_id` (151643 for this model). Copying it into `text_config` before load fixes generation.

---

### 3.2 `torch_dtype` vs `dtype` (transformers 5.x rename)

**Problem:** transformers 5.x renamed the `from_pretrained` argument `torch_dtype` → `dtype`. Passing the wrong keyword silently fails or warns depending on version.

**Patch** (`build_load_kwargs`):

```python
if transformers_major >= 5:
    kwargs["dtype"] = compute_dtype      # bf16 or fp16
else:
    kwargs["torch_dtype"] = compute_dtype
```

**Rationale:** Use bf16 when `torch.cuda.is_bf16_supported()` else fp16 — the dtype the checkpoint expects for GPU inference.

---

### 3.3 Meta-device weight detection

**Problem:** When VRAM is insufficient, `device_map="auto"` may leave some weights on the **`meta`** device (placeholders never materialized on GPU). The model appears loaded but produces nonsense.

**Patch:**

```python
def count_meta_parameters(model) -> int:
    return sum(1 for _, p in model.named_parameters() if p.device.type == "meta")

# After from_pretrained:
if count_meta_parameters(model):
    raise torch.cuda.OutOfMemoryError("weights left on meta device")
```

**Rationale:** Treat partial meta offload as OOM and trigger the 4-bit retry rather than running broken inference.

---

### 3.4 4-bit NF4 fallback on OOM

**Problem:** Full bf16 OneVision-8B needs ~17 GB. On a 20 GB card with other processes resident, full-precision load raises `OutOfMemoryError`.

**Patch:**

```python
try:
    model = AutoModelForCausalLM.from_pretrained(**build_load_kwargs(..., use_quantization=False))
    if count_meta_parameters(model):
        raise torch.cuda.OutOfMemoryError("weights left on meta device")
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(**build_load_kwargs(..., use_quantization=True))
```

With `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)`.

**Rationale:** 4-bit quantization fits comfortably in 20 GB while keeping all weights on GPU. Avoid partial CPU offload — that path was observed to produce bad outputs even when load appeared successful.

---

## 4. Inference workflow patches

### 4.1 No `argparse` in Jupyter notebooks

**Problem:** Jupyter injects kernel arguments (e.g. `--f=/path/to/kernel.json`). A notebook cell that calls `argparse.ArgumentParser().parse_args()` fails with:

```
error: unrecognized arguments: --f=...
```

**Solution:** Use plain Python variables at the top of a cell (`IMAGE_PATH`, `PROMPT`, `MODEL_PATH`) instead of CLI parsing. See `test_ov.ipynb`.

---

### 4.2 Qwen-VL chat template + `qwen_vl_utils`

**Problem:** OneVision is not a legacy LLaVA `USER: <image>\n... ASSISTANT:` model. It expects **Qwen-VL-style** multimodal messages and vision preprocessing.

**Solution:**

```python
from qwen_vl_utils import process_vision_info

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question},
    ],
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, ...)
```

Greedy decoding (`do_sample=False`) is used for reproducible benchmark answers.

---

### 4.3 Decode only newly generated tokens

**Problem:** Decoding the full `generated_ids` tensor includes the input prompt tokens, polluting the answer string.

**Patch:**

```python
trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
```

---

### 4.4 Variable `max_new_tokens` by probe type

**Problem:** Open-ended probes (`"Describe this image..."`) need more tokens than yes/no or count questions.

**Patch:**

```python
max_new_tokens = 128 if question.startswith("Describe") else 32
```

---

## 5. LLaVA-RLHF 7B (SFT+ / RLHF) — separate stack

The `SFTvsRLHF_multi.ipynb` notebook evaluates **LLaVA-RLHF-7B** via the original LLaVA codebase (`llava.model.builder.load_pretrained_model`), not HuggingFace OneVision. It does **not** need the transformers 5.x patches above.

| Setting | Value | Notes |
|---------|-------|-------|
| Loader | `load_pretrained_model(..., load_8bit=True)` | 8-bit quantization for 7B on 20 GB GPU |
| RLHF | `PeftModel.from_pretrained(model, LORA_PATH)` | LoRA adapter on top of SFT+ backbone |
| Prompt format | `USER: {DEFAULT_IMAGE_TOKEN}\n{question}\nASSISTANT:` | Classic LLaVA v1.5 template |
| Image processing | `process_images` + `tokenizer_image_token` | LLaVA-specific utils |

Known benign warning: unused vision-tower weights in the SFT checkpoint (weights present in file but not in the initialized module graph). Inference still works.

---

## 6. What we explicitly do **not** patch

| Approach | Why it was rejected |
|----------|---------------------|
| Editing `modeling_llavaonevision1_5.py` in the checkpoint | User requirement: patches in inference code only, no model file edits |
| Running inference on transformers 5.x after load patches | Logits are garbage; only 4.53.0 produces correct answers |
| Partial CPU offload (`device_map` spreading to CPU) | Loads but outputs are unreliable |
| `argparse` in notebooks | Breaks under Jupyter kernel flags |

---

## 7. Quick start

We use **two separate Conda environments** — one per model stack. Environment specs are exported in:

| Env | YAML file | Used for |
|-----|-----------|----------|
| `llava_rlhf` | [`mywork/llava_rlhf.yml`](../llava_rlhf.yml) | LLaVA-RLHF 7B SFT+ / RLHF (`SFTvsRLHF_multi.ipynb`) |
| `llava-ov` | [`mywork/llava-ov.yml`](../llava-ov.yml) | LLaVA-OneVision-1.5-8B (`llava_ov_multi.ipynb`) |

### Create environments (first time)

```bash
cd mywork

# LLaVA-RLHF 7B stack
conda env create -f llava_rlhf.yml
# Key pins: Python 3.10, torch 2.0.1+cu118, transformers 4.31.0, llava, peft

# LLaVA-OneVision stack
conda env create -f llava-ov.yml
# Key pins: Python 3.10, torch 2.4.0+cu121, transformers 4.53.0, qwen-vl-utils
```

To recreate after updating a YAML file: `conda env update -f <file>.yml --prune`

### OneVision 16-probe benchmark (`llava-ov`)

```bash
conda activate llava-ov

cd mywork/test_on_images
jupyter notebook llava_ov_multi.ipynb
```

### SFT+ / RLHF 16-probe benchmark (`llava_rlhf`)

```bash
conda activate llava_rlhf

cd mywork/test_on_images
jupyter notebook SFTvsRLHF_multi.ipynb
```

### Merge results (either env is fine)

```bash
conda activate llava-ov   # or llava_rlhf

cd mywork/test_on_images
python compare_probe_results.py
```

---

## 8. Patch application order

For OneVision inference, apply steps in this order:

1. `apply_transformers_compat_patches()` — only relevant if transformers ≥ 5 is installed
2. **Version gate** — abort if major ≥ 5 (inference)
3. `load_config()` — fix `pad_token_id`
4. `build_load_kwargs()` — dtype + optional 4-bit config
5. `from_pretrained()` — with meta-device check and OOM → 4-bit retry
6. `AutoProcessor.from_pretrained()`
7. Build Qwen-VL messages → `generate()` → trim decode

All of the above are implemented inline in `llava_ov_multi.ipynb`.
