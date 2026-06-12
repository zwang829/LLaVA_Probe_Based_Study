# LLaVA Model Comparison Benchmarks (`mywork/`)

Local evaluation suite comparing **LLaVA-RLHF** (7B, SFT+ / RLHF LoRA) and **LLaVA-OneVision-1.5-8B-Instruct** on controlled vision-language probes. Two benchmark tracks are provided:

| Track | Folder | Probes | Question |
|-------|--------|--------|----------|
| **Natural scenes** | [`natural_scenes/`](natural_scenes/) | 16 | Do models hallucinate on plausible real-world images? |
| **Extreme OOD** | [`extreme_case/`](extreme_case/) | 7 | Do models invent meaning or break under zero-semantic / synthetic inputs? |

Both tracks share a **schema v1.0 JSON** output format and Python merge scripts that export comparison CSVs.

---

## Directory layout

```
mywork/
├── README.md                          ← this file
├── llava_rlhf.yml                     ← Conda env for LLaVA-RLHF 7B
├── llava-ov.yml                       ← Conda env for OneVision-1.5
├── test_images/                       ← shared probe images (both benchmarks)
│   ├── test.jpg                       ← natural-scenes probes
│   ├── clean_counter.jpg
│   ├── empty_desk.jpg
│   ├── kids_drawing.jpg
│   ├── objects_overlapping.jpg
│   ├── abstract_painting.jpg
│   ├── pure_noise.jpg                 ← extreme OOD (generate_images.py)
│   ├── solid_color.jpg
│   ├── geometric_circle.jpg
│   └── test_noisy.jpg
├── natural_scenes/
│   ├── LLAVA_INFERENCE_PATCHES.md     ← OneVision patches & troubleshooting (detailed)
│   ├── SFTvsRLHF_multi.ipynb          ← RLHF track (llava_rlhf env)
│   ├── llava_ov_multi.ipynb           ← OneVision track (llava-ov env)
│   ├── compare_probe_results.py       ← merge → probe_comparison.csv
│   ├── cross_image_probe_results.json
│   ├── llava_ov_probe_results.json
│   └── probe_comparison.csv
└── extreme_case/
    ├── generate_images.py             ← create synthetic probe images
    ├── extreme_rlhf.ipynb             ← RLHF track (llava_rlhf env)
    ├── extreme_ov.ipynb               ← OneVision track (llava-ov env)
    ├── compare_extreme_results.py     ← merge → extreme_comparison.csv
    ├── llava_rlhf_extreme_results.json
    ├── llava_ov_extreme_results.json
    └── extreme_comparison.csv
```

**Probe images:** All notebooks read from [`test_images/`](test_images/). Natural-scene photographs live here alongside synthetic extreme-OOD images (`generate_images.py` writes the four synthetic files into the same folder).

**Model checkpoints:**

| Model | Path |
|-------|------|
| LLaVA-RLHF SFT+ | `../LLaVA-RLHF/models/LLaVA-RLHF-7B-v1.5-224/sft_model` |
| LLaVA-RLHF LoRA | `../LLaVA-RLHF/models/LLaVA-RLHF-7B-v1.5-224/rlhf_lora_adapter_model` |
| LLaVA-OneVision-1.5-8B | `../LLaVA-RLHF/models/LLaVA-OneVision-1.5-8B-Instruct` |

---

## Conda environments

Two environments are required — one per model stack:

| Env | YAML | Stack | Key packages |
|-----|------|-------|--------------|
| `llava_rlhf` | [`llava_rlhf.yml`](llava_rlhf.yml) | LLaVA-RLHF 7B | Python 3.10, torch 2.0.1+cu118, transformers 4.31.0, llava, peft |
| `llava-ov` | [`llava-ov.yml`](llava-ov.yml) | OneVision-1.5-8B | Python 3.10, torch 2.4.0+cu121, **transformers 4.53.0**, qwen-vl-utils |

### First-time setup

```bash
cd mywork

conda env create -f llava_rlhf.yml
conda env create -f llava-ov.yml
```

Update an existing env after YAML changes:

```bash
conda env update -f llava_rlhf.yml --prune
conda env update -f llava-ov.yml --prune
```

**Verified hardware:** NVIDIA RTX 4000 Ada (20 GB). OneVision runs in bf16 (~17 GB) or 4-bit NF4 fallback; RLHF 7B uses 8-bit load.

---

## Benchmark 1: Natural scenes (`natural_scenes/`)

### Purpose

16 cross-image **hallucination probes** on real photographs. Tests existence, count, relation, attribute, and open-ended answers where the model might invent objects or scenes that are plausible but wrong.

### Notebooks

| Notebook | Env | Models evaluated | Output JSON |
|----------|-----|------------------|-------------|
| [`SFTvsRLHF_multi.ipynb`](natural_scenes/SFTvsRLHF_multi.ipynb) | `llava_rlhf` | SFT+ backbone + RLHF LoRA | `cross_image_probe_results.json` |
| [`llava_ov_multi.ipynb`](natural_scenes/llava_ov_multi.ipynb) | `llava-ov` | LLaVA-OneVision-1.5-8B | `llava_ov_probe_results.json` |

Both notebooks use `IMAGE_DIR = ../test_images` (six real photographs listed in the directory tree above).

Run each notebook in its own kernel (do not mix envs). Then merge:

```bash
cd mywork/natural_scenes
python compare_probe_results.py
# → probe_comparison.csv
```

### Probe overview (16)

| IDs | Image | Focus |
|-----|-------|-------|
| T1–T2 | `test.jpg` | Office scene traps (keyboard, tissue count) |
| C1–C3 | `clean_counter.jpg` | Kitchen existence / relation |
| E1–E3 | `empty_desk.jpg` | Empty-desk traps |
| K1–K3 | `kids_drawing.jpg` | Drawing vs real object |
| O1–O3 | `objects_overlapping.jpg` | Depth / relation |
| A1–A2 | `abstract_painting.jpg` | Abstract art + open-ended hallucination check |

### Scoring

- **Structured probes:** loose substring match (yes/no, count, color, left/right, front/behind).
- **Open probe (A2):** penalizes invented natural-scene words (tree, person, sky, etc.).
- JSON includes per-probe `answers`, `correct`, `verdict`, and `summary.by_model`.

---

## Benchmark 2: Extreme OOD (`extreme_case/`)

### Purpose

7 probes on **synthetic or degraded** inputs to test limitations that natural-scene accuracy may hide: meaning projection on null input, perturbation stability, and shape priors.

### Workflow

```bash
cd mywork/extreme_case

# 1. Generate synthetic images (once)
python generate_images.py
# → ../test_images/pure_noise.jpg, solid_color.jpg, geometric_circle.jpg, test_noisy.jpg

# 2. RLHF evaluation
conda activate llava_rlhf
jupyter notebook extreme_rlhf.ipynb
# → llava_rlhf_extreme_results.json

# 3. OneVision evaluation
conda activate llava-ov
jupyter notebook extreme_ov.ipynb
# → llava_ov_extreme_results.json

# 4. Compare
python compare_extreme_results.py
# → extreme_comparison.csv
```

### Synthetic images (`generate_images.py`)

Writes into `mywork/test_images/` (same folder as the natural-scene photos). `test_noisy.jpg` is derived from `test.jpg` in that directory.

| File | Content |
|------|---------|
| `pure_noise.jpg` | RGB Gaussian noise (seed 42, σ=50, 512×512) |
| `solid_color.jpg` | Flat steelblue `(70, 130, 180)` |
| `geometric_circle.jpg` | Black circle on white background |
| `test_noisy.jpg` | `test_images/test.jpg` + uniform noise (ε=20) |

### Probe table (7)

| ID | Image | Type | Question (summary) | GT | Scored? |
|----|-------|------|-------------------|-----|---------|
| A1 | `pure_noise.jpg` | existence | Real photograph? | no | yes |
| A2 | `pure_noise.jpg` | qualitative | Describe in detail | noise | **no** (manual review) |
| B1 | `solid_color.jpg` | attribute | What color? | blue | yes |
| B2 | `solid_color.jpg` | existence | Sky present? | no | yes |
| C1 | `test_noisy.jpg` | count | Tissue boxes? | 1 | yes |
| C2 | `test_noisy.jpg` | existence | Keyboard present? | no | yes |
| D1 | `geometric_circle.jpg` | existence | Drawing of animal? | no | yes |

**Headline metric:** scored accuracy = correct / 6 (A2 excluded).

### Note on answer length

Notebooks use `max_new_tokens = 128` for `"Describe..."` probes and `32` otherwise. Verbose yes/no answers may be cut mid-sentence in JSON/CSV; the stored text is the full model output up to that token limit (not truncated by the CSV exporter). See notebook comments for details.

---

## JSON output format (schema v1.0)

Both benchmarks write a common structure:

```json
{
  "schema_version": "1.0",
  "benchmark": "cross_image_hallucination_probe | extreme_ood_probe",
  "run_id": "...",
  "models": [{"key": "...", "name": "...", "path": "..."}],
  "probes": [ ... ],
  "per_probe": [
    {
      "id": "A1",
      "image": "...",
      "type": "existence",
      "question": "...",
      "gt": "no",
      "note": "...",
      "answers": {"llava_rlhf": "...", "llava_ov": "..."},
      "correct": {"llava_rlhf": true, "llava_ov": false},
      "verdict": {"llava_rlhf": "correct", "llava_ov": "wrong"}
    }
  ],
  "summary": { "total": 7, "by_model": { ... } }
}
```

Merge scripts join the two model JSONs on probe `id` and export CSV with per-model `answer_*`, `correct_*`, `verdict_*`, plus `best_models` and `all_correct`.

---

## Inference patches (OneVision)

LLaVA-OneVision was built with **transformers 4.53.0**. Running on transformers 5.x can load after monkey-patches but produces **invalid logits**. All OneVision notebooks pin 4.53.0 and hard-block inference on transformers ≥ 5.

**Full documentation:** [`natural_scenes/LLAVA_INFERENCE_PATCHES.md`](natural_scenes/LLAVA_INFERENCE_PATCHES.md)

### Quick reference

| Problem | Solution |
|---------|----------|
| Wrong transformers version | `pip install 'transformers==4.53.0'` in `llava-ov` |
| Missing `SlidingWindowCache` (5.x) | Alias to `StaticCache` at runtime |
| Missing `layer_type_validation` (5.x) | No-op validator stub |
| Missing `PreTrainedModel.set_submodule` (5.x) | Submodule shim |
| Missing `ROPE_INIT_FUNCTIONS["default"]` (5.x) | Register default RoPE function |
| Missing `pad_token_id` in config | Read from `generation_config.json` |
| `torch_dtype` → `dtype` rename (5.x) | Version-gated kwarg |
| GPU OOM / meta-device weights | 4-bit NF4 retry; reject partial offload |
| Jupyter + `argparse` | Use variables, not CLI parsing in notebooks |

Patches are applied **inline in notebook code** — no checkpoint file edits.

### LLaVA-RLHF 7B (separate stack)

Uses the original LLaVA codebase (`load_pretrained_model`, 8-bit load, `PeftModel` for LoRA). Classic prompt template:

```
USER: <image>\n{question}\nASSISTANT:
```

Does **not** require transformers 5.x compatibility patches.

### OneVision inference flow

1. `apply_transformers_compat_patches()` (load-only if on 5.x)
2. Version gate → abort if transformers ≥ 5
3. `load_config()` → fix `pad_token_id`
4. `from_pretrained()` with bf16/fp16 or 4-bit fallback
5. Qwen-VL chat messages + `qwen_vl_utils.process_vision_info`
6. Greedy `generate()` → decode new tokens only

---

## End-to-end quick start

```bash
# --- Environments (once) ---
cd mywork
conda env create -f llava_rlhf.yml
conda env create -f llava-ov.yml

# --- Natural-scenes benchmark ---
conda activate llava_rlhf
cd natural_scenes && jupyter notebook SFTvsRLHF_multi.ipynb

conda activate llava-ov
cd natural_scenes && jupyter notebook llava_ov_multi.ipynb

cd natural_scenes && python compare_probe_results.py

# --- Extreme OOD benchmark ---
cd ../extreme_case
python generate_images.py

conda activate llava_rlhf
jupyter notebook extreme_rlhf.ipynb

conda activate llava-ov
jupyter notebook extreme_ov.ipynb

python compare_extreme_results.py
```

---

## Related files outside `mywork/`

| Path | Role |
|------|------|
| `LLaVA-RLHF/models/` | Local model checkpoints |

---

## Design notes

- **Two kernels, two envs:** RLHF and OneVision cannot share one Jupyter kernel due to incompatible torch/transformers stacks.
- **Comparison CSVs** are the primary deliverables for cross-model analysis; JSON files retain full probe metadata and raw answers.
