# Crier — the Fifth Pillar

> *"Hear ye, hear ye"* — Crier is the fifth pillar of Chamberlain: a cross-platform Python **library** that runs a small language model on the local NPU (or GPU, or CPU) of the host machine, behind a single clean interface. No HTTP, no Docker, no LM Studio dependency. Other pillars import it directly.

Architectural contract: see `specification.md` §7.

## Why a fifth pillar?

The other four pillars (Catchpole, Scribe, Miller, Bailiff) all rely on an external inference engine (LM Studio on a separate host) to serve completions. Crier closes that loop: when a request is small enough to be served *inside the process*, Catchpole can call into Crier directly instead of round-tripping to LM Studio. The eventual goal is a third routing target in Catchpole alongside `LOCAL` and `CLOUD`: `EMBEDDED`.

Crier is a **library** consumed in-process. It is intentionally *not* a service.

## Backend matrix

| Backend | OS | EP | Pip extra | Typical use |
| --- | --- | --- | --- | --- |
| `cpu` | Win / Linux / macOS | `CPUExecutionProvider` | `crier[cpu]` | Universal fallback. |
| `directml` | Windows | `DmlExecutionProvider` | `crier[directml]` | Generic Windows GPU/NPU (any vendor). |
| `coreml` | macOS | `CoreMLExecutionProvider` | `crier[coreml]` | Apple Silicon / Apple Neural Engine. |
| `ryzenai` | Windows / Linux | `VitisAIExecutionProvider` | `crier[ryzenai]` | AMD Ryzen AI 300 series (XDNA 2 NPU). |
| `openvino` | Windows / Linux | `OpenVINOExecutionProvider` | `crier[openvino]` | Intel Core Ultra (NPU + iGPU). |
| `qnn` | Windows / Linux | `QNNExecutionProvider` | `crier[qnn]` | Qualcomm Snapdragon X. |
| `cuda` | Windows / Linux | `CUDAExecutionProvider` | `crier[cuda]` | Discrete NVIDIA GPU (not an NPU, included for symmetry). |

**Important caveat.** `pip install crier[ryzenai]` installs the Python wheel; it does **not** install the XDNA driver or the AMD Ryzen AI Software stack. The same is true for Intel NPU drivers and Qualcomm QNN runtime. Run `crier doctor` after install to see what's missing on your host — it will tell you which Python package is present, which provider is registered, and what to install next.

## Install

Pick the extra that matches your host. CPU works everywhere and is the default.

```bash
# AMD Ryzen AI 300 / Strix Point laptop (Windows)
pip install "crier[ryzenai]"

# Intel Core Ultra (Windows or Linux)
pip install "crier[openvino]"

# Snapdragon X / Copilot+ PC (Windows)
pip install "crier[qnn]"

# Apple Silicon Mac
pip install "crier[coreml]"

# Generic Windows GPU/NPU passthrough
pip install "crier[directml]"

# Anywhere, no acceleration
pip install "crier[cpu]"

# Multiple at once is fine
pip install "crier[directml,ryzenai]"
```

After installing the extra you may also need a vendor system package:

| Vendor | What pip cannot install for you |
| --- | --- |
| AMD | Ryzen AI Software 1.3+, including the XDNA NPU driver. Download from the AMD developer portal. |
| Intel | Intel NPU driver from the Intel download centre, then OpenVINO runtime is installed by the extra. |
| Qualcomm | QNN runtime DLLs (`QnnHtp.dll` etc.) bundled with Snapdragon Windows or the QNN SDK. |
| Apple | Nothing — CoreML is part of macOS. |

## Diagnostic first

After install, run:

```bash
crier doctor
```

Sample output on a clean Linux dev box (only CPU installed):

```
   backend            status  detail
----------            ------  ----------------------------------------
       cpu  missing-package  Python package 'onnxruntime_genai' not installed.
                             hint: pip install crier[cpu]
  directml  missing-package  Python package 'onnxruntime_genai' not installed.
                             hint: pip install crier[directml]
   ryzenai  missing-package  Python package 'onnxruntime_genai' not installed.
                             hint: pip install crier[ryzenai]
...
```

Use it before reporting bugs.

## Usage

### Library

```python
from crier import LLM, Message, GenerationConfig

llm = LLM.load(
    model="phi-3.5-mini-instruct",   # logical preset name
    accelerator="auto",              # auto-detect the best EP for this host
    require_acceleration=False,      # set True to refuse silent CPU fallback
)

print(llm.info)
# BackendInfo(name='ryzenai', execution_provider='VitisAIExecutionProvider',
#             device='npu', model_name='phi-3.5-mini-instruct',
#             model_path='~/.cache/crier/models/...', accelerated=True,
#             attempted=(), fallback_reason=None)

reply = llm.generate(
    [
        Message(role="system", content="You are concise."),
        Message(role="user", content="Why is the sky blue?"),
    ],
    GenerationConfig(max_tokens=128, temperature=0.7),
)
print(reply.text)

# Streaming
for chunk in llm.stream(messages, config):
    print(chunk.text, end="", flush=True)

# Async (Catchpole-style consumers)
async for chunk in llm.astream(messages, config):
    await pipe.write(chunk.text)

llm.close()
```

### CLI (sanity check)

```bash
crier generate "Write a Python fizzbuzz." --accelerator auto --max-tokens 256
crier generate "..." --stream
crier doctor
```

### Bring your own model

The preset registry is a curated short list. For anything else, build a `ModelSpec`:

```python
from crier import LLM, ModelSpec

spec = ModelSpec(
    name="my-model",
    backend="ryzenai",
    repo_id="amd/Phi-4-mini-instruct-awq-asym-uint4-g128-lmhead-onnx-hybrid",
    revision="main",
    quantization="int4-awq-hybrid",
)
llm = LLM.load(model=spec, accelerator="ryzenai")
```

Local-only models work too:

```python
llm = LLM.load(model="/path/to/onnx/bundle", accelerator="cpu")
```

## Currently shipped presets

```python
from crier import list_presets
print(list_presets())
# [('phi-3.5-mini-instruct', 'cpu'),
#  ('phi-3.5-mini-instruct', 'cuda'),
#  ('phi-3.5-mini-instruct', 'directml'),
#  ('phi-3.5-mini-instruct', 'ryzenai')]
```

Presets pin to Microsoft / AMD official ONNX repositories. The Ryzen AI
preset points at AMD's **gated** Hugging Face repo — you must accept the
model card terms on huggingface.co and run `huggingface-cli login` before
the download will succeed.

The preset list is intentionally short — only combinations we've actually
verified ship as presets. For anything else (other Phi variants, Llama,
Qwen, Mistral), build a `ModelSpec` directly.

Crier never redistributes weights; you download from Hugging Face under each model's own licence.

## Concurrency contract

**One `LLM` supports one active generation at a time.** Streaming and `generate` are not safe to interleave on the same instance from multiple threads or tasks. The instance holds an internal lock that serialises calls; if you need parallelism, create multiple `LLM` instances (each will hold its own model in memory) or wrap calls in an external queue.

## Development

```bash
git clone https://github.com/quintindk/chamberlain
cd chamberlain/crier
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Fast unit tests (mocked, no ORT GenAI needed)
pytest

# Lint
ruff check src tests
```

### Unit vs integration tests

| Layer | What it covers | Where it runs | When |
| --- | --- | --- | --- |
| Unit (`tests/`) | Wrapper logic, backend selection, model resolution, error taxonomy. ORT GenAI is **mocked** at the adapter boundary. | Win + Linux + macOS × Python 3.10/3.11/3.12 | Every push (~30 s per cell). |
| Integration (`tests/integration/`) | Real model load + a 64-token generation through ORT GenAI on the CPU EP. Downloads ~2 GB on first run. | Win + Linux + macOS × Python 3.12 | Manual `workflow_dispatch` only, gated by `CRIER_RUN_INTEGRATION=1`. |

Run integration tests locally:

```bash
pip install -e ".[dev,cpu]"
CRIER_RUN_INTEGRATION=1 pytest -m integration tests/integration
```

## Roadmap

- Embeddings API on top of ORT GenAI's embedding support.
- True async streaming with cancellation + backpressure (current implementation is "good enough" but uses a worker thread).
- MLX backend for Apple Silicon (likely sharper than CoreML for Apple users).
- Catchpole integration: third routing target `EMBEDDED` that calls into Crier in-process.
- Tool / function calling pass-through once ORT GenAI exposes a stable surface for it.

## Licence

MIT. See `LICENSE`. Model weights downloaded by Crier are governed by their own licences (Microsoft Research Licence for Phi, Meta Llama Community Licence for Llama, etc.) — read them.
