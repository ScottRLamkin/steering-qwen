"""Device selection and model loading."""

import logging
import sys

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def pick_device(pref: str | None = None) -> torch.device:
    """Return ``pref`` if given, else the best available of cuda > mps > cpu."""
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def nf4_config() -> BitsAndBytesConfig:
    """The 4-bit config from the original notebook."""
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)


QUANT_KINDS = ("nf4", "fp4", "int8")


def quant_config(kind: str = "nf4") -> BitsAndBytesConfig:
    """bitsandbytes weight quantization: 4-bit ``nf4`` (the notebook's) or ``fp4``, or 8-bit ``int8``
    (LLM.int8, which keeps outlier features in bf16). All three run on MPS; int8 is about 3x slower there."""
    if kind == "nf4":
        return nf4_config()
    if kind == "fp4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.bfloat16)
    if kind == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"quantization {kind!r}: choose from {QUANT_KINDS}")


class _FirstOnly(logging.Filter):
    """Let a log message through once. bitsandbytes logs its int8 bf16->fp16 cast on every matmul."""

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix, self.seen = prefix, False

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.getMessage().startswith(self.prefix):
            return True
        first, self.seen = not self.seen, True
        return first


_INT8_CAST = _FirstOnly("MatMul8bitLt: inputs will be cast")


def quant_kind(model: PreTrainedModel) -> str | None:
    """``nf4``, ``fp4`` or ``int8`` for a bitsandbytes-quantized model, else None."""
    q = getattr(model, "quantization_config", None) or getattr(model.config, "quantization_config", None)
    if not q:
        return None
    get = q.get if isinstance(q, dict) else lambda k, d=None: getattr(q, k, d)
    if get("load_in_8bit"):
        return "int8"
    return get("bnb_4bit_quant_type", "nf4") if get("load_in_4bit") else str(get("quant_method", "quantized"))


def load_model(
    model_id: str = DEFAULT_MODEL,
    device: str | torch.device | None = None,
    quantize: bool | str | None = None,
    max_memory: dict | None = None,
    verbose: bool = True,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load tokenizer and model in bf16, optionally quantized via bitsandbytes.

    ``quantize`` is True (4-bit NF4), False, or one of ``QUANT_KINDS``. It defaults to NF4 on CUDA
    (as in the original notebook) and to unquantized bf16 elsewhere.
    bitsandbytes also runs NF4 on MPS and CPU, so ``quantize=True`` on a Mac reproduces the
    CUDA run's weights; unquantized bf16 needs ~9 GB for the 4B model.

    ``max_memory`` only applies on CUDA, where the model is placed with ``device_map="auto"``;
    e.g. ``{0: "5GiB", "cpu": "16GiB"}`` for a small GPU.
    """
    dev = pick_device(str(device) if device is not None else None)
    if quantize is None:
        quantize = dev.type == "cuda"
    kind = ("nf4" if quantize is True else quantize) if quantize else None

    kwargs: dict = {"dtype": torch.bfloat16}
    if kind:
        kwargs["quantization_config"] = quant_config(kind)
    if kind == "int8":
        log = logging.getLogger("bitsandbytes.autograd._functions")
        if _INT8_CAST not in log.filters:
            log.addFilter(_INT8_CAST)
    if dev.type == "cuda":
        kwargs["device_map"] = "auto"
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
    else:
        kwargs["device_map"] = {"": str(dev)}

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    if verbose:
        gib = model.get_memory_footprint() / 2**30
        q = {"nf4": "4-bit NF4", "fp4": "4-bit FP4", "int8": "8-bit int8"}.get(kind, "unquantized")
        print(f"Loaded {model_id} on {model.device} ({model.dtype}, {q}, {gib:.2f} GiB)", file=sys.stderr)
    return tok, model


def find_layers(m: nn.Module) -> tuple[nn.ModuleList, str]:
    """Locate the decoder layer list (path differs between text-only and multimodal wrappers)."""
    for path in ["model.layers", "model.language_model.layers", "language_model.model.layers", "layers"]:
        obj = m
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and hasattr(obj[0], "mlp"):
            return obj, path
    raise RuntimeError("print(model) and locate the decoder layer list")
