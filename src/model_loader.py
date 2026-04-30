"""Load Qwen2.5-1.5B-Instruct for hidden-state extraction."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model():
    """Load the model, tokenizer, and a small config dict.

    Returns
    -------
    model : transformers.PreTrainedModel
        Loaded in eval mode, on MPS if available else CPU, in float32.
    tokenizer : transformers.PreTrainedTokenizer
        With pad_token set to eos_token if missing.
    config : dict
        Keys: n_layers, hidden_size, n_hidden_states, device, dtype.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[model_loader] Using device: {device}")

    # float32 is intentional: float16 introduces numerical noise on hidden
    # states that can bias probe accuracies. Scientific rigor > speed here.
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model.to(device)
    model.eval()

    config = {
        "n_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "n_hidden_states": model.config.num_hidden_layers + 1,
        "device": device,
        "dtype": dtype,
    }
    return model, tokenizer, config
