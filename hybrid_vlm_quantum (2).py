"""
hybrid_vlm_quantum.py
======================

This module demonstrates how to build a **hybrid classical‑quantum vision
language model** by grafting a small quantum neural network (QNN) into
the TinyLlama‑VLM‑LoRA architecture.  The goal is to show how a
quantum circuit can be used as a feature extractor that sits
between a frozen vision encoder and a language model.  The example
targets image captioning but the same pattern could be adapted for
medical diagnosis or other multimodal tasks.

The implementation uses the following components:

* **CLIP vision encoder** – produces a 768‑dimensional [CLS] token from
  an RGB image.  We rely on `transformers.CLIPModel` for this.  The
  vision encoder is kept frozen during training.
* **LoRA‑adapted TinyLlama** – a 1.1‑billion parameter language model
  with Low‑Rank Adaptation (LoRA) adapters attached to the
  query/value projection sublayers of its attention blocks.  LoRA
  reduces the number of trainable parameters by decomposing a large
  matrix into two smaller low‑rank matrices【438111095739304†L103-L105】.  The base
  TinyLlama weights remain frozen and only the adapters, projector and
  gate are trained during fine‑tuning【504481890058751†L59-L81】.
* **Quantum projection layer** – a small quantum neural network built
  with PennyLane that processes a down‑sampled CLIP embedding.  It
  encodes the data into qubit rotations, applies entangling gates
  (variational parameters) and measures expectation values.  These
  expectation values are mapped back into the high‑dimensional space
  expected by TinyLlama.
* **Hybrid vision–language model** – glues the above pieces together.
  It takes pixel values, tokenizes a prompt, inserts the quantum
  processed vision token, and delegates caption generation to
  TinyLlama.

This example is written for educational purposes.  It is not intended
to be computationally efficient on today’s hardware; instead it shows
how to wrap a quantum circuit inside a PyTorch module and integrate it
with a pre‑trained transformer.  The quantum layer operates on only
a few qubits to stay within the noise budget of NISQ devices【245392272704463†L213-L244】.

Requirements
------------

To run this script you need the following Python packages installed:

```
pip install torch transformers peft pennylane pillow
```

The script does **not** run on the host used for this answer because
PyTorch and other heavy dependencies are not available.  You should
run it in your own environment with the appropriate libraries.

Example usage:

```
python hybrid_vlm_quantum.py --image_path path/to/image.jpg \
       --prompt "Describe the image: "
```

It will print the generated caption.  You can adjust the number of
qubits and other hyper‑parameters via command‑line arguments.

"""

import argparse
from typing import List, Tuple

# Third‑party imports.  These imports are enclosed in try/except blocks
# so that the module can be imported even on systems where the
# dependencies are missing.  If you run this script yourself you
# should install the required packages as described in the docstring.
try:
    import torch
    import torch.nn as nn
    from transformers import (
        CLIPModel,
        CLIPProcessor,
        AutoTokenizer,
        AutoModelForCausalLM,
    )
    from peft import PeftModel
    import pennylane as qml
    from PIL import Image
except ImportError as exc:
    raise ImportError(
        "Required packages are missing.  Please install torch, transformers, peft,"
        " pennylane and pillow to run this script."
    ) from exc


class QuantumProjection(nn.Module):
    """Quantum layer that maps a CLIP embedding into TinyLlama's hidden size.

    This layer performs three steps:

    1. **Pre‑processing:** reduce the 768‑dimensional CLIP [CLS] embedding
       down to a small number of features (equal to the number of qubits).
       This is implemented with a learnable linear projection.
    2. **Quantum circuit:** encode the down‑sampled features into qubit
       rotations, apply entangling gates with trainable parameters and
       measure expectation values.  The circuit follows the typical
       variational pattern used in hybrid algorithms【245392272704463†L213-L244】.
    3. **Post‑processing:** map the measured expectation values up to
       TinyLlama's hidden size (4096) via another linear layer.  This
       produces the “vision token” that will be concatenated with
       textual embeddings.

    Parameters
    ----------
    n_qubits : int
        Number of qubits / features used by the quantum circuit.
    hidden_size : int
        Target dimension for the vision token (TinyLlama hidden size).
    """

    def __init__(self, n_qubits: int = 4, hidden_size: int = 4096) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.hidden_size = hidden_size

        # Linear layer to project 768‑dimensional CLIP embedding down to n_qubits.
        self.preprocess = nn.Linear(768, n_qubits)

        # Define a PennyLane quantum device with 'n_qubits' wires.  We use
        # the default.qubit simulator here; on real hardware you could
        # switch to a hardware‑specific device.
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Shape of the variational parameters: [n_layers, n_qubits].  Here
        # we choose three layers of entangling gates for demonstration.
        self.weight_shape = (3, n_qubits)

        # Register the trainable weights as a PyTorch parameter so that
        # gradients flow through the quantum circuit.  The initial
        # values are drawn from a normal distribution.
        self.quantum_weights = nn.Parameter(
            torch.randn(self.weight_shape, dtype=torch.float32)
        )

        # Post‑processing linear layer: maps the n_qubit expectation
        # values up to TinyLlama's hidden size.
        self.postprocess = nn.Linear(n_qubits, hidden_size)

        # Define the quantum circuit as a QNode.  The `qml.qnode`
        # decorator binds our device and the PyTorch interface.  The
        # circuit expects two arguments: the input data (angle vector)
        # and the trainable weights.  It returns a list of expectation
        # values, one per qubit.
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> List[torch.Tensor]:
            # Embed classical data into quantum state using angle embedding.
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            # Apply variational entangling layers parameterized by 'weights'.
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            # Measure the expectation value of the Pauli‑Z operator on each qubit.
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Store the qnode in the module so it can be called in forward().
        self.circuit = circuit

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum projection layer.

        Parameters
        ----------
        cls_embedding : torch.Tensor
            A batch of CLIP [CLS] embeddings of shape ``(batch_size, 768)``.

        Returns
        -------
        torch.Tensor
            A batch of vision tokens of shape ``(batch_size, hidden_size)``.

        Notes
        -----
        This implementation uses PennyLane's *broadcasting* feature
        rather than iterating over the batch.  When a QNode receives a
        batch of inputs, it broadcasts the execution and returns a
        stacked tensor of results.  Broadcasting can be significantly
        faster than looping over each sample【129125191415563†L349-L396】.
        """
        # Reduce dimensionality to the number of qubits.  This
        # learnable projection selects which components of the CLIP
        # embedding are passed to the quantum circuit.
        reduced = self.preprocess(cls_embedding)  # (batch_size, n_qubits)

        # Execute the quantum circuit in a batched manner.  For
        # batched inputs the QNode returns either a tensor of shape
        # (batch_size, n_qubits) or a list of ``n_qubits`` tensors,
        # each of shape (batch_size,).  We handle both cases.
        expvals = self.circuit(reduced, self.quantum_weights)
        if isinstance(expvals, list):
            expvals_tensor = torch.stack(expvals, dim=1)  # (batch_size, n_qubits)
        else:
            expvals_tensor = expvals

        # Project the expectation values up to TinyLlama's hidden size.
        vision_token = self.postprocess(expvals_tensor)
        return vision_token


class HybridVisionLanguageModel(nn.Module):
    """Wrap CLIP, quantum projection and TinyLlama into a single module.

    This class mimics the structure of the TinyLlama‑VLM‑LoRA
    evaluation script but replaces the classical projector + gate with
    a quantum projection layer.  It handles tokenization, attention
    mask construction and the combination of vision and language
    representations.
    """

    def __init__(
        self,
        clip_model: CLIPModel,
        text_model: nn.Module,
        quantum_proj: QuantumProjection,
        device: torch.device,
        max_length: int = 50,
        prompt: str = "Describe the image: ",
    ) -> None:
        super().__init__()
        # Use only the vision sub‑module of CLIP.  Freeze its parameters
        # since we do not fine‑tune the vision encoder.
        self.vision = clip_model.vision_model
        for p in self.vision.parameters():
            p.requires_grad = False

        self.text_model = text_model
        self.quantum_proj = quantum_proj.to(device)
        self.device = device
        self.max_length = max_length
        self.prompt_text = prompt

    def forward(
        self,
        pixel_values: torch.Tensor,
        full_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the hybrid VLM.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Batch of preprocessed images of shape (B, 3, 224, 224) in
            FP16.  These are fed into the frozen CLIP vision encoder.
        full_input_ids : torch.Tensor
            Input token IDs of shape (B, L).  This includes the prompt
            followed by padding.  The last token is ignored during
            teacher‑forcing.
        attention_mask : torch.Tensor
            Attention mask of shape (B, L) where 1 indicates a valid
            token (not padding).
        prompt_len : torch.Tensor
            Length of the prompt for each sample (batch dimension).
        labels : torch.Tensor | None, optional
            If provided, labels for computing the cross‑entropy loss.

        Returns
        -------
        torch.Tensor
            The output from TinyLlama (loss and logits); see Hugging
            Face `transformers` documentation for details.
        """
        B, L = full_input_ids.size()

        # Run the frozen vision encoder.  We do this in no_grad
        # context to save memory and because we do not need
        # gradients through CLIP weights.
        with torch.no_grad():
            vision_out = self.vision(pixel_values=pixel_values)
            cls_embed = vision_out.last_hidden_state[:, 0, :]  # (B, 768)

        # Quantum processing: produce the vision token from the CLIP
        # embedding.  The result has dtype FP32; we cast to FP16 to
        # match the TinyLlama model's precision.
        vision_token_fp32 = self.quantum_proj(cls_embed)
        vision_token = vision_token_fp32.to(torch.float16)  # (B, hidden_size)

        # Prepare textual embeddings.  We drop the last token in
        # teacher‑forcing mode because TinyLlama predicts the next token.
        input_ids_trunc = full_input_ids[:, :-1]
        text_embeds = self.text_model.get_input_embeddings()(input_ids_trunc)
        # Concatenate the vision token at the beginning of the sequence.
        combined_embeds = torch.cat(
            [vision_token.unsqueeze(1), text_embeds], dim=1
        )  # (B, L, hidden_size)

        # Build the combined attention mask: a single "1" for the vision
        # token followed by the original mask (excluding the last
        # position).
        vision_attn = torch.ones((B, 1), device=self.device, dtype=attention_mask.dtype)
        combined_attn = torch.cat(
            [vision_attn, attention_mask[:, :-1]], dim=1
        )  # (B, L)

        # Generate labels for masked language modelling if not provided.
        if labels is None:
            # Initialize all positions to ignore_index = -100.
            labels = torch.full(
                (B, L), -100, dtype=torch.long, device=self.device
            )
            for i in range(B):
                P = prompt_len[i].item()
                total_nonpad = attention_mask[i].sum().item()
                raw_c = total_nonpad - P
                max_c = (L - 1 - P)
                c = min(raw_c, max(0, max_c))
                if c > 0:
                    start = 1 + P
                    end = 1 + P + c
                    labels[i, start:end] = full_input_ids[i, P : P + c]

        # Pass through the TinyLlama model.  In the LoRA setup the
        # TinyLlama weights are frozen and only the adapters are
        # trainable【504481890058751†L59-L81】.  The model accepts
        # `inputs_embeds` and `attention_mask` instead of raw tokens.
        outputs = self.text_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn,
            labels=labels,
        )
        return outputs


def load_models(device: torch.device) -> Tuple[CLIPModel, nn.Module, AutoTokenizer, CLIPProcessor]:
    """Load CLIP, LoRA‑adapted TinyLlama and tokenizer.

    The function follows the example from the TinyLlama‑VLM‑LoRA model
    card【504481890058751†L140-L169】.  It returns the frozen CLIP model,
    the language model with LoRA adapters, the tokenizer and the
    CLIP processor for preprocessing images.
    """
    # 1) CLIP vision + text
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", torch_dtype=torch.float16
    ).to(device)
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    # 2) Base TinyLlama and LoRA adapter.  The base model is kept
    # frozen; LoRA adapters add trainable low‑rank matrices onto
    # q_proj and v_proj layers【504481890058751†L59-L81】.
    base_llama = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    # The adapter weights live inside the TinyLlama‑VLM‑LoRA repository.
    llama = PeftModel.from_pretrained(
        base_llama,
        "sriram7737/TinyLlama-VLM-LoRA/adapter_model.safetensors",
    ).to(device)
    llama.eval()

    # 3) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "sriram7737/TinyLlama-VLM-LoRA/finetuned_qvlam_flickr30k_final"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return clip_model, llama, tokenizer, clip_processor


def generate_caption(
    image_path: str,
    prompt: str,
    n_qubits: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """Generate a caption for an image using the hybrid VLM.

    This convenience function loads the necessary models, constructs
    the hybrid architecture, preprocesses the image and runs
    inference.  It returns the generated caption as a string.
    """
    device_t = torch.device(device)
    # Load pre‑trained models and tokenizer
    clip_model, llama, tokenizer, clip_processor = load_models(device_t)

    # Instantiate the quantum projection layer
    quantum_proj = QuantumProjection(n_qubits=n_qubits)

    # Build the hybrid model
    hybrid_vlm = HybridVisionLanguageModel(
        clip_model=clip_model,
        text_model=llama,
        quantum_proj=quantum_proj,
        device=device_t,
        max_length=50,
        prompt=prompt,
    ).to(device_t)
    hybrid_vlm.eval()

    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    clip_inputs = clip_processor(images=img, return_tensors="pt")
    pixel_values = clip_inputs.pixel_values.to(device_t).to(torch.float16)

    # Tokenize the prompt.  We pad/truncate to max_length.
    prompt_tok = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=hybrid_vlm.max_length,
        truncation=True,
    )
    prompt_ids = prompt_tok.input_ids.to(device_t)
    prompt_attn = prompt_tok.attention_mask.to(device_t)
    prompt_len = torch.tensor([prompt_ids.size(1)], device=device_t)

    # Build a dummy full_input_ids padded to max_length (TinyLlama expects
    # full sequence; unused tokens will be ignored).
    full_input_ids = torch.full(
        (1, hybrid_vlm.max_length), tokenizer.pad_token_id, dtype=torch.long, device=device_t
    )
    full_input_ids[:, : prompt_ids.size(1)] = prompt_ids
    attention_mask = torch.zeros(
        (1, hybrid_vlm.max_length), dtype=torch.long, device=device_t
    )
    attention_mask[:, : prompt_attn.size(1)] = prompt_attn

    # Perform inference in no‑grad mode.  We avoid an initial call to
    # ``hybrid_vlm`` because that would run the vision and quantum
    # layers twice.  Instead we directly compute the vision token
    # and build the embeddings for generation.  See also the
    # commentary in the README about redundant computation.
    with torch.no_grad():
        # Compute the CLIP [CLS] embedding and pass it through the
        # quantum projection layer to produce the vision token.  The
        # projection returns FP32; cast to FP16 to match TinyLlama.
        cls_embed = hybrid_vlm.vision(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        vision_token = hybrid_vlm.quantum_proj(cls_embed).to(torch.float16)

        # Obtain text embeddings for the prompt.  We use only the
        # prompt IDs since there are no label tokens during
        # generation.
        text_embeds = hybrid_vlm.text_model.get_input_embeddings()(prompt_ids)
        combined_embeds = torch.cat(
            [vision_token.unsqueeze(1), text_embeds], dim=1
        )

        # Build the attention mask.  Unlike the training forward pass
        # there is no need to drop the last token; the entire prompt
        # should attend to the vision token and to itself.  We prepend
        # a ``1`` for the vision token then use the full prompt
        # attention mask【129125191415563†L349-L396】.
        vision_attn = torch.ones((1, 1), device=device_t, dtype=torch.long)
        combined_attn = torch.cat([vision_attn, prompt_attn], dim=1)

        # Generate the caption using TinyLlama.  We pass the
        # precomputed embeddings and attention mask directly.  The
        # LoRA adapters handle the multimodal context.
        generated_ids = hybrid_vlm.text_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn,
            max_length=hybrid_vlm.max_length,
            num_beams=3,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return caption


def main() -> None:
    # Parse command‑line arguments
    parser = argparse.ArgumentParser(description="Hybrid quantum‑classical VLM captioning")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image: ",
        help="Prompt text to prefix the caption generation",
    )
    parser.add_argument(
        "--n_qubits",
        type=int,
        default=4,
        help="Number of qubits (features) used by the quantum projection",
    )
    args = parser.parse_args()

    caption = generate_caption(args.image_path, args.prompt, n_qubits=args.n_qubits)
    print("Generated caption:", caption)


if __name__ == "__main__":
    main()