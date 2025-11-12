# Hybrid Quantum‑Classical Vision–Language Model

This repository demonstrates how to build a **hybrid classical‑quantum model** for image captioning (or diagnostic pipelines) by inserting a small quantum neural network (QNN) into the TinyLlama‑VLM‑LoRA architecture. The goal is not to produce a state‑of‑the‑art captioner but to illustrate how quantum circuits can augment vision–language models in a modular way. The code lives in [hybrid_vlm_quantum.py](hybrid_vlm_quantum.py).

---

## Background

### TinyLlama‑VLM‑LoRA

TinyLlama‑VLM‑LoRA is a lightweight vision–language model built by combining a **frozen CLIP vision encoder** with a TinyLlama language model adapted through **Low‑Rank Adaptation (LoRA)**. 

- **LoRA** reduces the number of trainable parameters by decomposing large weight matrices into two smaller low‑rank matrices. Only these small matrices are learned; the base TinyLlama weights remain frozen while training the adapter, projector, and gate layers.
- The CLIP encoder produces a 768‑dimensional CLS embedding which is projected down and expanded back up to form a single “vision token” that is prepended to the text tokens. TinyLlama then predicts the caption conditioned on this vision token.

### Hybrid Quantum–Classical Models

Quantum programming differs fundamentally from classical programming:
- **Classical programs** manipulate bits that are either 0 or 1.
- **Quantum programs** manipulate qubits that can exist in multiple states simultaneously (superposition) and become entangled.

These properties allow quantum computers to process information in parallel and, for some problems, offer speedups over classical algorithms.

- **Noisy intermediate‑scale quantum (NISQ)** devices currently support only small circuits and few qubits.
- **Hybrid algorithms** combine classical neural networks with quantum subroutines: a classical optimizer proposes parameters, a quantum circuit evaluates a feature map, and the results guide further updates—a feedback loop central to variational quantum algorithms used in quantum machine learning.

---

## Approach

We replace the classical projector+gate modules in TinyLlama‑VLM‑LoRA with a **quantum projection layer** using PennyLane. The hybrid architecture consists of:

1. **Vision encoding**: A frozen CLIP model converts the input image into a 768‑dimensional CLS embedding.
2. **Quantum projection**: A small quantum circuit compresses the CLS embedding, processes it via entangling gates with trainable parameters, and measures expectation values of Pauli‑Z operators. These values are mapped up to TinyLlama’s hidden dimension (4096).
3. **Caption generation**: The LoRA-adapted TinyLlama language model receives the vision token, followed by text embeddings and generates a caption.

The QNN acts as a nonlinear feature extractor in the quantum Hilbert space. Because NISQ devices are limited, the example uses just 4 qubits by default. You can adjust this parameter.

---

## Code Walkthrough

The core code is in [hybrid_vlm_quantum.py](hybrid_vlm_quantum.py).

### QuantumProjection (lines 67–121)

A `torch.nn.Module` that performs:
- **Preprocessing (line 82)**: Reduces the 768‑dimensional CLIP embedding to `n_qubits` with a learnable linear layer, selecting features fed into the quantum circuit.
- **Quantum circuit (lines 96–113)**: PennyLane qnode embeds the vector as rotation angles, applies three layers of entangling gates (variational params), and measures expectation values of Pauli‑Z for each qubit.
- **Post-processing (line 117)**: Maps expectation values to 4096 dimensions as the "vision token" for text concatenation.

### HybridVisionLanguageModel (lines 125–181)

Combines the frozen CLIP vision encoder, the quantum projection, and the LoRA-adapted TinyLlama:
- Extracts the CLS embedding from CLIP and processes it via the quantum projection (lines 148–155).
- Concatenates the vision token with truncated text embeddings (line 160).
- Constructs attention masks and labels for teacher forcing (lines 163–177).
- Calls the TinyLlama model on combined embeddings (only LoRA adapters are trained; base weights are frozen).

### Helper Functions

- `load_models`: Loads the frozen CLIP model, LoRA-adapted TinyLlama, and the tokenizer/processor (lines 186–209).
- `generate_caption`: Constructs the hybrid model, preprocesses an image, tokenizes the prompt, runs inference, and generates the caption (lines 213–260).
- `main`: Parses command-line arguments and prints the generated caption (lines 263–274).

---

## Running the Example

1. **Install dependencies** (need to be done on your own machine):

    ```sh
    pip install torch torchvision transformers peft pennylane pillow
    ```

2. **Download the TinyLlama‑VLM‑LoRA model components.** The script will pull weights automatically from Hugging Face on first run.

3. **Run the script on an image:**

    ```sh
    python hybrid_vlm_quantum.py --image_path path/to/your/image.jpg \
        --prompt "Describe the image: " --n_qubits 4
    ```

    This prints the generated caption. You can experiment with more qubits or different prompts. Note: increasing `n_qubits` increases quantum circuit depth and number of parameters, which may challenge today's NISQ hardware.

---

## Alternative Approaches and Extensions

### Using Qiskit or Cirq

- **Qiskit**: Offers a NeuralNetwork interface to wrap quantum circuits for differentiable layers in PyTorch. You can use TwoLocal circuits and SamplerQNN/EstimatorQNN, with support for noise modeling and real-device execution.
- **Cirq / TensorFlow Quantum**: Define parameterized circuits with Cirq, use tfq for integration with TensorFlow layers, and replace the PyTorch-based QuantumProjection with analogous TensorFlow layers.

### Other Integration Points

- **Replace only the gate**: Keep the classical projector (768→512), but use a quantum circuit for the gate (512→4096).
- **Quantum attention**: Compute attention scores or modify values within TinyLlama’s attention blocks using a quantum circuit.
- **Diagnostic pipelines**: Encode medical images into quantum states (e.g., amplitude encoding) and input expectation values to a classifier or language model as needed.

---

## Limitations

Hybrid quantum–classical models face serious hardware constraints: NISQ devices allow only shallow and small circuits, and noise is significant. Quantum enhancements are mostly theoretical for now, but these models offer a testbed for future advances as machines become more capable.

---

## References

- **Low‑Rank Adaptation (LoRA)**: [LoRA overview](https://huggingface.co/docs/peft/conceptual_guides/lora)
- **TinyLlama‑VLM‑LoRA architecture**: [Model card](https://huggingface.co/sriram7737/TinyLlama-VLM-LoRA)
- **Quantum programming concepts**: [Quantum programming articles](https://www.postquantum.com/resources/)
- **Variational quantum algorithms**: [Intro to VQAs](https://www.quera.com/glossary/variational-quantum-algorithm)
