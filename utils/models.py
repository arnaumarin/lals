"""Models, occupations, images, and hidden-state extraction from the four VLMs (GPU).

Every model wrapper exposes:

* ``visual_token_states(image, layers)`` -> ``{layer: [n_visual_tokens, dim]}``: output of decoder
  block ``layer`` (0-indexed) at the image-token positions, for the prompt "Describe this
  image." in the model's chat format.
* ``text_token_states(texts, layers)`` -> ``({layer: [n_tokens, dim]}, metadata)``: text-only
  forward pass of each sentence on its own (no padding, at most 128 tokens), dropping special
  tokens and the first remaining token (attention sink). Builds the text database.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------------------------
# Models, occupations and images
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    hf_id: str
    family: str  # "qwen", "llava" or "internvl"
    n_layers: int
    sweep_layers: tuple[int, ...]
    text_db: str  # key of the text database the model is scored against
    processor_id: str | None = None  # defaults to hf_id


MODELS: dict[str, ModelSpec] = {m.key: m for m in [
    ModelSpec("qwen2vl", "Qwen2-VL-7B", "Qwen/Qwen2-VL-7B-Instruct", "qwen", 28,
              (4, 8, 12, 16, 20, 24, 27), "qwen2vl"),
    ModelSpec("qwen25vl", "Qwen2.5-VL-7B", "Qwen/Qwen2.5-VL-7B-Instruct", "qwen", 28,
              (4, 8, 12, 16, 20, 24, 27), "qwen25vl"),
    ModelSpec("llava", "LLaVA-v1.6-7B", "llava-hf/llava-v1.6-mistral-7b-hf", "llava", 32,
              (4, 9, 14, 18, 23, 27, 31), "llava"),
    ModelSpec("internvl", "InternVL2.5-8B", "OpenGVLab/InternVL2_5-8B", "internvl", 32,
              (4, 9, 14, 18, 23, 27, 31), "internvl"),
    # Pre-instruction-tuning checkpoint (Fig. 11). As in the paper's run, it uses the
    # Instruct processor (chat template) and is scored against the Instruct text database.
    ModelSpec("qwen2vl_base", "Qwen2-VL-7B (Base)", "Qwen/Qwen2-VL-7B", "qwen", 28,
              (4, 8, 12, 16, 20, 24, 27), "qwen2vl", "Qwen/Qwen2-VL-7B-Instruct"),
]}

# Image-folder key -> label used in the paper's figures.
OCCUPATIONS: dict[str, str] = {
    "construction": "Construction",
    "firefighter": "Firefighter",
    "flight_attendant": "Pilot",
    "delivery_driver": "Delivery Driver",
    "chef": "Chef",
    "scientist": "Scientist",
    "florist": "Florist",
    "waiter": "Waiter",
    "librarian": "Librarian",
    "cleaning": "Maids/Cleaning",
    "nurse": "Nurse",
    "hairdresser": "Hairdresser",
    "babysitter": "Babysitter",
    "preschool_teacher": "Preschool Teacher",
    "makeup_artist": "Makeup Artist",
}

MAX_EDGE = 1024  # images are downscaled so that their longest edge is at most 1024 px


def list_images(folder: str | Path, n: int | None = None) -> list[Path]:
    """Image files of a folder in the order used by the layer sweeps (lexicographic by name)."""
    folder = Path(folder)
    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
    return files[:n] if n else files


def load_image(path: str | Path, max_edge: int = MAX_EDGE) -> Image.Image:
    """RGB image, downscaled (LANCZOS) so that its longest edge is at most ``max_edge``."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    return img


def load_vlm(key: str) -> "VLM":
    """Load the model ``MODELS[key]`` (bfloat16, ``device_map="auto"``)."""
    spec = MODELS[key]
    if spec.family == "qwen":
        return QwenVL(spec.hf_id, processor_id=spec.processor_id)
    return {"llava": LlavaNext, "internvl": InternVL}[spec.family](spec.hf_id)


# ---------------------------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------------------------

VISUAL_PROMPT = "Describe this image."


def _capture(store: dict[int, torch.Tensor], layer: int):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        store[layer] = (hidden if hidden.ndim == 3 else hidden.unsqueeze(0)).detach().cpu()
    return hook


class VLM:
    tokenizer = None
    text_model: torch.nn.Module  # language backbone (its ``.layers`` are hooked)
    image_token_id: int
    device: torch.device

    @property
    def n_layers(self) -> int:
        return len(self.text_model.layers)

    def _skip_ids(self) -> set[int]:
        return set(self.tokenizer.all_special_ids) | {self.image_token_id}

    def _run_with_hooks(self, forward, layers: list[int]) -> dict[int, torch.Tensor]:
        captured: dict[int, torch.Tensor] = {}
        hooks = [self.text_model.layers[layer].register_forward_hook(_capture(captured, layer))
                 for layer in layers]
        try:
            with torch.no_grad():
                forward()
        finally:
            for h in hooks:
                h.remove()
        return captured

    def _visual_inputs(self, image: Image.Image):
        """(forward callable, boolean mask of the image-token positions)."""
        raise NotImplementedError

    def visual_token_states(self, image: Image.Image, layers: list[int]) -> dict[int, torch.Tensor]:
        forward, mask = self._visual_inputs(image)
        captured = self._run_with_hooks(forward, layers)
        return {layer: h[0, mask[:h.shape[1]], :] for layer, h in captured.items()}

    def text_token_states(self, texts: list[str], layers: list[int]
                          ) -> tuple[dict[int, torch.Tensor], list[dict]]:
        skip = self._skip_ids()
        states: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        metadata: list[dict] = []
        for text in texts:
            input_ids = self.tokenizer(text, truncation=True, max_length=128,
                                       return_tensors="pt")["input_ids"].to(self.device)
            captured = self._run_with_hooks(lambda: self.text_model(input_ids=input_ids), layers)
            positions = [p for p, tid in enumerate(input_ids[0].tolist()) if tid not in skip][1:]
            for content_pos, p in enumerate(positions):
                tid = int(input_ids[0, p])
                metadata.append({"token_str": self.tokenizer.decode([tid]), "token_id": tid,
                                 "caption": text, "position": content_pos})
            for layer in layers:
                states[layer].append(captured[layer][0, positions, :])
        return {layer: torch.cat(states[layer]) for layer in layers}, metadata


class QwenVL(VLM):
    """Qwen2-VL-7B / Qwen2.5-VL-7B (Instruct or base)."""

    def __init__(self, model_id: str, processor_id: str | None = None, dtype=torch.bfloat16):
        from transformers import AutoProcessor

        if "Qwen2.5" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        else:
            from transformers import Qwen2VLForConditionalGeneration as ModelCls
        self.model = ModelCls.from_pretrained(model_id, torch_dtype=dtype, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(processor_id or model_id)
        self.tokenizer = self.processor.tokenizer
        self.device = next(self.model.parameters()).device
        self.image_token_id = self.model.config.image_token_id
        inner = self.model.model
        self.text_model = inner.language_model if hasattr(inner, "language_model") else inner

    def visual_grid(self, image: Image.Image) -> tuple[int, int]:
        """(rows, columns) of the visual-token grid, i.e. after the 2x2 patch merge."""
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, h, w = self.processor(text=[text], images=[image], return_tensors="pt")["image_grid_thw"][0].tolist()
        merge = self.processor.image_processor.merge_size
        return int(h // merge), int(w // merge)

    def _visual_inputs(self, image):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VISUAL_PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        return lambda: self.model(**inputs), (inputs["input_ids"][0] == self.image_token_id).cpu()


class LlavaNext(VLM):
    """LLaVA-v1.6-Mistral-7B (LLaVA-NeXT, AnyRes tiling)."""

    def __init__(self, model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf", dtype=torch.bfloat16):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto").eval()
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.device = next(self.model.parameters()).device
        config = self.model.config
        self.image_token_id = getattr(config, "image_token_index", getattr(config, "image_token_id", None))
        lm = self.model.language_model if hasattr(self.model, "language_model") else self.model.model.language_model
        self.text_model = lm.model if hasattr(lm, "model") and hasattr(lm.model, "layers") else lm

    def _visual_inputs(self, image):
        inputs = self.processor(text=f"[INST] <image>\n{VISUAL_PROMPT} [/INST]", images=[image],
                                return_tensors="pt").to(self.device)
        return lambda: self.model(**inputs), (inputs["input_ids"][0] == self.image_token_id).cpu()


class InternVL(VLM):
    """InternVL2.5-8B (remote code). Images are resized to one 448x448 tile (256 visual tokens)."""

    IMAGE_SIZE = 448
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_id: str = "OpenGVLab/InternVL2_5-8B", dtype=torch.bfloat16):
        import importlib

        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, device_map="auto",
                                               trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.device = next(self.model.parameters()).device
        self.dtype = dtype
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = self.image_token_id
        self.lm = self.model.language_model
        self.text_model = self.lm.model
        package = type(self.model).__module__.rsplit(".", 1)[0]
        self._get_conv_template = importlib.import_module(f"{package}.conversation").get_conv_template

    def _visual_inputs(self, image):
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])
        pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(self.device, dtype=self.dtype)
        template = self._get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], f"<image>\n{VISUAL_PROMPT}")
        template.append_message(template.roles[1], None)
        image_tokens = "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token + "</img>"
        enc = self.tokenizer(template.get_prompt().replace("<image>", image_tokens, 1), return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            embeds = self.lm.get_input_embeddings()(input_ids)
            flat = embeds.reshape(-1, embeds.shape[-1])
            flat[input_ids.reshape(-1) == self.image_token_id] = (
                self.model.extract_feature(pixel_values).reshape(-1, flat.shape[-1]).to(flat.device))
            embeds = flat.reshape(embeds.shape)
        attention_mask = enc["attention_mask"].to(self.device)
        return (lambda: self.lm(inputs_embeds=embeds, attention_mask=attention_mask),
                (input_ids[0] == self.image_token_id).cpu())
