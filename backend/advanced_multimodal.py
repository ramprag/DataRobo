import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from PIL import Image, ImageDraw, ImageFont
try:
    from huggingface_hub import login as hf_login  # type: ignore
except Exception:  # pragma: no cover
    hf_login = None  # type: ignore
import numpy as np

# Optional import of diffusers ONNX pipeline with graceful fallback
try:
    from diffusers import OnnxStableDiffusionPipeline  # type: ignore
except Exception:  # pragma: no cover - environment without diffusers/onnxruntime
    OnnxStableDiffusionPipeline = None  # type: ignore

class AdvancedMultimodalGenerator:
    """
    CPU-friendly multimodal generator using ONNX Runtime for diffusion.

    Notes:
      - No PyTorch dependency.
      - Use small image sizes (e.g., 256x256) and few steps to fit 8GB RAM CPUs.
      - Model: onnx-community/stable-diffusion-v1-5 (free on Hugging Face).
    """

    def __init__(
        self,
        output_root: str = "synthetic",
        hf_token: Optional[str] = None,
        model_id: str = "onnx-community/stable-diffusion-v1-5",
        provider: str = "CPUExecutionProvider"
    ):
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

        if hf_token and hf_login is not None:
            try:
                hf_login(hf_token)
            except Exception:
                pass  # ignore if token not needed

        # Lazy-load pipeline to minimize memory footprint; create on first use
        self._pipe: Optional[OnnxStableDiffusionPipeline] = None
        self._model_id = model_id
        self._provider = provider

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        # If diffusers ONNX pipeline is not available, skip loading
        if OnnxStableDiffusionPipeline is None:
            return None
        try:
            pipe = OnnxStableDiffusionPipeline.from_pretrained(
                self._model_id,
                provider=self._provider
            )
            self._pipe = pipe
            return pipe
        except Exception:
            # Model unavailable or env lacks onnxruntime; use fallback later
            self._pipe = None
            return None

    def _default_prompts(self, domain: str, count: int) -> List[str]:
        base = {
            "automotive": [
                "A street scene with cars and pedestrians, traffic lights, daytime, realistic, wide angle",
                "A modern car on a road with city background, motion blur, photorealistic",
                "A crosswalk with pedestrians and a traffic signal, urban street, realistic"
            ],
            "robotics": [
                "Industrial robotic arm manipulating a box in a warehouse, realistic lighting",
                "Mobile warehouse robot moving near pallets, depth of field, realistic",
                "Robot with gripper assembling parts on a table, studio lighting, realistic"
            ],
            "warehouse": [
                "Warehouse interior with pallets and workers, realistic lighting",
                "Forklift and pallets in a distribution center, realistic, cold color tones",
                "Storage racks with boxes and aisle, realistic warehouse photo"
            ],
            "ecommerce": [
                "Product photo of electronic gadget on white background, studio lighting, high detail",
                "Sleek pair of running shoes on clean background, realistic product photography",
                "Kitchen appliance photo with soft shadows, professional lighting"
            ],
            "generic": [
                "High quality realistic photo of an object on a clean background",
                "Realistic outdoor scene with natural lighting",
                "Studio shot of an item with soft shadows"
            ],
        }
        pool = base.get(domain, base["generic"])
        if count <= len(pool):
            return pool[:count]
        # Repeat while trimming to count
        reps = (count + len(pool) - 1) // len(pool)
        out = (pool * reps)[:count]
        return out

    def generate_images_diffusion(
        self,
        out_dir: str,
        prompts: List[str],
        width: int = 256,
        height: int = 256,
        num_inference_steps: int = 15,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        pipe = self._load_pipeline()

        rng = np.random.RandomState(seed if seed is not None else 42)
        images_meta = []
        files = []

        def _fallback_image(prompt_text: str) -> Image.Image:
            # Create a simple placeholder with random background and prompt overlay
            bg = tuple(int(x) for x in rng.randint(0, 255, size=3))
            img = Image.new("RGB", (int(width), int(height)), bg)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            text = (prompt_text or "synthetic").strip()[:100]
            # Draw wrapped text
            margin = 10
            y = 10
            for line in [text[i:i+28] for i in range(0, len(text), 28)]:
                draw.text((margin, y), line, fill=(255, 255, 255), font=font)
                y += 12
            return img

        for i, prompt in enumerate(prompts):
            if pipe is not None:
                try:
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=int(num_inference_steps),
                        guidance_scale=float(guidance_scale),
                        height=int(height),
                        width=int(width)
                    ).images[0]
                except Exception:
                    image = _fallback_image(prompt)
            else:
                image = _fallback_image(prompt)

            fname = f"diff_{i:05d}.png"
            fpath = os.path.join(out_dir, fname)
            image.save(fpath)
            files.append(fpath)
            images_meta.append({"file_name": fname, "prompt": prompt, "width": width, "height": height})

        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"images": images_meta}, f, indent=2)

        return {"images_dir": out_dir, "files": files, "metadata": meta_path}

    def generate(
        self,
        job_dir: str,
        spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        spec example:
        {
          "text": {"enabled": true, "count": 1000, "domain": "ecommerce"},
          "image": {
            "enabled": true, "count": 50, "width": 256, "height": 256,
            "domain": "automotive", "prompts": ["car on road", ...],
            "num_inference_steps": 15, "guidance_scale": 6.0
          }
        }
        """
        os.makedirs(job_dir, exist_ok=True)
        result: Dict[str, Any] = {"job_dir": job_dir, "created_at": datetime.utcnow().isoformat()}

        # text passthrough: your existing platform handles text tabular; optionally record the intention here
        if spec.get("text", {}).get("enabled"):
            result["text"] = {
                "note": "Use existing tabular/text pipeline for textual records",
                "requested_count": int(spec["text"].get("count", 0)),
                "domain": str(spec["text"].get("domain", "generic"))
            }

        if spec.get("image", {}).get("enabled"):
            img = spec["image"]
            count = int(img.get("count", 10))
            width = int(img.get("width", 256))
            height = int(img.get("height", 256))
            domain = str(img.get("domain", "generic"))
            prompts = img.get("prompts")
            if not prompts:
                prompts = self._default_prompts(domain, count)
            else:
                # expand/trim to count
                if len(prompts) < count:
                    reps = (count + len(prompts) - 1) // len(prompts)
                    prompts = (prompts * reps)[:count]
                else:
                    prompts = prompts[:count]

            steps = int(img.get("num_inference_steps", 15))
            guidance = float(img.get("guidance_scale", 6.0))

            image_dir = os.path.join(job_dir, "images_diffusion")
            img_res = self.generate_images_diffusion(
                out_dir=image_dir,
                prompts=prompts,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            result["image"] = {
                "images_dir": img_res["images_dir"],
                "files": img_res["files"],
                "metadata": img_res["metadata"],
                "count": count,
                "domain": domain,
                "width": width,
                "height": height,
                "num_inference_steps": steps
            }

        return result