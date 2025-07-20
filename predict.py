from typing import Optional
from cog import BasePredictor, Input, Path
import os
from PIL import Image
from pydub import AudioSegment
from dice_talk import DICE_Talk  # or correct import
import torch
import time

MODEL_CACHE = "checkpoints"

os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

class Predictor(BasePredictor):
    def setup(self) -> None:
        # No need to download weights!
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = DICE_Talk(device_id=device)
        print("DICE-Talk pipeline initialized.")

    def predict(
        self,
        image: Path = Input(description="Input portrait image (will be cropped if face is detected)."),
        audio: Path = Input(description="Input audio file (WAV, MP3, etc.) for the voice."),
        min_resolution: int = Input(
            description="Minimum image resolution for processing. Lower values use less memory but may reduce quality.",
            default=512,
            ge=256,
            le=1024,
        ),
        inference_steps: int = Input(
            description="Number of diffusion steps. Higher values may improve quality but take longer.",
            default=25,
            ge=5,
            le=50,
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible results. Leave blank for a random seed.",
            default=None
        )
    ) -> Path:
        print("Starting prediction...")
        start_time = time.time()

        tmp_dir = "/src/tmp_path"
        res_dir = "/src/res_path"
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)

        tmp_image_path = os.path.join(tmp_dir, "input_image.png")
        tmp_audio_path = os.path.join(tmp_dir, "input_audio.wav")
        res_video_path = os.path.join(res_dir, "output.mp4")

        img_in = Image.open(str(image))
        img_in.save(tmp_image_path, "PNG")
        print(f"Saved input image to: {tmp_image_path}")

        audio_segment = AudioSegment.from_file(str(audio))
        audio_segment.export(tmp_audio_path, format="wav")
        print(f"Converted and saved audio to: {tmp_audio_path}")

        print("Generating talking face animation with DICE-Talk...")
        self.pipe.talking_head(
            image_path=tmp_image_path,
            audio_path=tmp_audio_path,
            output_path=res_video_path,
            min_resolution=min_resolution,
            inference_steps=inference_steps,
            seed=seed
        )
        print(f"Video generation complete")
        end_time = time.time()
        print(f"Total prediction time: {end_time - start_time:.2f} seconds")
        return Path(res_video_path)
