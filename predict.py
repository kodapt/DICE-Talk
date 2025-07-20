# Prediction interface for Cog ⚙️
# https://cog.run/python

import subprocess
from typing import Optional
from cog import BasePredictor, Input, Path
import os
from PIL import Image
from pydub import AudioSegment
from dice_talk import DICE_Talk  # CHANGE THIS if your main class is elsewhere
import torch
import time

MODEL_CACHE = "checkpoints"

# Set environment variables for local caching
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

BASE_URL=  f"https://weights.replicate.delivery/default/dice-talk/{MODEL_CACHE}/"

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        
        model_files = [
            # TODO: List all required model files for DICE-Talk here
            "DICE-Talk.tar"
        ]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")

        # Initialize DICE-Talk pipeline
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
        """Generate a talking face video from a portrait image and audio file."""
        print("Starting prediction...")
        start_time = time.time()
        
        # Create temporary directories if they don't exist
        tmp_dir = "/src/tmp_path"
        res_dir = "/src/res_path"
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)
        
        tmp_image_path = os.path.join(tmp_dir, "input_image.png")
        tmp_audio_path = os.path.join(tmp_dir, "input_audio.wav")
        res_video_path = os.path.join(res_dir, "output.mp4")
        
        # Save input image as PNG
        img_in = Image.open(str(image))
        img_in.save(tmp_image_path, "PNG")
        print(f"Saved input image to: {tmp_image_path}")
        
        # Convert input audio to WAV format
        audio_segment = AudioSegment.from_file(str(audio))
        audio_segment.export(tmp_audio_path, format="wav")
        print(f"Converted and saved audio to: {tmp_audio_path}")
        
        # Preprocess and generate
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
