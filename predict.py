import subprocess
from typing import Optional
from cog import BasePredictor, Input, Path
import os
from PIL import Image
from pydub import AudioSegment
from dice_talk import DICE_Talk
import torch
import time
from huggingface_hub import hf_hub_download

MODEL_CACHE = "checkpoints"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

BASE_URL = f"https://weights.replicate.delivery/default/sonic/{MODEL_CACHE}/"
HF_REPO = "EEEELY/DICE-Talk"

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

def setup_checkpoints():
    # Download files from Replicate CDN
    cdn_model_files = [
        "models--openai--whisper-tiny.tar",
        "models--stabilityai--stable-video-diffusion-img2vid-xt.tar",
        "stable-video-diffusion-img2vid-xt.tar",
        "whisper-tiny.tar",
    ]
    for model_file in cdn_model_files:
        url = BASE_URL + model_file
        dest_path = os.path.join(MODEL_CACHE, model_file)
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        if not os.path.exists(dest_path.replace(".tar", "")):
            download_weights(url, dest_path)

    # Download custom weights from Hugging Face
    hf_files = [
        "DICE-Talk/unet.pth",
        "DICE-Talk/pose_guider.pth",
        "DICE-Talk/audio_linear.pth",
        "DICE-Talk/emo_model.pth",
        "yoloface_v5m.pt",
        "RIFE/RIFE_HDv3.pkl",
        "RIFE/IFNet_HDv3.py"
    ]
    hf_local_paths = {}
    for hf_file in hf_files:
        local_path = hf_hub_download(HF_REPO, hf_file, cache_dir=MODEL_CACHE)
        hf_local_paths[hf_file] = local_path
    return hf_local_paths

class Predictor(BasePredictor):
    def setup(self) -> None:
        # Download everything at setup
        self.hf_local_paths = setup_checkpoints()

        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")

        # You may need to pass checkpoint paths into DICE_Talk if required!
        self.pipe = DICE_Talk(
            device_id=device
        )
        print("DICE-TALK pipeline initialized.")

    def predict(
        self,
        image: Path = Input(description="Input portrait image (will be cropped if face is detected)."),
        audio: Path = Input(description="Input audio file (WAV, MP3, etc.) for the voice."),
        emotion: Optional[Path] = Input(description="Emotion annotation file.", default=None),
        ref_scale: float = Input(default=3.0),
        emo_scale: float = Input(default=6.0),
        min_resolution: int = Input(default=512),
        inference_steps: int = Input(default=25),
        keep_resolution: bool = Input(default=False),
        seed: Optional[int] = Input(default=None),
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
        audio_segment = AudioSegment.from_file(str(audio))
        audio_segment.export(tmp_audio_path, format="wav")
        
        expand_ratio = 0.5
        face_info = self.pipe.preprocess(tmp_image_path, expand_ratio=expand_ratio)
        processed_image_path = tmp_image_path
        
        if face_info and face_info.get('face_num', 0) > 0 and 'crop_bbox' in face_info:
            crop_image_path = os.path.join(tmp_dir, "face_crop.png")
            self.pipe.crop_image(tmp_image_path, crop_image_path, face_info['crop_bbox'])
            processed_image_path = crop_image_path
            print("Using cropped face image for processing")
        else:
            print("Using original image for processing (no face detected)")
        
        print("Generating talking face animation...")
        self.pipe.process(
            processed_image_path,
            tmp_audio_path,
            str(emotion) if emotion is not None else None,
            res_video_path,
            min_resolution=min_resolution,
            inference_steps=inference_steps,
            ref_scale=ref_scale,
            emo_scale=emo_scale,
            keep_resolution=keep_resolution,
            seed=seed,
        )
        print(f"Video generation complete")
        
        end_time = time.time()
        print(f"Total prediction time: {end_time - start_time:.2f} seconds")
        
        return Path(res_video_path)
