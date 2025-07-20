from typing import Optional
from cog import BasePredictor, Input, Path
import os
from PIL import Image
from pydub import AudioSegment
from dice_talk import DICE_Talk

class Predictor(BasePredictor):
    def setup(self):
        # device_id=0 for CUDA, or -1 for CPU fallback
        self.pipe = DICE_Talk(device_id=0)
        print("DICE-Talk model initialized.")

    def predict(
        self,
        image: Path = Input(description="Portrait image (PNG/JPG)"),
        audio: Path = Input(description="Audio file (WAV, MP3, etc)"),
        emotion: Optional[Path] = Input(description="(Optional) Emotion reference (video or image)", default=None),
        ref_scale: float = Input(description="Reference scale (default 3.0)", default=3.0),
        emo_scale: float = Input(description="Emotion scale (default 6.0)", default=6.0),
        crop: bool = Input(description="Crop detected face", default=False),
        seed: Optional[int] = Input(description="Random seed", default=None)
    ) -> Path:
        tmp_dir = "/src/tmp_dicetalk"
        os.makedirs(tmp_dir, exist_ok=True)
        img_path = os.path.join(tmp_dir, "input.png")
        audio_path = os.path.join(tmp_dir, "input.wav")
        output_path = os.path.join(tmp_dir, "result.mp4")
        emotion_path = None

        Image.open(str(image)).save(img_path)
        AudioSegment.from_file(str(audio)).export(audio_path, format="wav")
        if emotion:
            emotion_path = os.path.join(tmp_dir, "input_emotion.mp4")
            with open(emotion_path, "wb") as fout, open(str(emotion), "rb") as fin:
                fout.write(fin.read())

        # Preprocess and optionally crop
        face_info = self.pipe.preprocess(img_path, expand_ratio=0.5)
        print("Face info:", face_info)
        processed_image_path = img_path
        if crop and face_info and face_info.get('face_num', 0) > 0 and 'crop_bbox' in face_info:
            crop_image_path = img_path + '.crop.png'
            self.pipe.crop_image(img_path, crop_image_path, face_info['crop_bbox'])
            processed_image_path = crop_image_path
            print("Using cropped face image for processing.")
        else:
            print("Using original image for processing.")

        # Inference
        self.pipe.process(
            processed_image_path,
            audio_path,
            emotion_path,
            output_path,
            min_resolution=512,
            inference_steps=25,
            ref_scale=ref_scale,
            emo_scale=emo_scale,
            seed=seed
        )
        print("Video generation complete:", output_path)
        return Path(output_path)
