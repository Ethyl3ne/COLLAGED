import shutil
import os
from gradio_client import Client

def get_next_usage_folder():
    outputs_dir = "Outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    existing_folders = []
    for item in os.listdir(outputs_dir):
        if os.path.isdir(os.path.join(outputs_dir, item)) and item.startswith("Usage"):
            try:
                num = int(item.replace("Usage", ""))
                existing_folders.append(num)
            except ValueError:
                continue
    
    next_num = max(existing_folders) + 1 if existing_folders else 1
    return os.path.join(outputs_dir, f"Usage{next_num}")

class MusicGenerator:
    def __init__(self, session_folder=None):
        if session_folder is None:
            session_folder = get_next_usage_folder()
        
        self.session_folder = session_folder
        os.makedirs(self.session_folder, exist_ok=True)
        print(f"Music session folder created: {self.session_folder}")
    
    def generate_music(self, text_prompt, duration=30, output_subdir="music"):
        """Generate music and save to session folder"""
        output_dir = os.path.join(self.session_folder, output_subdir)
        return generate_music_from_prompt(
            text_prompt, duration, output_dir, self.session_folder
        )

def generate_music_from_prompt(text_prompt, duration=30, output_dir=None, session_folder=None):
    """
    Generate music using MelodyFlow based on a text prompt
    
    Args:
        text_prompt (str): Text description for music generation
        duration (int): Duration in seconds (default 30)
        output_dir (str): Directory to save generated music files
        session_folder (str): Session folder for organization
    
    Returns:
        list: Paths to generated music files, or empty list if failed
    """
    try:
        print(f"Generating music: '{text_prompt}'")
        print(f"Duration: {duration} seconds")
        
        if session_folder is None:
            session_folder = get_next_usage_folder()
            
        if output_dir is None:
            output_dir = os.path.join(session_folder, "music")
        elif not os.path.isabs(output_dir):
            output_dir = os.path.join(session_folder, output_dir)

        os.makedirs(output_dir, exist_ok=True)
        print(f"Using session folder: {session_folder}")
        
        client = Client("facebook/MelodyFlow")
        result = client.predict(
            model="facebook/melodyflow-t24-30secs",
            text=text_prompt,
            solver="midpoint",
            steps=128,
            target_flowstep=0,
            regularize=False,
            regularization_strength=0.2,
            duration=duration,
            melody=None,
            api_name="/predict"
        )
        
        saved_files = []
        for wav_path in result:
            target_path = os.path.join(output_dir, os.path.basename(wav_path))
            shutil.copy(wav_path, target_path)
            saved_files.append(target_path)
            print(f"Music saved to: {target_path}")
        
        return saved_files
        
    except Exception as e:
        print(f"Music generation failed: {e}")
        return []

if __name__ == "__main__":
    music_gen = MusicGenerator()
    music_gen.generate_music("A calm 8-bit melody suitable for a peaceful meadow")