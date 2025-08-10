from io import BytesIO
import IPython
import json
import os
from PIL import Image
import requests
import time
from dotenv import load_dotenv

from image_processing import create_pixel_art_from_image

class StableDiffusionGenerator:
    def __init__(self, session_folder=None):
        """Initialize the Stable Diffusion generator"""
        load_dotenv()
        
        if session_folder is None:
            session_folder = self.get_next_usage_folder()
        
        self.session_folder = os.path.abspath(session_folder)
        os.makedirs(self.session_folder, exist_ok=True)
        print(f"📁 Session folder created: {self.session_folder}")
        
        # Get API key
        api_key = os.getenv('STABILITY_API_KEY')
        
        if not api_key:
            raise ValueError("STABILITY_API_KEY not found in .env file.")
        
        self.api_key = api_key
        print(f"Stability API key found: {api_key}")

        self.default_params = {
            "aspect_ratio": "1:1",
            "seed": 0,
            "output_format": "png",
            "style_preset": "pixel-art"
        }

        self.pixel_processing_enabled = True
        self.pixel_size = (64, 64)
        self.n_colors = 16
        self.remove_background = False
            
    def get_next_usage_folder(self):
        """Get the next sequential usage folder (Usage1, Usage2, etc.)"""
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
        
        # Get next number
        next_num = max(existing_folders) + 1 if existing_folders else 1
        usage_folder = os.path.join(outputs_dir, f"Usage{next_num}")
        
        os.makedirs(usage_folder, exist_ok=True)
        
        return usage_folder
    
    def _ensure_session_path(self, path):
        """Ensure a path is within the session folder"""
        if path is None:
            return None
            
        if os.path.isabs(path):
            return path
            
        return os.path.join(self.session_folder, path)

    def send_generation_request(self, host, params):
        """Send a generation request to Stability AI API using multipart/form-data"""
        headers = {
            "Accept": "image/*",
            "Authorization": f"Bearer {self.api_key}"
        }

        files = {}
        for key, value in params.items():
            if value is not None:  
                files[key] = (None, str(value))

        print(f"Sending multipart request to {host}...")
        response = requests.post(
            host,
            headers=headers,
            files=files  
        )
        
        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        return response

    def generate_image_from_text(self, prompt, output_path=None, negative_prompt="", **kwargs):
        """
        Generate an image from text prompt
        
        Args:
            prompt (str): Text description for image generation
            output_path (str): Optional path to save the image (will be placed in session folder if relative)
            negative_prompt (str): Negative prompt to avoid certain elements
            **kwargs: Additional parameters (aspect_ratio, style_preset, etc.)
            
        Returns:
            dict: Result with success status and image data
        """
        try:
            print(f"🎨 Generating image with prompt: {prompt}")
            
            # Ensure output path is within session folder
            if output_path:
                final_output_path = self._ensure_session_path(output_path)
            else:
                # Create default path within session folder
                final_output_path = os.path.join(self.session_folder, "generated_image.png")
            
            print(f"💾 Will save to: {final_output_path}")
            
            # Set up parameters
            params = self.default_params.copy()
            params.update(kwargs)
            params.update({
                "prompt": prompt,
                "negative_prompt": negative_prompt
            })
            
            # Remove style_preset if set to None or "None"
            if params.get("style_preset") in [None, "None"]:
                params.pop("style_preset", None)
            
            # Remove empty negative prompt
            if not params.get("negative_prompt"):
                params.pop("negative_prompt", None)
            
            host = "https://api.stability.ai/v2beta/stable-image/generate/core"
            
            response = self.send_generation_request(host, params)
            
            # Decode response
            output_image = response.content
            finish_reason = response.headers.get("finish-reason")
            seed = response.headers.get("seed")
            
            # Check for NSFW classification
            if finish_reason == 'CONTENT_FILTERED':
                return {
                    "success": False,
                    "error": "Generation failed NSFW classifier"
                }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
            
            # Save image
            # Save raw image first
            raw_image_path = final_output_path.replace('.png', '_raw.png')
            with open(raw_image_path, "wb") as f:
                f.write(output_image)

            # Load image for processing
            image = Image.open(raw_image_path)

            # Apply pixel art processing if enabled
            if self.pixel_processing_enabled:
                print(f"🎨 Applying pixel art processing...")
                print(f"   Size: {self.pixel_size}, Colors: {self.n_colors}, Remove BG: {self.remove_background}")
                
                processed_image = create_pixel_art_from_image(
                    image,
                    pixel_size=self.pixel_size,
                    n_colors=self.n_colors,
                    remove_bg=self.remove_background,
                    bg_threshold=240
                )
                
                # Save processed image
                processed_image.save(final_output_path)
                image = processed_image
                print(f"✅ Pixel art processing complete")
                
                # Clean up raw image if processing was successful
                try:
                    os.remove(raw_image_path)
                except:
                    pass
            else:
                # Just save the raw image
                with open(final_output_path, "wb") as f:
                    f.write(output_image)
                image = Image.open(final_output_path)
            
            print(f"✅ Image saved: {final_output_path}")
            
            return {
                "success": True,
                "image": image,
                "saved_path": final_output_path,
                "session_folder": self.session_folder,
                "seed": seed,
                "finish_reason": finish_reason,
                "message": "Image generated successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Generation failed: {str(e)}"
            }

    def generate_image_with_reference(self, text_prompt, reference_image_path, output_path=None, **kwargs):
        """
        Generate an image based on text prompt (reference image ignored for Stable Diffusion)
        
        Args:
            text_prompt (str): Text description for image generation
            reference_image_path (str): Path to reference image (ignored, for compatibility)
            output_path (str): Optional path to save the generated image
            **kwargs: Additional parameters
            
        Returns:
            dict: Result with success status and image data
        """
        print("⚠️ Stable Diffusion generator only supports text prompts. Ignoring reference image.")
        print("📝 Generating from text prompt only...")
        
        return self.generate_image_from_text(text_prompt, output_path, **kwargs)

    def set_style_preset(self, style_preset):
        """Set the default style preset"""
        self.default_params["style_preset"] = style_preset
        print(f"Style preset set to: {style_preset}")

    def set_aspect_ratio(self, aspect_ratio):
        """Set the default aspect ratio"""
        valid_ratios = ["21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"]
        if aspect_ratio in valid_ratios:
            self.default_params["aspect_ratio"] = aspect_ratio
            print(f"Aspect ratio set to: {aspect_ratio}")
        else:
            print(f"Invalid aspect ratio. Valid options: {', '.join(valid_ratios)}")
    def set_pixel_processing(self, enabled=True, pixel_size=(64, 64), n_colors=16, remove_background=False):
        """Set pixel art processing parameters"""
        self.pixel_processing_enabled = enabled
        self.pixel_size = pixel_size
        self.n_colors = n_colors
        self.remove_background = remove_background
        print(f"Pixel processing: {'enabled' if enabled else 'disabled'}")
        if enabled:
            print(f"   Size: {pixel_size}, Colors: {n_colors}, Remove BG: {remove_background}")
def main():
    """Main function to demonstrate the Stable Diffusion generator"""
    
    print("="*60)
    print("STABLE DIFFUSION IMAGE GENERATOR")
    print("="*60)
    
    try:
        # Initialize the generator
        generator = StableDiffusionGenerator()
    except ValueError as e:
        print(f"\nSetup Error: {e}")
        return
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return
    
    print(f"\n🎨 STABLE DIFFUSION IMAGE GENERATOR")
    print("="*60)
    print(f"📁 Session folder: {generator.session_folder}")
    print("🔧 Required packages: pip install requests pillow python-dotenv")
    print("🎯 Default settings: pixel-art style, 1:1 aspect ratio")
    print("📝 TEXT-ONLY: This generator only accepts text prompts (no reference images)")
    print("\nGeneration Options:")
    print("  1. Single Image Generation")
    print("  2. Settings")
    print("\nCommands:")
    print("  - 'single' or '1' for single image generation")
    print("  - 'settings' or '2' to adjust generation settings")
    print("  - 'quit' to exit")
    print("="*60)
    
    # Main loop
    while True:
        print("\n" + "="*50)
        print("Choose generation type:")
        print("1. Single Image")
        print("2. Settings")
        print("3. Quit")
        print("="*50)
        
        choice = input("Enter your choice (1-3): ").strip().lower()
        
        if choice in ['quit', '3', 'q']:
            break
        
        elif choice in ['settings', '2']:
            print(f"\nCurrent settings:")
            print(f"  - Style preset: {generator.default_params.get('style_preset', 'None')}")
            print(f"  - Aspect ratio: {generator.default_params.get('aspect_ratio', '1:1')}")
            print(f"  - Output format: {generator.default_params.get('output_format', 'png')}")
            
            print("\nAdjust settings:")
            
            # Style preset
            print("\nAvailable style presets:")
            presets = ["pixel-art", "anime", "photographic", "digital-art", "comic-book", "fantasy-art", "line-art", "analog-film", "neon-punk", "isometric", "low-poly", "origami", "watercolor", "cinematic", "None"]
            for i, preset in enumerate(presets, 1):
                print(f"  {i}. {preset}")
            
            preset_choice = input(f"Choose style preset (1-{len(presets)}) or press Enter to keep current: ").strip()
            if preset_choice and preset_choice.isdigit():
                idx = int(preset_choice) - 1
                if 0 <= idx < len(presets):
                    generator.set_style_preset(presets[idx] if presets[idx] != "None" else None)
            
            # Aspect ratio
            print("\nAvailable aspect ratios:")
            ratios = ["21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"]
            for i, ratio in enumerate(ratios, 1):
                print(f"  {i}. {ratio}")
            
            ratio_choice = input(f"Choose aspect ratio (1-{len(ratios)}) or press Enter to keep current: ").strip()
            if ratio_choice and ratio_choice.isdigit():
                idx = int(ratio_choice) - 1
                if 0 <= idx < len(ratios):
                    generator.set_aspect_ratio(ratios[idx])
            
            print(f"\nSettings updated!")
            continue
        
        elif choice in ['single', '1']:
            # Single image generation
            print("\nSINGLE IMAGE GENERATION")
            print(f"Images will be saved to: {generator.session_folder}")
            print("Commands:")
            print("  - Just type a description to generate an image")
            print("  - 'save:filename.png your prompt' to specify save name")
            print("  - 'back' to return to main menu")
            
            while True:
                user_input = input("\nImage prompt: ").strip()
                
                if user_input.lower() == 'back':
                    break
                
                if not user_input:
                    continue
                
                # Parse save command
                output_filename = None
                prompt = user_input
                
                if 'save:' in user_input:
                    try:
                        save_start = user_input.find('save:') + 5
                        save_end = user_input.find(' ', save_start)
                        if save_end == -1:
                            print("❌ Please provide a prompt after the save filename")
                            continue
                        
                        output_filename = user_input[save_start:save_end].strip()
                        prompt = user_input[save_end:].strip()
                        
                        print(f"Will save as: {output_filename}")
                    except Exception as e:
                        print(f"Error parsing save command: {e}")
                        continue
                
                print(f"\nGenerating: {prompt}")
                
                # Generate image
                result = generator.generate_image_from_text(prompt, output_filename)
                
                if result["success"]:
                    print(f"{result['message']}")
                    if result.get('saved_path'):
                        print(f"📁 Saved to: {result['saved_path']}")
                    if result.get('seed'):
                        print(f"Seed: {result['seed']}")
                    
                    # Try to display the image
                    try:
                        if result.get('image'):
                            result['image'].show()
                    except Exception as e:
                        print(f"Could not display image: {e}")
                else:
                    print(f"Generation failed: {result['error']}")
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()