
import os
import json
import base64
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import traceback

try:
    from gemini_generation import GeminiImageGenerator
    GEMINI_GENERATION_AVAILABLE = True
except ImportError:
    print("Gemini image generation not available. Install required packages for gemini_generation.py")
    GEMINI_GENERATION_AVAILABLE = False

try:
    from stable_diffusion import StableDiffusionGenerator
    STABLE_DIFFUSION_AVAILABLE = True
except ImportError:
    print("Stable Diffusion generation not available. Install required packages for stable_diffusion.py")
    STABLE_DIFFUSION_AVAILABLE = False

IMAGE_GENERATION_AVAILABLE = GEMINI_GENERATION_AVAILABLE or STABLE_DIFFUSION_AVAILABLE
try:
    from music import generate_music_from_prompt
    MUSIC_GENERATION_AVAILABLE = True
except ImportError as e:
    print(f"Music generation not available. Install gradio_client to enable music generation.")
    print(f"Import error details: {e}")
    MUSIC_GENERATION_AVAILABLE = False
except Exception as e:
    print(f"Unexpected error importing music module: {e}")
    MUSIC_GENERATION_AVAILABLE = False

class GameAssetPromptGenerator:
    def __init__(self):
        load_dotenv()
        current_dir = os.getcwd()
        env_path = os.path.join(current_dir, '.env')
        print(f"Looking for .env file at: {env_path}")
        print(f".env file exists: {os.path.exists(env_path)}")
        
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            print(f"Gemini API key found: {api_key}")
        else:
            print("GEMINI_API_KEY not found in environment variables")
            print("Available environment variables starting with 'GEMINI':")
            for key in os.environ:
                if key.startswith('GEMINI'):
                    print(f"  - {key}")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')  
        self.gemini_generator = None
        self.stable_diffusion_generator = None
        
        if GEMINI_GENERATION_AVAILABLE:
            try:
                self.gemini_generator = GeminiImageGenerator()
                print("Gemini image generator initialized")
            except Exception as e:
                print(f"Gemini generator initialization failed: {e}")
                self.gemini_generator = None
        
        if STABLE_DIFFUSION_AVAILABLE:
            try:
                self.stable_diffusion_generator = StableDiffusionGenerator()
                print("Stable Diffusion generator initialized")
            except Exception as e:
                print(f"Stable Diffusion initialization failed: {e}")
                self.stable_diffusion_generator = None
        
        
        self.asset_types = {
            1: "sprite",
            2: "environment", 
            3: "misc_asset",
            4: "music",
            5: "animation"
        }
        
        self.asset_names = {
            1: "Character/Sprite",
            2: "Environment/Background",
            3: "Miscellaneous Asset (items, weapons, UI)",
            4: "Music/Audio",
            5: "Animation"
        }

    def get_system_prompt(self, selected_assets):
        """Generate system prompt based on selected asset types"""
        
        base_prompt = """
You are an expert game asset designer and prompt engineer. Your role is to analyze user input (text and/or images) and generate detailed, style-coherent prompts for creating game assets.

The user has specifically requested the following asset types: """ + ", ".join(selected_assets) + """

Your response must be in JSON format with this structure:

```json
{
    "analysis": "Brief analysis of the input style, theme, and what the user is requesting",
    "generated_assets": """ + str(selected_assets) + """,
"""
        sections = []
        
        if "sprite" in selected_assets:
            sections.append('''    "sprite": {
        "prompt": "Detailed prompt for generating one single pixel art character/sprite with a blank background",
        "technical_specs": "16-bit style, specify the direction it's facing (default to staring at screen), full body",
        "style_notes": "Additional style guidance specific to sprites"
    }''')
        
        if "environment" in selected_assets:
            sections.append('''    "environment": {
        "prompt": "Detailed prompt for generating a pixel art environment/background",
        "technical_specs": "16-bit style, parallax-ready layers, specify having either paths or open space for people to move about",
        "style_notes": "Additional style guidance specific to environments"
    }''')
        
        if "misc_asset" in selected_assets:
            sections.append('''    "misc_asset": {
        "asset_type": "weapon/item/tile/ui_element/prop",
        "prompt": "Detailed prompt for generating the miscellaneous pixel art asset",
        "technical_specs": "16-bit style, appropriate resolution for asset type",
        "style_notes": "Additional style guidance specific to this asset type"
    }''')
        
        if "music" in selected_assets:
            sections.append('''    "music": {
        "prompt": "Detailed prompt for generating background music or audio",
        "genre": "chiptune/orchestral/ambient/electronic",
        "mood": "energetic/calm/mysterious/epic",
        "technical_specs": "BPM, key signature, instrumentation suggestions",
        "style_notes": "Additional audio style guidance"
    }''')
        
        if "animation" in selected_assets:
            sections.append('''    "animation": {
                "frames": "The integer number of animation frames",
                "prompts": ["Detailed prompt for generating sprite animation sequences for first frame", "prompt for second frame"],
                "animation_type": "idle/walk/attack/jump/death",
                "technical_specs": "Frame count, timing, resolution per frame",
                "style_notes": "Animation style and timing guidance"

                - For animations, specify sprite direction if necessary. For each frame, describe the motions and exact actions taking place. For instance, a jumping animation can be broken into preparing, mid-air, and landing
                - Try to keep the image focused on the same sprite/item/environment and maintain similar aspect ratios
            }''')
        
        
        sections.append('''    "tone_mood": ["mood1", "mood2", "mood3"],
    "color_palette": ["color1", "color2", "color3"],
    "theme": "Overall theme that ties all assets together"''')
        
        full_prompt = base_prompt + ",\n".join(sections) + """
}
```

**Important Guidelines:**
- Only generate content for the asset types the user has requested
- Focus on pixel art aesthetic for visual assets
- Ensure style coherence between all generated assets
- Use clear, descriptive language for AI generation tools
- Include specific technical requirements
- Maintain thematic consistency across all asset types
- If conflicts arise, prioritize user text input over audio or image input
- Provide detailed, actionable prompts that AI tools can execute effectively
"""
        
        return full_prompt

    def display_asset_menu(self):
        """Display the asset selection menu"""
        print("\n" + "="*50)
        print("SELECT ASSET TYPES TO GENERATE")
        print("="*50)
        print("Enter the numbers for the assets you want to generate:")
        print()
        for num, name in self.asset_names.items():
            print(f"  {num}. {name}")
        print()
        print("Examples:")
        print("  - Enter '1' for character only")
        print("  - Enter '1,2' for character and environment")
        print("  - Enter '1,2,3,4' for character, environment, item, and music")
        print("  - Enter '5' for animation only")
        print("="*50)

    def get_image_generation_backend(self, selected_assets):
        """Get user's choice for image generation backend"""
        
        visual_assets = [asset for asset in selected_assets if asset in ['sprite', 'environment', 'misc_asset', 'animation']]
        
        if not visual_assets:
            print("\nNo visual assets selected - skipping image generation backend selection")
            return None
        
        
        if 'animation' in selected_assets:
            if GEMINI_GENERATION_AVAILABLE:
                print("\nAnimation selected: automatically using Gemini 2.0 flash preview image generation")
                print("   (Animation generation is only supported by Gemini)")
                return 'gemini'
            else:
                print("\nAnimation selected but Gemini not available!")
                print("Animation generation requires Gemini 2.0 Flash")
                return None
        
        
        available_backends = []
        if GEMINI_GENERATION_AVAILABLE:
            available_backends.append(('gemini', 'Gemini 2.0 Flash', 'Fast generation, good quality, built-in pixel art processing'))
        if STABLE_DIFFUSION_AVAILABLE:
            available_backends.append(('stable_diffusion', 'stable_diffusion', 'High quality images, more artistic control, text-only prompts'))
        
        if not available_backends:
            print("\nNo image generation backends available!")
            return None
        
        if len(available_backends) == 1:
            backend_id, backend_name, _ = available_backends[0]
            print(f"\nAutomatically using {backend_name} (only available backend)")
            return backend_id
        
        print("\n" + "="*50)
        print("SELECT IMAGE GENERATION BACKEND")
        print("="*50)
        print("Choose which AI model to use for generating images:")
        
        for i, (backend_id, backend_name, description) in enumerate(available_backends, 1):
            print(f"  {i}. {backend_name}")
            print(f"     - {description}")
            print()
        
        print("="*50)
        
        while True:
            try:
                choice = input(f"Enter your choice (1-{len(available_backends)}): ").strip()
                
                if choice.isdigit():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(available_backends):
                        backend_id, backend_name, _ = available_backends[choice_idx]
                        print(f"Selected: {backend_name}")
                        return backend_id
                    else:
                        print(f" - Please enter a number between 1 and {len(available_backends)}")
                else:
                    print(f" - Please enter a number between 1 and {len(available_backends)}")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye, see you again! 👋")
                return None
        backend_id, backend_name, _ = available_backends[choice_idx]
        print(f"Selected: {backend_name}")
        return backend_id  

    def get_user_asset_selection(self):
        """Get user's asset type selection"""
        while True:
            try:
                user_input = input("Enter asset numbers (separated by commas): ").strip()
                
                if not user_input:
                    print("Please enter at least one number.")
                    continue
                
                
                selected_numbers = []
                for num_str in user_input.split(','):
                    num_str = num_str.strip()
                    if num_str.isdigit():
                        num = int(num_str)
                        if num in self.asset_types:
                            selected_numbers.append(num)
                        else:
                            print(f"Invalid number: {num}. Please use numbers 1-5.")
                            break
                    else:
                        print(f"Invalid input: '{num_str}'. Please enter numbers only.")
                        break
                else:
                    
                    if selected_numbers:
                        
                        if 5 in selected_numbers and len(selected_numbers) > 1:
                            print("Animation (5) can only be selected by itself, not with other asset types.")
                            print("   Please select either just '5' for animation, or other numbers without '5'.")
                            continue
                        
                        selected_assets = [self.asset_types[num] for num in selected_numbers]
                        selected_names = [self.asset_names[num] for num in selected_numbers]
                        
                        print(f"\nSelected: {', '.join(selected_names)}")
                        return selected_assets
                    else:
                        print("Please enter at least one valid number.")
                        continue
                        
            except KeyboardInterrupt:
                print("\n\nGoodbye, see you again! 👋")
                return None
            except Exception as e:
                print(f"Error: {e}")
                continue

    def get_generation_settings(self, selected_assets, backend=None):
        """Get image generation settings from user with per-asset customization"""
        settings = {'backend': backend}
        
        
        visual_assets = [asset for asset in selected_assets if asset in ['sprite', 'environment', 'misc_asset', 'animation']]
        
        if not visual_assets or not backend:
            return settings
        
        if not IMAGE_GENERATION_AVAILABLE and backend == 'gemini':
            print("Gemini image generation not available - settings will be saved but not used")
            return settings
        
        
        settings['per_asset'] = {}
        
        if backend == 'stable_diffusion':
            print("\nSTABLE DIFFUSION SETTINGS:")
            
            
            print("\nGLOBAL SETTINGS (applied to all assets):")
            
            
            print("Style presets:")
            presets = ["pixel-art", "anime", "photographic", "digital-art", "comic-book", 
                    "fantasy-art", "line-art", "analog-film", "neon-punk", "isometric", 
                    "low-poly", "origami", "watercolor", "cinematic", "None"]
            for i, preset in enumerate(presets, 1):
                print(f"  {i}. {preset}")
            
            preset_choice = input(f"Choose style preset (1-{len(presets)}) [default: pixel-art]: ").strip()
            if preset_choice and preset_choice.isdigit():
                idx = int(preset_choice) - 1
                if 0 <= idx < len(presets):
                    settings['style_preset'] = presets[idx] if presets[idx] != "None" else None
            else:
                settings['style_preset'] = "pixel-art"
            
            
            negative_prompt = input("Global negative prompt (things to avoid) [optional]: ").strip()
            if negative_prompt:
                settings['negative_prompt'] = negative_prompt
            
            
            print("\nPIXEL ART PROCESSING:")
            enable_pixel = input("Enable pixel art processing for all assets? (y/n) [default: y]: ").strip().lower()
            settings['pixelate_images'] = enable_pixel not in ['n', 'no', 'false']
            
            if settings['pixelate_images']:
                
                colors_input = input("Number of colors for all assets [default: 16]: ").strip()
                if colors_input:
                    try:
                        settings['n_colors'] = int(colors_input)
                    except:
                        print("Invalid number, using default 16")
                        settings['n_colors'] = 16
                else:
                    settings['n_colors'] = 16
            
            
            print("\nPER-ASSET SETTINGS:")
            ratios = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"]
            
            for asset in visual_assets:
                if asset == 'animation':
                    continue  
                    
                asset_name = {
                    'sprite': 'Sprite',
                    'environment': 'Environment', 
                    'misc_asset': 'Misc Asset'
                }.get(asset, asset.title())
                
                print(f"\n{asset_name.upper()} SETTINGS:")
                
                default_ratio = "1:1" if asset in ('sprite', 'misc_asset') else "16:9"

                ratio_choice = input(f"Aspect ratio (1-{len(ratios)}) [default: {default_ratio}]: ").strip()
                if ratio_choice and ratio_choice.isdigit():
                    idx = int(ratio_choice) - 1
                    if 0 <= idx < len(ratios):
                        selected_ratio = ratios[idx]
                    else:
                        selected_ratio = default_ratio
                else:
                    selected_ratio = default_ratio
                
                
                if 'per_asset' not in settings:
                    settings['per_asset'] = {}
                settings['per_asset'][asset] = {'aspect_ratio': selected_ratio}
                
                
                if settings['pixelate_images']:
                    
                    if asset == 'sprite':
                        default_size = "32,32"
                        size_desc = "character sprites"
                    elif asset == 'environment':
                        default_size = "512,288"
                        size_desc = "backgrounds/environments"
                    else:  
                        default_size = "64,64"
                        size_desc = "items/props"
                    
                    size_input = input(f"Pixel size for {asset_name.lower()} (width,height) [default: {default_size} - good for {size_desc}]: ").strip()
                    if size_input:
                        try:
                            width, height = map(int, size_input.split(','))
                            settings['per_asset'][asset]['pixel_size'] = (width, height)
                        except:
                            print(f"Invalid format, using default {default_size}")
                            width, height = map(int, default_size.split(','))
                            settings['per_asset'][asset]['pixel_size'] = (width, height)
                    else:
                        width, height = map(int, default_size.split(','))
                        settings['per_asset'][asset]['pixel_size'] = (width, height)
                    
                    
                    default_remove_bg = "n"
                    settings['per_asset'][asset]['remove_background'] = default_remove_bg == 'y'
                    
                    print(f"{asset_name}: {selected_ratio}, {settings['per_asset'][asset]['pixel_size']}, BG removal: {settings['per_asset'][asset]['remove_background']}")
            
            return settings
        
        elif backend == 'gemini':
            print("\nGEMINI 2.0 FLASH SETTINGS:")
            
            
            print("\nGLOBAL PIXEL ART SETTINGS:")
            pixelate = input("Enable pixel art processing? (y/n) [default: y]: ").strip().lower()
            settings['pixelate_images'] = pixelate not in ['n', 'no', 'false']
            
            if settings['pixelate_images']:
                
                colors_input = input("Number of colors for all assets [default: 16]: ").strip()
                if colors_input:
                    try:
                        settings['n_colors'] = int(colors_input)
                    except:
                        print("Invalid number, using default 16")
                        settings['n_colors'] = 16
                else:
                    settings['n_colors'] = 16
                
                
                print("\nPER-ASSET PIXEL SIZES:")
                
                for asset in visual_assets:
                    if asset == 'animation':
                        continue  
                        
                    asset_name = {
                        'sprite': 'Sprite',
                        'environment': 'Environment', 
                        'misc_asset': 'Misc Asset'
                    }.get(asset, asset.title())
                    
                    print(f"\n{asset_name.upper()} PIXEL SIZE:")
                    
                    
                    if asset == 'sprite':
                        default_size = "32,32"
                        size_desc = "character sprites"
                    elif asset == 'environment':
                        default_size = "512,288"
                        size_desc = "backgrounds/environments"
                    else:  
                        default_size = "64,64"
                        size_desc = "items/props"
                    
                    size_input = input(f"Pixel size for {asset_name.lower()} (width,height) [default: {default_size} - good for {size_desc}]: ").strip()
                    if size_input:
                        try:
                            width, height = map(int, size_input.split(','))
                            if 'per_asset' not in settings:
                                settings['per_asset'] = {}
                            settings['per_asset'][asset] = {'pixel_size': (width, height)}
                            print(f"{asset_name}: {width}x{height} pixels")
                        except:
                            print(f"Invalid format, using default {default_size}")
                            width, height = map(int, default_size.split(','))
                            if 'per_asset' not in settings:
                                settings['per_asset'] = {}
                            settings['per_asset'][asset] = {'pixel_size': (width, height)}
                            print(f"{asset_name}: {width}x{height} pixels (default)")
                    else:
                        width, height = map(int, default_size.split(','))
                        if 'per_asset' not in settings:
                            settings['per_asset'] = {}
                        settings['per_asset'][asset] = {'pixel_size': (width, height)}
                        print(f"{asset_name}: {width}x{height} pixels (default)")
            else:
                
                for asset in visual_assets:
                    if asset != 'animation':
                        if 'per_asset' not in settings:
                            settings['per_asset'] = {}
                        settings['per_asset'][asset] = {}
        
        
        if "animation" in selected_assets:
            print("\nANIMATION SETTINGS:")
            
            
            while True:
                frames_input = input("Number of animation frames (2-30) [default: 8]: ").strip()
                if not frames_input:
                    settings['num_frames'] = 8
                    break
                try:
                    num_frames = int(frames_input)
                    if 2 <= num_frames <= 30:
                        settings['num_frames'] = num_frames
                        break
                    else:
                        print("Please enter a number between 2 and 30")
                except ValueError:
                    print("Please enter a valid number")
            
            
            duration_input = input("Frame duration in milliseconds [default: 500]: ").strip()
            if duration_input:
                try:
                    settings['frame_duration'] = int(duration_input)
                except:
                    print("Invalid duration, using default 500ms")
                    settings['frame_duration'] = 500
            else:
                settings['frame_duration'] = 500
            
            
            cleanup = input("Delete individual frames after creating GIF? (y/n) [default: n]: ").strip().lower()
            settings['cleanup_frames'] = cleanup in ['y', 'yes', 'true']
        
        
        print("\nOUTPUT SETTINGS:")
        output_dir = input("Output directory [default: current directory]: ").strip()
        if output_dir and os.path.exists(output_dir):
            settings['output_directory'] = output_dir
        else:
            settings['output_directory'] = "."
        
        
        print("\nREFERENCE IMAGE (used by Gemini 2.0/Stable Diff.):")
        if "animation" in selected_assets:
            print("This will be used for style consistency across all animation frames.")
        base_image = input("Reference image path (optional): ").strip()
        if base_image and os.path.exists(base_image):
            settings['reference_image'] = base_image
            print(f"Using reference image: {base_image}")
        else:
            settings['reference_image'] = None
        
        return settings



    def get_media_inputs(self):
        """Get media file inputs from user"""
        print("\n" + "="*40)
        print("OPTIONAL: ADD REFERENCE MEDIA")
        print("="*40)
        print("You can provide reference files to influence the generation:")
        print("Images: For visual style, color palette, art direction")
        print("Audio: For mood, atmosphere, musical style")
        print("Text: Additional context or specific requirements")
        print()
        print("Leave blank and press Enter to skip any option.")
        print("="*40)
        
        
        print("\nADDITIONAL TEXT DESCRIPTION:")
        print("Any extra details, style notes, or specific requirements?")
        additional_text = input("Extra description (optional): ").strip()
        
        
        print("\nIMAGE REFERENCE:")
        print("Provide path to an image file for visual style reference.")
        print("Supported formats: .jpg, .jpeg, .png, .gif, .bmp")
        image_path = input("Image file path (optional): ").strip()
        
        if image_path and not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            image_path = None
        elif image_path:
            print(f"Image file found: {image_path}")
        
        
        print("\nAUDIO REFERENCE:")
        print("Provide path to an audio file for mood/style reference.")
        print("Supported formats: .wav, .mp3, .flac, .ogg, .m4a")
        audio_path = input("Audio file path (optional): ").strip()
        
        if audio_path and not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            audio_path = None
        elif audio_path:
            print(f"Audio file found: {audio_path}")
        
        
        media_count = sum(1 for x in [additional_text, image_path, audio_path] if x)
        if media_count > 0:
            print(f"\nUsing {media_count} additional input(s) for generation.")
        else:
            print(f"\nNo additional media provided - using text description only.")
        
        return additional_text, image_path, audio_path

    def upload_media_file(self, file_path):
        """
        Upload a media file (image or audio) to Gemini using the files API
        
        Args:
            file_path (str): Path to the media file
            
        Returns:
            genai.File: Uploaded file object or None if failed
        """
        try:
            print(f"Uploading {file_path} to Gemini...")
            
            uploaded_file = genai.upload_file(file_path)
            print(f"Upload successful: {uploaded_file.name}")
            return uploaded_file
        except Exception as e:
            print(f"Failed to upload {file_path}: {str(e)}")
            return None

    def generate_asset_prompts(self, user_input, selected_assets, additional_text=None, image_path=None, audio_path=None, generation_settings=None):
        """
        Generate asset prompts based on user input and selected asset types
        
        Args:
            user_input (str): User's text description
            selected_assets (list): List of selected asset types
            additional_text (str): Optional additional text description
            image_path (str): Optional path to image file
            audio_path (str): Optional path to audio file
            
        Returns:
            dict: Parsed JSON response with requested asset prompts
        """
        
        try:
            
            content_parts = []
            
            
            system_prompt = self.get_system_prompt(selected_assets)
            content_parts.append(system_prompt)
            content_parts.append(f"\nUser Input: {user_input}")

            
            if "animation" in selected_assets and generation_settings:
                frame_count = generation_settings.get('num_frames', 8)
                content_parts.append(f"\nIMPORTANT: The user wants exactly {frame_count} animation frames. Generate exactly {frame_count} prompts in the 'prompts' array and set 'frames' to {frame_count}.")
            
            
            if additional_text:
                content_parts.append(f"\nAdditional Context: {additional_text}")
            
            
            if image_path and os.path.exists(image_path):
                try:
                    
                    uploaded_image = self.upload_media_file(image_path)
                    if uploaded_image:
                        content_parts.append("\nPlease analyze the provided image. Extract the visual style, color palette, artistic elements, and thematic content. Generate appropriate asset prompts that match the visual style shown in the image.")
                        content_parts.append(uploaded_image)
                    else:
                        
                        img = Image.open(image_path)
                        content_parts.append("\nPlease analyze the provided image. Extract the visual style, color palette, artistic elements, and thematic content. Generate appropriate asset prompts that match the visual style shown in the image.")
                        content_parts.append(img)
                except Exception as e:
                    print(f"Image processing error: {e}")
                    
            
            
            if audio_path and os.path.exists(audio_path):
                uploaded_audio = self.upload_media_file(audio_path)
                if uploaded_audio:
                    content_parts.append("\nPlease analyze the provided audio file. Extract the musical style, mood, tempo, genre, and energy level. Generate asset prompts that create visual elements matching the audio's aesthetic and emotional tone.")
                    content_parts.append(uploaded_audio)
                else:
                    print(f"Failed to upload audio, continuing without audio analysis")
            
            
            content_parts.append(f"\nProvide your response in the specified JSON format, including exactly these asset types: {', '.join(selected_assets)}.")
            
            
            response = self.model.generate_content(content_parts)
            
            
            response_content = response.text
            
            
            try:
                
                if "```json" in response_content:
                    json_start = response_content.find("```json") + 7
                    json_end = response_content.find("```", json_start)
                    response_content = response_content[json_start:json_end].strip()
                elif "```" in response_content:
                    json_start = response_content.find("```") + 3
                    json_end = response_content.find("```", json_start)
                    response_content = response_content[json_start:json_end].strip()
                
                parsed_response = json.loads(response_content)
                return {
                    "success": True,
                    "data": parsed_response,
                    "raw_response": response.text
                }
            except json.JSONDecodeError:
                
                return {
                    "success": False,
                    "error": "Failed to parse JSON response",
                    "raw_response": response.text
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"API call failed: {str(e)}",
                "raw_response": None
            }

    def display_results(self, data):
        """Helper method to display results in a formatted way"""
        generated_assets = data.get('generated_assets', [])
        
        print(f"\nANALYSIS: {data.get('analysis', 'N/A')}")
        print(f"\nGENERATED: {', '.join(generated_assets)}")
        print(f"\nTHEME: {data.get('theme', 'N/A')}")
        
        
        if 'sprite' in generated_assets and 'sprite' in data:
            sprite = data['sprite']
            print(f"\nSPRITE PROMPT:")
            print(f"   {sprite.get('prompt', 'N/A')}")
            print(f"   Technical: {sprite.get('technical_specs', 'N/A')}")
            if sprite.get('style_notes'):
                print(f"   Style: {sprite.get('style_notes')}")
        
        
        if 'environment' in generated_assets and 'environment' in data:
            environment = data['environment']
            print(f"\nENVIRONMENT PROMPT:")
            print(f"   {environment.get('prompt', 'N/A')}")
            print(f"   Technical: {environment.get('technical_specs', 'N/A')}")
            if environment.get('style_notes'):
                print(f"   Style: {environment.get('style_notes')}")
        
        
        if 'misc_asset' in generated_assets and 'misc_asset' in data:
            misc_asset = data['misc_asset']
            asset_type = misc_asset.get('asset_type', 'asset')
            print(f"\n{asset_type.upper()} PROMPT:")
            print(f"   {misc_asset.get('prompt', 'N/A')}")
            print(f"   Technical: {misc_asset.get('technical_specs', 'N/A')}")
            if misc_asset.get('style_notes'):
                print(f"   Style: {misc_asset.get('style_notes')}")
        
        
        if 'music' in generated_assets and 'music' in data:
            music = data['music']
            print(f"\nMUSIC PROMPT:")
            print(f"   {music.get('prompt', 'N/A')}")
            print(f"   Genre: {music.get('genre', 'N/A')}")
            print(f"   Mood: {music.get('mood', 'N/A')}")
            print(f"   Technical: {music.get('technical_specs', 'N/A')}")
            if music.get('style_notes'):
                print(f"   Style: {music.get('style_notes')}")
        
        
        if 'animation' in generated_assets and 'animation' in data:
            animation = data['animation']
            print(f"\nANIMATION PROMPT:")
            print(f"   Type: {animation.get('animation_type', 'N/A')}")
            print(f"   Technical: {animation.get('technical_specs', 'N/A')}")
            if animation.get('style_notes'):
                print(f"   Style: {animation.get('style_notes')}")
            
            
            if animation.get('prompts') and isinstance(animation['prompts'], list):
                print(f"   Frame Prompts:")
                for i, frame_prompt in enumerate(animation['prompts']):
                    print(f"      Frame {i+1}: {frame_prompt}")
        
        print(f"\nTONE/MOOD: {', '.join(data.get('tone_mood', []))}")
        print(f"\nCOLOR PALETTE: {', '.join(data.get('color_palette', []))}")
    
    def enhance_prompt_for_background(self, original_prompt, asset_settings):
        """
        This is only still here because of procrastination
        """
        return original_prompt

    def generate_images_from_prompts(self, result_data, selected_assets, generation_settings):
        """Generate actual images using the generated prompts"""
        
        backend = generation_settings.get('backend')
        
        if not backend:
            print("No image generation backend selected")
            return []
        
        if not result_data.get('success'):
            print("Prompt generation failed, cannot generate images")
            return []
        
        print(f"\nGENERATING ACTUAL IMAGES/ANIMATIONS using {backend.upper()}")
        print("="*50)
        
        
        generator = None  
        
        if backend == 'gemini':
            if not self.gemini_generator:
                print("Gemini generator not available")
                return []
            generator = self.gemini_generator
            
            
            if generation_settings.get('pixelate_images') is not None:
                generator.pixelate_images = generation_settings['pixelate_images']
            if generation_settings.get('pixel_size'):
                generator.pixel_size = generation_settings['pixel_size']
            if generation_settings.get('n_colors'):
                generator.n_colors = generation_settings['n_colors']
                
        elif backend == 'stable_diffusion':
            if not self.stable_diffusion_generator:
                print("Stable Diffusion generator not available")
                return []
            generator = self.stable_diffusion_generator
            
            
            if generation_settings.get('remove_background', False):
                print("Background removal enabled - adding white background instructions to prompts")
            
            
            pixel_enabled = generation_settings.get('pixelate_images', True)
            pixel_size = generation_settings.get('pixel_size', (64, 64))
            n_colors = generation_settings.get('n_colors', 16)
            remove_bg = False
            
            generator.set_pixel_processing(
                enabled=pixel_enabled,
                pixel_size=pixel_size,
                n_colors=n_colors,
                remove_background=remove_bg
            )
        else:
            print(f"Unknown backend: {backend}")
            return []
        
        
        if generator is None:
            print(f"Failed to initialize generator for backend: {backend}")
            return []
        
        generated_files = []
        data = result_data['data']
        output_dir = generation_settings.get('output_directory', '.')
        
        

        if 'sprite' in selected_assets and 'sprite' in data:
            sprite_prompt = data['sprite'].get('prompt', '')
            if sprite_prompt:
                print(f"\nGenerating sprite...")
                
                
                sprite_settings = generation_settings.get('per_asset', {}).get('sprite', {})
                
                
                if backend == 'stable_diffusion':
                    enhanced_prompt = self.enhance_prompt_for_background(sprite_prompt, sprite_settings)
                else:
                    enhanced_prompt = sprite_prompt
                style_data = result_data.get('data', {})
                tone_mood = ', '.join(style_data.get('tone_mood', []))
                color_palette = ', '.join(style_data.get('color_palette', []))
                theme = style_data.get('theme', '')

                
                style_suffix = f". Theme: {theme}"
                if tone_mood:
                    style_suffix += f", mood: {tone_mood}"
                if color_palette:
                    style_suffix += f", color palette: {color_palette}"

                enhanced_prompt = enhanced_prompt + style_suffix
                output_path = os.path.join(output_dir, "generated_sprite.png")
                reference_image = generation_settings.get('reference_image')

                try:
                    
                    kwargs = {}
                    
                    if backend == 'stable_diffusion':
                        
                        if 'aspect_ratio' in sprite_settings:
                            kwargs['aspect_ratio'] = sprite_settings['aspect_ratio']
                        
                        
                        if generation_settings.get('style_preset'):
                            kwargs['style_preset'] = generation_settings['style_preset']
                        if generation_settings.get('negative_prompt'):
                            kwargs['negative_prompt'] = generation_settings['negative_prompt']
                        
                        
                        if generation_settings.get('pixelate_images', True):
                            pixel_size = sprite_settings.get('pixel_size', (32, 32))
                            n_colors = generation_settings.get('n_colors', 16)
                            remove_bg = False
                            
                            generator.set_pixel_processing(
                                enabled=True,
                                pixel_size=pixel_size,
                                n_colors=n_colors,
                                remove_background=remove_bg
                            )
                        else:
                            generator.set_pixel_processing(enabled=False)
                    
                    elif backend == 'gemini':
                        
                        if generation_settings.get('pixelate_images', True):
                            sprite_pixel_size = sprite_settings.get('pixel_size', (32, 32))
                            n_colors = generation_settings.get('n_colors', 16)
                            
                            
                            original_pixel_size = generator.pixel_size
                            original_n_colors = generator.n_colors
                            
                            generator.pixel_size = sprite_pixel_size
                            generator.n_colors = n_colors
                            
                            print(f"Using sprite-specific pixel settings: {sprite_pixel_size[0]}x{sprite_pixel_size[1]}, {n_colors} colors")
                    
                    if reference_image and backend == 'gemini':
                        result = generator.generate_image_with_reference(
                            enhanced_prompt, reference_image, output_path, **kwargs
                        )
                    else:
                        result = generator.generate_image_from_text(
                            enhanced_prompt, output_path, **kwargs
                        )
                    
                    
                    if backend == 'gemini' and generation_settings.get('pixelate_images', True):
                        generator.pixel_size = original_pixel_size
                        generator.n_colors = original_n_colors
                    
                    if result['success']:
                        generated_files.append(result.get('saved_path', output_path))
                        print(f"Sprite saved: {result.get('saved_path', output_path)}")
                    else:
                        print(f"Sprite generation failed: {result['error']}")
                except Exception as e:
                    print(f"Sprite generation exception: {e}")
            else:
                print(f"No sprite prompt available")

        

        
        if 'environment' in selected_assets and 'environment' in data:
            env_prompt = data['environment'].get('prompt', '')
            if env_prompt:
                print(f"\nGenerating environment...")
                
                
                env_settings = generation_settings.get('per_asset', {}).get('environment', {})
                
                
                if backend == 'stable_diffusion':
                    enhanced_prompt = self.enhance_prompt_for_background(env_prompt, env_settings)
                else:
                    enhanced_prompt = env_prompt
                style_data = result_data.get('data', {})
                tone_mood = ', '.join(style_data.get('tone_mood', []))
                color_palette = ', '.join(style_data.get('color_palette', []))
                theme = style_data.get('theme', '')

                
                style_suffix = f". Theme: {theme}"
                if tone_mood:
                    style_suffix += f", mood: {tone_mood}"
                if color_palette:
                    style_suffix += f", color palette: {color_palette}"

                enhanced_prompt = enhanced_prompt + style_suffix
                output_path = os.path.join(output_dir, "generated_environment.png")
                reference_image = generation_settings.get('reference_image')
                
                try:
                    
                    kwargs = {}
                    
                    if backend == 'stable_diffusion':
                        
                        if 'aspect_ratio' in env_settings:
                            kwargs['aspect_ratio'] = env_settings['aspect_ratio']
                        
                        
                        if generation_settings.get('style_preset'):
                            kwargs['style_preset'] = generation_settings['style_preset']
                        if generation_settings.get('negative_prompt'):
                            kwargs['negative_prompt'] = generation_settings['negative_prompt']
                        
                        
                        if generation_settings.get('pixelate_images', True):
                            pixel_size = env_settings.get('pixel_size', (512, 288))
                            n_colors = generation_settings.get('n_colors', 16)
                            remove_bg = False
                            
                            generator.set_pixel_processing(
                                enabled=True,
                                pixel_size=pixel_size,
                                n_colors=n_colors,
                                remove_background=remove_bg
                            )
                        else:
                            generator.set_pixel_processing(enabled=False)
                    
                    elif backend == 'gemini':
                        
                        if generation_settings.get('pixelate_images', True):
                            env_pixel_size = env_settings.get('pixel_size', (512, 288))
                            n_colors = generation_settings.get('n_colors', 16)
                            
                            
                            original_pixel_size = generator.pixel_size
                            original_n_colors = generator.n_colors
                            
                            generator.pixel_size = env_pixel_size
                            generator.n_colors = n_colors
                            
                            print(f"Using environment-specific pixel settings: {env_pixel_size[0]}x{env_pixel_size[1]}, {n_colors} colors")
                    
                    if reference_image and backend == 'gemini':
                        result = generator.generate_image_with_reference(
                            enhanced_prompt, reference_image, output_path, **kwargs
                        )
                    else:
                        result = generator.generate_image_from_text(
                            enhanced_prompt, output_path, **kwargs
                        )
                    
                    
                    if backend == 'gemini' and generation_settings.get('pixelate_images', True):
                        generator.pixel_size = original_pixel_size
                        generator.n_colors = original_n_colors
                    
                    if result['success']:
                        generated_files.append(result.get('saved_path', output_path))
                        print(f"Environment saved: {result.get('saved_path', output_path)}")
                    else:
                        print(f"Environment generation failed: {result['error']}")
                except Exception as e:
                    print(f"Environment generation exception: {e}")
            else:
                print(f"No environment prompt available")

        
        if 'misc_asset' in selected_assets and 'misc_asset' in data:
            misc_prompt = data['misc_asset'].get('prompt', '')
            asset_type = data['misc_asset'].get('asset_type', 'asset')
            if misc_prompt:
                print(f"\nGenerating {asset_type}...")
                
                
                misc_settings = generation_settings.get('per_asset', {}).get('misc_asset', {})
                
                
                if backend == 'stable_diffusion':
                    enhanced_prompt = self.enhance_prompt_for_background(misc_prompt, misc_settings)
                else:
                    enhanced_prompt = misc_prompt
                style_data = result_data.get('data', {})
                tone_mood = ', '.join(style_data.get('tone_mood', []))
                color_palette = ', '.join(style_data.get('color_palette', []))
                theme = style_data.get('theme', '')

                
                style_suffix = f". Theme: {theme}"
                if tone_mood:
                    style_suffix += f", mood: {tone_mood}"
                if color_palette:
                    style_suffix += f", color palette: {color_palette}"

                enhanced_prompt = enhanced_prompt + style_suffix
                output_path = os.path.join(output_dir, f"generated_{asset_type.replace('/', '_')}.png")
                reference_image = generation_settings.get('reference_image')
                
                try:
                    
                    kwargs = {}
                    
                    if backend == 'stable_diffusion':
                        
                        if 'aspect_ratio' in misc_settings:
                            kwargs['aspect_ratio'] = misc_settings['aspect_ratio']
                        
                        
                        if generation_settings.get('style_preset'):
                            kwargs['style_preset'] = generation_settings['style_preset']
                        if generation_settings.get('negative_prompt'):
                            kwargs['negative_prompt'] = generation_settings['negative_prompt']
                        
                        
                        if generation_settings.get('pixelate_images', True):
                            pixel_size = misc_settings.get('pixel_size', (64, 64))
                            n_colors = generation_settings.get('n_colors', 16)
                            remove_bg = False
                            
                            generator.set_pixel_processing(
                                enabled=True,
                                pixel_size=pixel_size,
                                n_colors=n_colors,
                                remove_background=remove_bg
                            )
                        else:
                            generator.set_pixel_processing(enabled=False)
                    
                    elif backend == 'gemini':
                        
                        if generation_settings.get('pixelate_images', True):
                            misc_pixel_size = misc_settings.get('pixel_size', (64, 64))
                            n_colors = generation_settings.get('n_colors', 16)
                            
                            
                            original_pixel_size = generator.pixel_size
                            original_n_colors = generator.n_colors
                            
                            generator.pixel_size = misc_pixel_size
                            generator.n_colors = n_colors
                            
                            print(f"Using misc asset-specific pixel settings: {misc_pixel_size[0]}x{misc_pixel_size[1]}, {n_colors} colors")
                    
                    if reference_image and backend == 'gemini':
                        result = generator.generate_image_with_reference(
                            enhanced_prompt, reference_image, output_path, **kwargs
                        )
                    else:
                        result = generator.generate_image_from_text(
                            enhanced_prompt, output_path, **kwargs
                        )
                    
                    
                    if backend == 'gemini' and generation_settings.get('pixelate_images', True):
                        generator.pixel_size = original_pixel_size
                        generator.n_colors = original_n_colors
                    
                    if result['success']:
                        generated_files.append(result.get('saved_path', output_path))
                        print(f"{asset_type.title()} saved: {result.get('saved_path', output_path)}")
                    else:
                        print(f"{asset_type.title()} generation failed: {result['error']}")
                except Exception as e:
                    print(f"{asset_type.title()} generation exception: {e}")
            else:
                print(f"No {asset_type} prompt available")

        
        if 'animation' in selected_assets and 'animation' in data:
            if backend != 'gemini':
                print("\nAnimation generation is only supported by Gemini backend")
            else:
                animation_data = data['animation']
                animation_type = animation_data.get('animation_type', 'animation')
                
                
                gemini_frame_count = animation_data.get('frames', generation_settings.get('num_frames', 8))
                frame_prompts = animation_data.get('prompts', [])
                
                style_data = result_data.get('data', {})
                tone_mood = ', '.join(style_data.get('tone_mood', []))
                color_palette = ', '.join(style_data.get('color_palette', []))
                theme = style_data.get('theme', '')

                enhanced_frame_prompts = []
                for prompt in frame_prompts:
                    style_suffix = f". Theme: {theme}"
                    if tone_mood:
                        style_suffix += f", mood: {tone_mood}"
                    if color_palette:
                        style_suffix += f", color palette: {color_palette}"
                    enhanced_frame_prompts.append(prompt + style_suffix)
                frame_prompts = enhanced_frame_prompts
                print(f"\nAnimation details from Gemini:")
                print(f"   Type: {animation_type}")
                print(f"   Frames: {gemini_frame_count}")
                print(f"   Frame prompts: {len(frame_prompts)} provided")
                
                try:
                    
                    base_image_path = generation_settings.get('reference_image')
                    if not base_image_path:
                        
                        if frame_prompts:
                            print(f"\nGenerating base image for animation...")
                            base_image_path = os.path.join(output_dir, "animation_base.png")
                            
                            result = generator.generate_image_from_text(
                                frame_prompts[0], base_image_path
                            )
                            
                            if not result['success']:
                                print(f"Base image generation failed: {result['error']}")
                                return generated_files
                        else:
                            print(f"No frame prompts available for base image generation")
                            return generated_files
                    
                    if base_image_path and os.path.exists(base_image_path):
                        print(f"\nGenerating animation: {animation_type}")
                        
                        frame_duration = generation_settings.get('frame_duration', 500)
                        cleanup_frames = generation_settings.get('cleanup_frames', False)
                        
                        gif_output_path = os.path.join(output_dir, f"generated_{animation_type}_animation.gif")
                        reference_image = generation_settings.get('reference_image')

                        
                        reference_width, reference_height = None, None
                        if base_image_path and os.path.exists(base_image_path):
                            try:
                                with Image.open(base_image_path) as ref_img:
                                    reference_width, reference_height = ref_img.size
                                print(f"Reference image dimensions: {reference_width}x{reference_height}")
                            except Exception as e:
                                print(f"Could not get reference image dimensions: {e}")

                        
                        original_pixelate = generator.pixelate_images
                        original_pixel_size = generator.pixel_size
                        original_n_colors = generator.n_colors
                        
                        
                        if generation_settings.get('pixelate_images', True):
                            
                            animation_settings = generation_settings.get('per_asset', {}).get('animation', {})
                            if animation_settings.get('pixel_size'):
                                generator.pixel_size = animation_settings['pixel_size']
                                print(f"Using animation-specific pixel settings: {generator.pixel_size}")
                            
                            generator.pixelate_images = True
                            if generation_settings.get('n_colors'):
                                generator.n_colors = generation_settings['n_colors']

                        try:
                            result = generator.generate_complete_animation(
                                base_image_path,
                                animation_type,
                                gemini_frame_count,
                                gif_output_path,
                                frame_duration,
                                cleanup_frames,
                                use_previous_frame=True,
                                frame_prompts=frame_prompts,
                                reference_image=reference_image,
                                target_size=(reference_width, reference_height) if reference_width and reference_height else None
                            )
                            
                            if result['success']:
                                generated_files.append(result['gif_path'])
                                print(f"Animation saved: {result['gif_path']}")
                                if result.get('failed_frames'):
                                    print(f"Some frames failed: {result['failed_frames']}")
                            else:
                                print(f"Animation generation failed: {result['error']}")
                        
                        finally:
                            
                            generator.pixelate_images = original_pixelate
                            generator.pixel_size = original_pixel_size
                            generator.n_colors = original_n_colors
                            print(f"Restored original generator settings")
                            
                    else:
                        print(f"No base image available for animation generation")
                except Exception as e:
                    print(f"Animation generation exception: {e}")
                    
                    try:
                        generator.pixelate_images = original_pixelate
                        generator.pixel_size = original_pixel_size
                        generator.n_colors = original_n_colors
                    except:
                        pass
        
        return generated_files

    def generate_music(self, result_data, selected_assets):
        """Generate actual music files if music was requested and generation succeeded"""

        
        if 'music' not in selected_assets:
            print("No music requested")
            return []
            
        if not MUSIC_GENERATION_AVAILABLE:
            print("Music generation not available")
            return []
        
        if not result_data.get('success') or 'music' not in result_data.get('data', {}):
            print("No music data in results")
            return []
        
        music_data = result_data['data']['music']
        music_prompt = music_data.get('prompt', '')

        
        style_data = result_data.get('data', {})
        tone_mood = ', '.join(style_data.get('tone_mood', []))
        theme = style_data.get('theme', '')

        
        if theme:
            music_prompt += f". Theme: {theme}"
        if tone_mood:
            music_prompt += f", mood: {tone_mood}"        
        if not music_prompt:
            print("No music prompt generated, skipping music generation")
            return []
        
        print(f"\nGENERATING ACTUAL MUSIC FILES")
        print(f"Using prompt: {music_prompt}")
        
        try:
            
            generated_files = generate_music_from_prompt(music_prompt)
            
            if generated_files:
                print(f"Generated {len(generated_files)} music file(s)")
                return generated_files
            else:
                print("Music generation failed - no files returned")
                return []
                
        except Exception as e:
            print(f"Music generation failed with exception: {e}")
            print(f"Exception details: {traceback.format_exc()}")
            return []


def main():
    """Main function to demonstrate the API usage"""
    
    print("="*50)
    print("ENVIRONMENT SETUP CHECK")
    print("="*50)
    
    try:
        generator = GameAssetPromptGenerator()
    except ValueError as e:
        print(f"\nSetup Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have a .env file in the same directory as this script")
        print("2. Your .env file should contain: GEMINI_API_KEY=your-key-here")
        print("3. Make sure there are no spaces around the = sign")
        print("4. Get your API key from: https://aistudio.google.com/app/apikey")
        return
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return
    
    while True:
        try:
            generator.display_asset_menu()
            selected_assets = generator.get_user_asset_selection()

            if selected_assets is None: 
                break

            backend = generator.get_image_generation_backend(selected_assets)
            if backend is None and any(asset in selected_assets for asset in ['sprite', 'environment', 'misc_asset', 'animation']):
                break

            generation_settings = generator.get_generation_settings(selected_assets, backend)

            print("\n" + "="*30)
            print("DESCRIBE YOUR ASSET:")
            print("** These are used by Gemini-Flash to create detailed prompts")
            print("\n" + "_"*30)
            print("Examples:")

            if "animation" in selected_assets:
                print("  - 'A cat leaps gracefully and lands on its feet'")
                print("  - Type 'back()' to go back to asset selection")
                print("  - Type 'quit()' to exit")
            else:
                print("  - 'A wizard holding a wooden staff'")
                print("  - 'Lakeside forest in the afternoon'")
                print("  - 'Epic orchestral battle theme with drums'")
                print("  - 'image:/path/to/image.jpg Create matching assets'")
                print("  - 'audio:/path/to/song.wav Create matching assets'")
                print("  - Type 'back()' to go back to asset selection")
                print("  - Type 'quit()' to exit")
            
            user_input = input("\nDescribe your asset: ").strip()
            
            if user_input.lower() == 'quit()':
                print("\nBye, see you again! 👋")
                break
            elif user_input.lower() == 'back()':
                continue
            elif not user_input:
                print("Please enter a description. 😢")
                continue
            
            image_path = None
            audio_path = None
            
            if 'image:' in user_input:
                try:
                    image_start = user_input.find('image:') + 6
                    image_end = user_input.find(' ', image_start)
                    if image_end == -1:
                        image_end = len(user_input)
                    
                    image_path = user_input[image_start:image_end].strip()
                    user_input = user_input.replace(f'image:{image_path}', '').strip()
                    
                    if not os.path.exists(image_path):
                        print(f"Image file not found: {image_path}")
                        continue
                        
                    print(f"Using image: {image_path}")
                except Exception as e:
                    print(f"Error parsing image command: {e}")
                    continue
            
            
            if 'audio:' in user_input:
                try:
                    audio_start = user_input.find('audio:') + 6
                    audio_end = user_input.find(' ', audio_start)
                    if audio_end == -1:
                        audio_end = len(user_input)
                    
                    audio_path = user_input[audio_start:audio_end].strip()
                    user_input = user_input.replace(f'audio:{audio_path}', '').strip()
                    
                    if not os.path.exists(audio_path):
                        print(f"Audio file not found: {audio_path}")
                        continue
                        
                    print(f"Using audio: {audio_path}")
                except Exception as e:
                    print(f"Error parsing audio command: {e}")
                    continue
            
            if not user_input.strip() and (image_path or audio_path):
                user_input = "Analyze the provided media and create matching game assets"
            
            print("\nGenerating asset prompts...")
            
            reference_image_path = generation_settings.get('reference_image')
            result = generator.generate_asset_prompts(user_input, selected_assets, None, image_path or reference_image_path, audio_path, generation_settings)
            
            if result["success"]:
                generator.display_results(result["data"])
                
                visual_assets = [asset for asset in selected_assets if asset in ['sprite', 'environment', 'misc_asset', 'animation']]
                
                if IMAGE_GENERATION_AVAILABLE and visual_assets and generation_settings.get('backend'):
                    print("\n" + "="*40)
                    backend_name = generation_settings.get('backend', 'Unknown').title()
                    generate_choice = input(f"Generate actual images using {backend_name}? (y/n): ").strip().lower()
                    
                    if generate_choice in ['y', 'yes']:
                        generated_files = generator.generate_images_from_prompts(
                            result, selected_assets, generation_settings
                        )
                        
                        if generated_files:
                            print(f"\nGENERATION COMPLETE!")
                            print(f"Generated {len(generated_files)} file(s):")
                            for file_path in generated_files:
                                print(f"   - {file_path}")
                        else:
                            print("\nNo files were generated successfully")
                else:
                    print("WHY DIDN'T YOU CLICK YES")
                if 'music' in selected_assets:
                    print(f"\nDEBUG: About to call music generation...")
                    try:
                        music_files = generator.generate_music(result, selected_assets)
                        if music_files:
                            print(f"\nMUSIC FILES GENERATED:")
                            for file_path in music_files:
                                print(f"   {file_path}")
                        else:
                            print(f"\nNo music files were generated")
                    except Exception as e:
                        print(f"Music generation error: {e}")
            else:
                print(f"\nError: {result['error']}")
                if result.get('raw_response'):
                    print(f"\nRaw response: {result['raw_response']}")

            
            
            print("\n" + "="*30)
            while True:
                continue_choice = input("Generate more assets? (y/n): ").strip().lower()
                if continue_choice in ['y', 'yes']:
                    break
                elif continue_choice in ['n', 'no']:
                    print("\n\nThanks for using the Game Asset Prompt Generator, see you again! 👋")
                    return
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
                    
        except KeyboardInterrupt:
            print("\n\nGoodbye, see you again! 👋")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            continue

if __name__ == "__main__":
    main()