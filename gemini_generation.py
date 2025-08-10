import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import time
from typing import Optional, List, Dict, Any

def get_next_usage_folder():
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
    
    
    next_num = max(existing_folders) + 1 if existing_folders else 1
    usage_folder = os.path.join(outputs_dir, f"Usage{next_num}")
    
    
    os.makedirs(usage_folder, exist_ok=True)
    
    return usage_folder


try:
    from image_processing import create_pixel_art_from_image, create_palette_preview, apply_palette_to_image
    PIXEL_ART_AVAILABLE = True
except ImportError:
    PIXEL_ART_AVAILABLE = False
    print("Pixel art processing not available. Install required packages for image_processing.py")

class GeminiImageGenerator:
    def __init__(self, pixelate_images=True, pixel_size=(64, 64), n_colors=16, remove_background=False, session_folder=None):
        
        if session_folder is None:
            session_folder = get_next_usage_folder()
        
        
        self.session_folder = os.path.abspath(session_folder)
        os.makedirs(self.session_folder, exist_ok=True)
        print(f"Session folder created: {self.session_folder}")
        
        
        load_dotenv()
        
        
        self.pixelate_images = pixelate_images and PIXEL_ART_AVAILABLE
        self.pixel_size = pixel_size
        self.n_colors = n_colors
        self.remove_background = remove_background
        
        
        current_dir = os.getcwd()
        env_path = os.path.join(current_dir, '.env')
        print(f"Looking for .env file at: {env_path}")
        print(f".env file exists: {os.path.exists(env_path)}")
        
        if self.pixelate_images:
            print(f"Pixel art mode enabled: {self.pixel_size[0]}x{self.pixel_size[1]} pixels, {self.n_colors} colors")
        elif not PIXEL_ART_AVAILABLE:
            print("Pixel art mode disabled: image_processing module not available")
        else:
            print("Regular image mode (no pixelation)")
        
        
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            print(f"Gemini API key found: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else api_key}")
        else:
            print("GEMINI_API_KEY not found in environment variables")
            print("Available environment variables starting with 'GEMINI':")
            for key in os.environ:
                if key.startswith('GEMINI'):
                    print(f"  - {key}")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")
        
        
        self.client = genai.Client(api_key=api_key)
        
        
        self.model_name = "gemini-2.0-flash-preview-image-generation"
    
    def _ensure_session_path(self, path: Optional[str]) -> Optional[str]:
        """Ensure a path is within the session folder"""
        if path is None:
            return None
            
        
        if os.path.isabs(path):
            return path
            
        
        return os.path.join(self.session_folder, path)

    def apply_palette_to_image(image, palette_image, pixel_size=(16, 16)):
        """
        Apply an existing color palette to a new image for consistency
        
        Args:
            image (PIL.Image): Input image to recolor
            palette_image (PIL.Image): Image containing the reference palette
            pixel_size (tuple): Target pixel dimensions (ignored if preserve_size=True)
            preserve_size (bool): If True, keep original image dimensions
            
        Returns:
            PIL.Image: Image with applied palette
        """
        try:
            palette_colors = extract_palette_from_reference_image(palette_image)
            
            if len(palette_colors) == 0:
                print("Could not extract palette, fall back: standard quantization")
                return create_pixel_art_from_image(
                    image, 
                    pixel_size=pixel_size, 
                    n_colors=16, 
                )
            
            print(f"Applying palette with {len(palette_colors)} colors")
            
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
            
            

            resized_image = resize_image(image, pixel_size)
            
            recolored_image = map_image_to_palette(resized_image, palette_colors)
            
            return recolored_image
            
        except Exception as e:
            print(f"Error applying palette: {e}")
            print("Fall back: standard pixel art processing")
            return create_pixel_art_from_image(
                image, 
                pixel_size=pixel_size,
            )
    def _apply_pixel_art_processing(self, image: Image.Image, output_path: Optional[str] = None, palette_image: Optional[Image.Image] = None) -> Image.Image:
        """
        Apply pixel art processing to an image with automatic palette generation and saving
        
        Args:
            image (PIL.Image): Input image to process
            output_path (str): Optional path to save the processed image
            palette_image (PIL.Image): Optional palette image for consistency
            
        Returns:
            PIL.Image: Processed image (pixelated if enabled, otherwise original)
        """
        if not self.pixelate_images or not PIXEL_ART_AVAILABLE:
            if output_path:
                
                abs_output_path = os.path.abspath(output_path)
                abs_output_dir = os.path.dirname(abs_output_path)
                if abs_output_dir:
                    os.makedirs(abs_output_dir, exist_ok=True)
                image.save(abs_output_path)
                print(f"Original image saved to: {abs_output_path}")
            return image
        
        try:
            print(f"Applying pixel art processing: {self.pixel_size[0]}x{self.pixel_size[1]}, {self.n_colors} colors")
            
            if palette_image is not None:
                
                print("Using existing color palette for consistency")
                
                
                pixel_art = apply_palette_to_image(
                    image,
                    palette_image,
                    pixel_size=self.pixel_size
                )
            else:
                
                pixel_art = create_pixel_art_from_image(
                    image, 
                    pixel_size=self.pixel_size,
                    n_colors=self.n_colors,
                    remove_bg=self.remove_background,
                    bg_threshold=240
                )
            
            print(f"Pixel art processing complete: {pixel_art.size}")

            
            if output_path:
                
                abs_output_path = os.path.abspath(output_path)
                abs_output_dir = os.path.dirname(abs_output_path)
                if abs_output_dir:
                    os.makedirs(abs_output_dir, exist_ok=True)
                pixel_art.save(abs_output_path)
                print(f"Pixel art saved to: {abs_output_path}")
                
                
                try:
                    palette_preview = create_palette_preview(pixel_art)
                    if palette_preview:
                        
                        palette_filename = os.path.splitext(abs_output_path)[0] + "_palette.png"
                        palette_preview.save(palette_filename)
                        print(f"Color palette saved to: {palette_filename}")
                except Exception as e:
                    print(f"Could not create color palette: {e}")
            
            return pixel_art
            
        except Exception as e:
            print(f"Pixel art processing failed: {e}")
            print("Returning original image")
            if output_path:
                
                abs_output_path = os.path.abspath(output_path)
                abs_output_dir = os.path.dirname(abs_output_path)
                if abs_output_dir:
                    os.makedirs(abs_output_dir, exist_ok=True)
                image.save(abs_output_path)
                print(f"Original image saved to: {abs_output_path}")
            return image
    def _apply_pixel_art_at_target_size(self, image: Image.Image, target_size: tuple, output_path: Optional[str] = None, palette_image: Optional[Image.Image] = None) -> Image.Image:
        """
        Apply pixel art processing directly at target dimensions (no intermediate sizing)
        
        Args:
            image (PIL.Image): Input image (already at target size)
            target_size (tuple): Target dimensions (width, height)
            output_path (str): Optional path to save the pixelated image
            palette_image (PIL.Image): Optional palette image to maintain consistency
            
        Returns:
            PIL.Image: Pixelated image at target size
        """
        if not self.pixelate_images or not PIXEL_ART_AVAILABLE:
            if output_path:
                image.save(output_path)
            return image
        
        try:
            print(f"Applying pixel art processing directly at target size: {target_size}")
            
            if palette_image is not None:
                
                print("Using existing color palette for consistency")
                
                
                pixel_art = apply_palette_to_image(
                    image,
                    palette_image,
                    pixel_size=target_size  
                )
            else:
                
                pixel_art = create_pixel_art_from_image(
                    image, 
                    pixel_size=target_size,  
                    n_colors=self.n_colors,
                    remove_bg=self.remove_background,
                    bg_threshold=10
                )
            
            print(f"Pixel art processed directly at target size: {target_size}")

            if output_path:
                abs_output_path = os.path.abspath(output_path)
                abs_output_dir = os.path.dirname(abs_output_path)
                if abs_output_dir:  
                    os.makedirs(abs_output_dir, exist_ok=True)
                pixel_art.save(abs_output_path)
                print(f"Pixel art saved to: {abs_output_path}")
            
            return pixel_art
            
        except Exception as e:
            print(f"Pixel art processing failed: {e}")
            print("Returning original image")
            if output_path:
                image.save(output_path)
            return image


    def generate_image_from_text(self, prompt: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an image from text prompt
        
        Args:
            prompt (str): Text description for image generation
            output_path (str): Optional path to save the image (will be placed in session folder if relative)
            
        Returns:
            dict: Result with success status and image data
        """
        try:
            print(f"Generating image with prompt: {prompt}")
            
            
            if output_path:
                final_output_path = self._ensure_session_path(output_path)
            else:
                
                pixel_suffix = "_pixelated" if self.pixelate_images else ""
                final_output_path = os.path.join(self.session_folder, f"generated_image{pixel_suffix}.png")
            
            print(f"Will save to: {final_output_path}")
            
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            
            if not response or not response.candidates:
                return {
                    "success": False,
                    "error": "No response or candidates received from API"
                }
            
            
            for candidate in response.candidates:
                if not candidate.content or not candidate.content.parts:
                    continue
                    
                for part in candidate.content.parts:
                    if part.text is not None:
                        print(f"Model response: {part.text}")
                    elif part.inline_data is not None and part.inline_data.data is not None:
                        try:
                            
                            image_data = part.inline_data.data
                            if isinstance(image_data, bytes):
                                image = Image.open(BytesIO(image_data))
                            else:
                                return {
                                    "success": False,
                                    "error": "Invalid image data type received"
                                }
                            
                            
                            processed_image = self._apply_pixel_art_processing(image, final_output_path)
                            
                            return {
                                "success": True,
                                "image": processed_image,
                                "saved_path": final_output_path,
                                "session_folder": self.session_folder,
                                "message": "Image generated successfully" + (" and pixelated" if self.pixelate_images else "")
                            }
                            
                        except Exception as e:
                            print(f"Error processing image data: {e}")
                            return {
                                "success": False,
                                "error": f"Failed to process image: {str(e)}"
                            }
            
            
            return {
                "success": False,
                "error": "No image data found in response",
                "raw_response": str(response)
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"API call failed: {str(e)}",
                "raw_response": None
            }
    
    def generate_image_with_reference(self, text_prompt: str, reference_image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an image based on text prompt and reference image
        
        Args:
            text_prompt (str): Text description for image generation
            reference_image_path (str): Path to reference image
            output_path (str): Optional path to save the generated image (will be placed in session folder if relative)
            
        Returns:
            dict: Result with success status and image data
        """
        try:
            if not os.path.exists(reference_image_path):
                return {
                    "success": False,
                    "error": f"Reference image not found: {reference_image_path}"
                }
            
            print(f"Generating image with reference: {reference_image_path}")
            print(f"Text prompt: {text_prompt}")
            
            
            if output_path:
                final_output_path = self._ensure_session_path(output_path)
            else:
                
                pixel_suffix = "_pixelated" if self.pixelate_images else ""
                ref_basename = os.path.splitext(os.path.basename(reference_image_path))[0]
                final_output_path = os.path.join(self.session_folder, f"generated_with_{ref_basename}{pixel_suffix}.png")
            
            print(f"Will save to: {final_output_path}")
            
            
            reference_image = Image.open(reference_image_path)
            
            
            enhanced_prompt = f"Based on the style and elements of the reference image, generate an image: {text_prompt}"
            
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[enhanced_prompt, reference_image],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            
            if not response or not response.candidates:
                return {
                    "success": False,
                    "error": "No response or candidates received from API"
                }
            
            
            for candidate in response.candidates:
                if not candidate.content or not candidate.content.parts:
                    continue
                    
                for part in candidate.content.parts:
                    if part.text is not None:
                        print(f"Model response: {part.text}")
                    elif part.inline_data is not None and part.inline_data.data is not None:
                        try:
                            image_data = part.inline_data.data
                            if isinstance(image_data, bytes):
                                image = Image.open(BytesIO(image_data))
                            else:
                                return {
                                    "success": False,
                                    "error": "Invalid image data type received"
                                }
                            
                            
                            processed_image = self._apply_pixel_art_processing(image, final_output_path)
                            
                            return {
                                "success": True,
                                "image": processed_image,
                                "saved_path": final_output_path,
                                "session_folder": self.session_folder,
                                "message": "Image generated successfully with reference" + (" and pixelated" if self.pixelate_images else "")
                            }
                            
                        except Exception as e:
                            return {
                                "success": False,
                                "error": f"Failed to process image: {str(e)}"
                            }
            
            return {
                "success": False,
                "error": "No image data found in response",
                "raw_response": str(response)
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"API call failed: {str(e)}"
            }

    def generate_animation_frames(self, base_image_path: str, animation_description: str, num_frames: int, 
                                frame_duration: float = 0.5, output_dir: str = "animation_frames", 
                                use_previous_frame: bool = True, frame_prompts: Optional[List[str]] = None, 
                                reference_image: Optional[str] = None, target_size: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Generate animation frames based on a base image and animation description
        
        Args:
            base_image_path (str): Path to the base/reference image
            animation_description (str): Description of how the image should animate
            num_frames (int): Number of frames to generate
            frame_duration (float): Duration of each frame in seconds (for final GIF)
            output_dir (str): Directory to save individual frames (relative to session folder)
            use_previous_frame (bool): Whether to use the previous frame as reference for consistency
            frame_prompts (list): Optional list of specific frame prompts from Gemini
            reference_image (str): Optional user-uploaded reference image for style consistency
            target_size (tuple): Optional target size for all frames (width, height)
            
        Returns:
            dict: Result with success status and frame paths
        """
        try:
            if not os.path.exists(base_image_path):
                return {
                    "success": False,
                    "error": f"Base image not found: {base_image_path}"
                }
            
            print(f"Creating {num_frames} animation frames")
            print(f"Animation: {animation_description}")
            print(f"Base image: {base_image_path}")
            print(f"Using previous frame context: {use_previous_frame}")
            
            
            user_reference_available = reference_image and os.path.exists(reference_image)
            if user_reference_available:
                print(f"Using user reference image for style consistency: {os.path.basename(reference_image)}")
                print(f"User reference will be added as the first frame of the animation")
            
            
            full_output_dir = self._ensure_session_path(output_dir)
            if full_output_dir:
                os.makedirs(full_output_dir, exist_ok=True)
                print(f"Animation frames will be saved to: {full_output_dir}")
            else:
                return {
                    "success": False,
                    "error": "Failed to create output directory path"
                }
            
            frame_paths = []
            failed_frames = []
            palette_image = None
            
            
            if not target_size:
                reference_path = reference_image if user_reference_available else base_image_path
                try:
                    with Image.open(reference_path) as ref_img:
                        target_size = ref_img.size
                        print(f"Using reference image dimensions: {target_size}")
                except Exception as e:
                    print(f"Could not get reference dimensions: {e}")
                    target_size = None
            
            
            if user_reference_available and reference_image:
                try:
                    print(f"\nAdding user reference as frame 0 (initial frame)")
                    
                    
                    user_ref_image = Image.open(reference_image)
                    
                    
                    if target_size and user_ref_image.size != target_size:
                        print(f"Resizing reference from {user_ref_image.size} to {target_size}")
                        user_ref_image = user_ref_image.resize(target_size, Image.Resampling.LANCZOS)
                    
                    first_frame_path = os.path.join(full_output_dir, "frame_000.png")
                    
                    
                    if self.pixelate_images and PIXEL_ART_AVAILABLE:
                        
                        processed_reference = self._apply_pixel_art_at_target_size(
                            user_ref_image, target_size, first_frame_path, None
                        )
                    else:
                        
                        user_ref_image.save(first_frame_path)
                        processed_reference = user_ref_image
                    
                    frame_paths.append(first_frame_path)
                    
                    
                    if self.pixelate_images and PIXEL_ART_AVAILABLE:
                        try:
                            palette_image = create_palette_preview(processed_reference)
                            palette_path = os.path.join(full_output_dir, "colorpalette.png")
                            palette_image.save(palette_path)
                            print(f"Color palette created from user reference: {palette_path}")
                        except Exception as e:
                            print(f"Could not create palette from reference: {e}")
                    
                    print(f"User reference added as frame 0: {first_frame_path}")
                    
                except Exception as e:
                    print(f"Failed to process user reference image: {e}")
                    print(f"Continuing with generated frames only")
                    user_reference_available = False
            
            
            previous_frame_path = reference_image if user_reference_available else base_image_path
            start_frame_num = 1 if user_reference_available else 0
            
            for frame_index in range(num_frames):
                actual_frame_num = start_frame_num + frame_index
                print(f"\nGenerating frame {actual_frame_num + 1}/{num_frames + (1 if user_reference_available else 0)}")
                
                
                progress = frame_index / max(1, num_frames - 1) if num_frames > 1 else 0.0
                
                
                if frame_prompts and frame_index < len(frame_prompts):
                    frame_prompt = frame_prompts[frame_index]
                    print(f"Using Gemini frame prompt: {frame_prompt[:80]}...")
                else:
                    frame_prompt = self._create_enhanced_frame_prompt(
                        animation_description, progress, frame_index, num_frames, base_image_path
                    )
                    print(f"Using generated frame prompt: {frame_prompt[:80]}...")
                
                
                frame_path = os.path.join(full_output_dir, f"frame_{actual_frame_num:03d}.png")
                
                
                reference_images = []
                reference_descriptions = []
                
                
                if user_reference_available and reference_image:
                    reference_images.append(reference_image)
                    reference_descriptions.append("style reference")
                
                
                if use_previous_frame and len(frame_paths) > 0 and previous_frame_path and os.path.exists(previous_frame_path):
                    reference_images.append(previous_frame_path)
                    reference_descriptions.append("previous frame")
                elif not user_reference_available and frame_index == 0:
                    
                    reference_images.append(base_image_path)
                    reference_descriptions.append("base image")
                
                print(f"Using references: {', '.join(reference_descriptions)}")
                
                
                result = self.generate_image_with_multiple_references_for_animation(
                    frame_prompt, reference_images, frame_path, palette_image, 
                    is_first_frame=(frame_index == 0 and not user_reference_available),
                    target_size=target_size  
                )
                
                if result["success"]:
                    frame_paths.append(frame_path)
                    previous_frame_path = frame_path  
                    
                    
                    if frame_index == 0 and not user_reference_available and self.pixelate_images and result.get("palette_image") and PIXEL_ART_AVAILABLE:
                        palette_image = result["palette_image"]
                        
                        palette_path = os.path.join(full_output_dir, "colorpalette.png")
                        if palette_image:
                            palette_image.save(palette_path)
                            print(f"Color palette saved: {palette_path}")
                    
                    print(f"Frame {actual_frame_num + 1} saved: {frame_path}")
                else:
                    failed_frames.append(actual_frame_num)
                    print(f"Frame {actual_frame_num + 1} failed: {result['error']}")
                
                
                time.sleep(1)
            
            if not frame_paths:
                return {
                    "success": False,
                    "error": "No frames were generated successfully"
                }
            
            
            total_frames_generated = len(frame_paths)
            expected_total = num_frames + (1 if user_reference_available else 0)
            
            print(f"\n📊 Frame generation summary:")
            print(f"   - User reference frame: {'Added' if user_reference_available else 'None'}")
            print(f"   - Generated frames: {total_frames_generated - (1 if user_reference_available else 0)}/{num_frames}")
            print(f"   - Total frames: {total_frames_generated}/{expected_total}")
            print(f"   - Target size: {target_size if target_size else 'Original'}")
            print(f"   - Frames saved to: {full_output_dir}")
            if failed_frames:
                print(f"   - Failed frames: {failed_frames}")
            
            return {
                "success": True,
                "frame_paths": frame_paths,
                "failed_frames": failed_frames,
                "total_frames": len(frame_paths),
                "frame_duration": frame_duration,
                "palette_image": palette_image,
                "user_reference_included": user_reference_available,
                "output_directory": full_output_dir,  
                "message": f"Generated {len(frame_paths)} total frames ({num_frames} generated + {1 if user_reference_available else 0} reference)" + (" (pixelated)" if self.pixelate_images else "")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Animation generation failed: {str(e)}"
            }



    def generate_image_with_multiple_references_for_animation(self, text_prompt: str, reference_image_paths: List[str], 
                                                            output_path: Optional[str] = None, 
                                                            palette_image: Optional[Image.Image] = None, 
                                                            is_first_frame: bool = False,
                                                            target_size: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Generate an image with multiple references specifically for animation
        
        Args:
            text_prompt (str): Text description for image generation
            reference_image_paths (list): List of paths to reference images
            output_path (str): Optional path to save the generated image
            palette_image (PIL.Image): Palette image for consistency
            is_first_frame (bool): Whether this is the first frame (to generate palette)
            target_size (tuple): Target size to avoid double resizing
            
        Returns:
            dict: Result with success status and image data
        """
        try:
            if not reference_image_paths:
                return {
                    "success": False,
                    "error": "No reference images provided"
                }
            
            valid_references = []
            for ref_path in reference_image_paths:
                if ref_path and os.path.exists(ref_path):
                    valid_references.append(ref_path)
                else:
                    print(f"Reference image not found, skipping: {ref_path}")
            
            if not valid_references:
                return {
                    "success": False,
                    "error": "No valid reference images found"
                }
            
            print(f"Generating animation frame with {len(valid_references)} reference(s)")
            print(f"Text prompt: {text_prompt}")
            
            reference_images = []
            for ref_path in valid_references:
                try:
                    ref_img = Image.open(ref_path)
                    reference_images.append(ref_img)
                    print(f"Loaded reference: {os.path.basename(ref_path)}")
                except Exception as e:
                    print(f"Failed to load reference {ref_path}: {e}")
            
            if not reference_images:
                return {
                    "success": False,
                    "error": "Failed to load any reference images"
                }
            
            if len(reference_images) == 1:
                enhanced_prompt = f"Based on the style and elements of the reference image, generate an image: {text_prompt}"
            else:
                enhanced_prompt = f"Based on the style of the first reference image and the composition/continuity of the second reference image, generate an image: {text_prompt}. Maintain the artistic style from the first reference while ensuring smooth animation continuity from the second reference."
            
            content_parts = [enhanced_prompt] + reference_images
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            if not response or not response.candidates:
                return {
                    "success": False,
                    "error": "No response or candidates received from API"
                }
            
            for candidate in response.candidates:
                if not candidate.content or not candidate.content.parts:
                    continue
                    
                for part in candidate.content.parts:
                    if part.text is not None:
                        print(f"Model response: {part.text}")
                    elif part.inline_data is not None and part.inline_data.data is not None:
                        try:
                            image_data = part.inline_data.data
                            if isinstance(image_data, bytes):
                                image = Image.open(BytesIO(image_data))
                            else:
                                return {
                                    "success": False,
                                    "error": "Invalid image data type received"
                                }
                            
                            
                            if target_size and image.size != target_size:
                                print(f"Resizing generated image from {image.size} to {target_size}")
                                image = image.resize(target_size, Image.Resampling.LANCZOS)
                            
                            
                            processed_image = self._apply_pixel_art_at_target_size(
                                image, target_size or image.size, output_path, palette_image
                            )
                            
                            result = {
                                "success": True,
                                "image": processed_image,
                                "saved_path": output_path,
                                "message": "Animation frame generated successfully with multiple references" + (" and pixelated" if self.pixelate_images else "")
                            }
                            
                            if is_first_frame and self.pixelate_images and PIXEL_ART_AVAILABLE:
                                try:
                                    palette = create_palette_preview(processed_image)
                                    result["palette_image"] = palette
                                except Exception as e:
                                    print(f"Could not create palette: {e}")
                            
                            return result
                            
                        except Exception as e:
                            return {
                                "success": False,
                                "error": f"Failed to process image: {str(e)}"
                            }
            
            return {
                "success": False,
                "error": "No image data found in response",
                "raw_response": str(response)
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"API call failed: {str(e)}"
            }


    def generate_image_with_reference_for_animation(self, text_prompt: str, reference_image_path: str, 
                                                  output_path: Optional[str] = None, 
                                                  palette_image: Optional[Image.Image] = None, 
                                                  is_first_frame: bool = False) -> Dict[str, Any]:
        """
        Generate an image with reference specifically for animation (handles palette consistency)
        
        Args:
            text_prompt (str): Text description for image generation
            reference_image_path (str): Path to reference image
            output_path (str): Optional path to save the generated image
            palette_image (PIL.Image): Palette image for consistency
            is_first_frame (bool): Whether this is the first frame (to generate palette)
            
        Returns:
            dict: Result with success status and image data
        """
        try:
            if not os.path.exists(reference_image_path):
                return {
                    "success": False,
                    "error": f"Reference image not found: {reference_image_path}"
                }
            
            print(f"Generating animation frame with reference: {reference_image_path}")
            print(f"Text prompt: {text_prompt}")
            
            
            reference_image = Image.open(reference_image_path)
            
            

            enhanced_prompt = f"Based on the style and elements of the reference image, generate an image: {text_prompt}"
            
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[enhanced_prompt, reference_image],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(f"Model response: {part.text}")
                elif part.inline_data is not None:
                    try:
                        image = Image.open(BytesIO(part.inline_data.data))
                        
                        
                        processed_image = self._apply_pixel_art_processing(image, output_path, palette_image)
                        
                        result = {
                            "success": True,
                            "image": processed_image,
                            "saved_path": output_path,
                            "message": "Animation frame generated successfully" + (" and pixelated" if self.pixelate_images else "")
                        }
                        
                        
                        if is_first_frame and self.pixelate_images and PIXEL_ART_AVAILABLE:
                            try:
                                palette = create_palette_preview(processed_image)
                                result["palette_image"] = palette
                            except Exception as e:
                                print(f"Could not create palette: {e}")
                        
                        return result
                        
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Failed to process image: {str(e)}"
                        }
            
            return {
                "success": False,
                "error": "No image data found in response",
                "raw_response": str(response)
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"API call failed: {str(e)}"
            }
    
    def _create_enhanced_frame_prompt(self, animation_description, progress, frame_num, total_frames, base_image_path):
        """
        Create an enhanced frame-specific prompt with better context preservation
        
        Args:
            animation_description (str): Base animation description
            progress (float): Animation progress (0.0 to 1.0)
            frame_num (int): Current frame number
            total_frames (int): Total number of frames
            base_image_path (str): Path to original base image for context
            
        Returns:
            str: Enhanced frame-specific prompt
        """
        
        progress_percent = int(progress * 100)
        
        
        if progress == 0.0:
            stage_description = "at the very beginning of the animation"
        elif progress < 0.25:
            stage_description = "in the early stage of the animation"
        elif progress < 0.5:
            stage_description = "in the middle-early stage of the animation"
        elif progress < 0.75:
            stage_description = "in the middle-late stage of the animation"
        elif progress < 1.0:
            stage_description = "near the end of the animation"
        else:
            stage_description = "at the final stage of the animation"
        
        
        frame_prompt = (
            f"IMPORTANT: Maintain EXACT same character orientation, facing direction, and pose as the reference image. "
            f"Do NOT flip or mirror the character horizontally. Keep the same art style, colors, lighting, and background. "
            f"Show this scene {stage_description} ({progress_percent}% complete) where {animation_description}. "
            f"This is frame {frame_num + 1} of {total_frames}. "
            f"Only animate the specific motion described ({animation_description}) while keeping everything else "
            f"identical to the reference image, especially the character's facing direction and orientation. "
            f"Make subtle, natural changes that show smooth progression of the animation. "
            f"If the character was facing left, keep them facing left. If facing right, keep facing right. "
            f"Preserve all visual details, proportions, and spatial relationships from the reference."
        )
        
        return frame_prompt
    
    def create_gif_from_frames(self, frame_paths, output_path, duration=500, loop=0):
        """
        Create an animated GIF from a list of frame images
        
        Args:
            frame_paths (list): List of paths to frame images
            output_path (str): Path for the output GIF
            duration (int): Duration of each frame in milliseconds
            loop (int): Number of loops (0 = infinite)
            
        Returns:
            dict: Result with success status
        """
        try:
            if not frame_paths:
                return {
                    "success": False,
                    "error": "No frame paths provided"
                }
            
            print(f"Creating GIF from {len(frame_paths)} frames")
            
            
            frames = []
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    frame = Image.open(frame_path)
                    
                    if frame.mode == 'RGBA':
                        frame = frame.convert('RGB')
                    frames.append(frame)
                else:
                    print(f"Frame not found: {frame_path}")
            
            if not frames:
                return {
                    "success": False,
                    "error": "No valid frames found"
                }
            
            
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=loop
            )
            
            print(f"GIF created: {output_path}")
            return {
                "success": True,
                "gif_path": output_path,
                "frame_count": len(frames),
                "message": f"GIF created successfully with {len(frames)} frames" + (" (pixelated)" if self.pixelate_images else "")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"GIF creation failed: {str(e)}"
            }
    
    def generate_complete_animation(self, base_image_path, animation_description, num_frames, 
                                gif_output_path=None, frame_duration=500, cleanup_frames=False, 
                                use_previous_frame=True, frame_prompts=None, reference_image=None,
                                target_size=None, preserve_reference_size=True):
        """
        Generate a complete animation from base image to final GIF
        
        Args:
            base_image_path (str): Path to the base/reference image
            animation_description (str): Description of the animation
            num_frames (int): Number of frames to generate
            gif_output_path (str): Path for output GIF (auto-generated if None)
            frame_duration (int): Duration of each frame in milliseconds
            cleanup_frames (bool): Whether to delete individual frames after creating GIF
            use_previous_frame (bool): Whether to use previous frame as reference for consistency
            frame_prompts (list): Optional list of specific frame prompts from Gemini
            reference_image (str): Optional user-uploaded reference image for style consistency
            target_size (tuple): Optional target size (overrides preserve_reference_size if set)
            preserve_reference_size (bool): Whether to preserve the reference image size
            
        Returns:
            dict: Result with success status and animation info
        """
        
        if preserve_reference_size and not target_size:
            reference_path = reference_image if reference_image and os.path.exists(reference_image) else base_image_path
            try:
                with Image.open(reference_path) as ref_img:
                    target_size = ref_img.size
                    print(f"Will preserve reference image size: {target_size}")
            except Exception as e:
                print(f"Could not get reference dimensions: {e}")
                target_size = None

        try:
            print("Starting complete animation generation...")
            
            
            frames_result = self.generate_animation_frames(
                base_image_path, 
                animation_description, 
                num_frames,
                use_previous_frame=use_previous_frame,
                frame_prompts=frame_prompts,
                reference_image=reference_image,
                target_size=target_size
                
            )
            
            if not frames_result["success"]:
                return frames_result
            
            
            if not gif_output_path:
                base_name = os.path.splitext(os.path.basename(base_image_path))[0]
                pixel_suffix = "_pixelated" if self.pixelate_images else ""
                gif_filename = f"{base_name}{pixel_suffix}_animated.gif"
                gif_output_path = os.path.join(self.session_folder, gif_filename)
            else:
                
                gif_output_path = self._ensure_session_path(gif_output_path)
            
            print(f"Creating GIF: {gif_output_path}")
            
            
            gif_result = self.create_gif_from_frames(
                frames_result["frame_paths"],
                gif_output_path,
                duration=frame_duration
            )
            
            if not gif_result["success"]:
                return gif_result
            
            if cleanup_frames:
                print("Cleaning up individual frames...")
                frames_cleaned = 0
                for frame_path in frames_result["frame_paths"]:
                    try:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                            frames_cleaned += 1
                    except Exception as e:
                        print(f"Could not delete {frame_path}: {e}")
                
                print(f"Cleaned up {frames_cleaned} frame files")
                
                try:
                    frames_dir = frames_result.get("output_directory")
                    if frames_dir and os.path.exists(frames_dir):
                        remaining_files = os.listdir(frames_dir)
                        
                        if not remaining_files or (len(remaining_files) == 1 and remaining_files[0] == "colorpalette.png"):
                            if remaining_files:  
                                os.remove(os.path.join(frames_dir, "colorpalette.png"))
                            os.rmdir(frames_dir)
                            print(f"Removed empty frames directory: {frames_dir}")
                except Exception as e:
                    print(f"Could not remove frames directory: {e}")
            
            return {
                "success": True,
                "gif_path": gif_output_path,
                "frame_paths": frames_result["frame_paths"] if not cleanup_frames else [],
                "frames_directory": frames_result.get("output_directory"),
                "total_frames": frames_result["total_frames"],
                "failed_frames": frames_result["failed_frames"],
                "frame_duration": frame_duration,
                "palette_image": frames_result.get("palette_image"),
                "session_folder": self.session_folder,  
                "message": f"Complete animation created: {gif_output_path}" + (" (pixelated)" if self.pixelate_images else "")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Complete animation generation failed: {str(e)}"
            } 

def main():
    """Main function to demonstrate image and animation generation"""
    
    print("="*60)
    print("GEMINI 2.0 IMAGE & ANIMATION GENERATOR WITH PIXEL ART")
    print("="*60)
    
    try:
        
        
        generator = GeminiImageGenerator(
            pixelate_images=True,    
            pixel_size=(64, 64),     
            n_colors=16              
        )
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
    
    print(f"\nGEMINI IMAGE & ANIMATION GENERATOR WITH PIXEL ART")
    print("="*60)
    print(f"Session folder: {generator.session_folder}")
    print("Required packages: pip install google-genai pillow python-dotenv scikit-learn numpy")
    print("Note: Uses the new google.genai client (not google.generativeai)")
    print("Pixel Art: All generated images will be converted to pixel art!")
    print(f"Pixel settings: {generator.pixel_size[0]}x{generator.pixel_size[1]} pixels, {generator.n_colors} colors")
    print("\nGeneration Options:")
    print("  1. Single Image Generation")
    print("  2. Animation Generation")
    print("\nCommands:")
    print("  - 'single' or '1' for single image generation")
    print("  - 'animation' or '2' for animation generation")
    print("  - 'settings' to adjust pixel art settings")
    print("  - 'quit' to exit")
    print("\nSingle Image Examples:")
    print("  - 'A majestic dragon flying over a medieval castle'")
    print("  - 'ref:castle.jpg A dragon in the same art style'")
    print("  - 'save:my_dragon.png A friendly cartoon dragon'")
    print("\nAnimation Examples:")
    print("  - Base image: cat.jpg, Frames: 8, Description: 'the cat slowly opens and closes its eyes'")
    print("  - Base image: tree.jpg, Frames: 12, Description: 'leaves falling from the tree in autumn wind'")
    print("="*60)
    
    
    while True:
        print("\n" + "="*50)
        print("Choose generation type:")
        print("1. Single Image")
        print("2. Animation")
        print("3. Settings")
        print("4. Quit")
        print("="*50)
        
        choice = input("Enter your choice (1-4): ").strip().lower()
        
        if choice in ['quit', '4', 'q']:
            break
        
        elif choice in ['settings', '3']:
            print(f"\nCurrent pixel art settings:")
            print(f"  - Pixelate images: {generator.pixelate_images}")
            print(f"  - Pixel size: {generator.pixel_size[0]}x{generator.pixel_size[1]}")
            print(f"  - Number of colors: {generator.n_colors}")
            
            print("\nEnter new settings (or press Enter to keep current):")
            
            
            toggle = input(f"Enable pixelation? (y/n) [current: {'y' if generator.pixelate_images else 'n'}]: ").strip().lower()
            if toggle in ['y', 'yes']:
                generator.pixelate_images = True
            elif toggle in ['n', 'no']:
                generator.pixelate_images = False
            
            if generator.pixelate_images:
                
                size_input = input(f"Pixel size (width,height) [current: {generator.pixel_size[0]},{generator.pixel_size[1]}]: ").strip()
                if size_input:
                    try:
                        width, height = map(int, size_input.split(','))
                        generator.pixel_size = (width, height)
                        print(f"Pixel size set to: {generator.pixel_size}")
                    except:
                        print("Invalid format. Use: width,height (e.g., 32,32)")
                
                
                colors_input = input(f"Number of colors [current: {generator.n_colors}]: ").strip()
                if colors_input:
                    try:
                        generator.n_colors = int(colors_input)
                        print(f"Number of colors set to: {generator.n_colors}")
                    except:
                        print("Invalid number")
            
            print(f"\nSettings updated!")
            continue
        
        elif choice in ['single', '1']:
            
            print("\nSINGLE IMAGE GENERATION")
            print(f"Images will be saved to: {generator.session_folder}")
            print("Commands:")
            print("  - Just type a description to generate an image")
            print("  - 'ref:/path/to/image.jpg your prompt' to use reference image")
            print("  - 'save:filename.png your prompt' to specify save name (within session folder)")
            print("  - 'back' to return to main menu")
            
            while True:
                user_input = input("\nImage prompt: ").strip()
                
                if user_input.lower() == 'back':
                    break
                
                if not user_input:
                    continue
                
                
                reference_image = None
                output_filename = None  
                prompt = user_input
                
                
                if 'ref:' in user_input:
                    try:
                        ref_start = user_input.find('ref:') + 4
                        ref_end = user_input.find(' ', ref_start)
                        if ref_end == -1:
                            print("Please provide a prompt after the reference image path")
                            continue
                        
                        reference_image = user_input[ref_start:ref_end].strip()
                        prompt = user_input[ref_end:].strip()
                        
                        if not os.path.exists(reference_image):
                            print(f"Reference image not found: {reference_image}")
                            continue
                            
                        print(f"Using reference image: {reference_image}")
                    except Exception as e:
                        print(f"Error parsing reference command: {e}")
                        continue
                
                
                if 'save:' in user_input:
                    try:
                        save_start = user_input.find('save:') + 5
                        save_end = user_input.find(' ', save_start)
                        if save_end == -1:
                            print("Please provide a prompt after the save filename")
                            continue
                        
                        output_filename = user_input[save_start:save_end].strip()
                        prompt = user_input[save_end:].strip()
                        
                        print(f"Will save as: {output_filename} (in session folder)")
                    except Exception as e:
                        print(f"Error parsing save command: {e}")
                        continue
                
                print(f"\nGenerating: {prompt}")
                
                
                if reference_image:
                    result = generator.generate_image_with_reference(prompt, reference_image, output_filename)
                else:
                    result = generator.generate_image_from_text(prompt, output_filename)
                
                if result["success"]:
                    print(f"{result['message']}")
                    if result.get('saved_path'):
                        print(f"Saved to: {result['saved_path']}")
                    
                    
                    try:
                        if result.get('image'):
                            result['image'].show()
                    except Exception as e:
                        print(f"Could not display image: {e}")
                else:
                    print(f"Generation failed: {result['error']}")
                    if result.get('raw_response'):
                        print(f"🔍 Raw response: {result['raw_response']}")
        
        elif choice in ['animation', '2']:
            
            print("\nANIMATION GENERATION")
            print(f"Animation will be saved to: {generator.session_folder}")
            print("Note: For animations, a colorpalette.png will be created from the first frame")
            print("and used consistently across all frames for better visual consistency.")
            
            
            while True:
                base_image_path = input("\nEnter path to base image: ").strip()
                if not base_image_path:
                    print("Please provide a base image path")
                    continue
                if not os.path.exists(base_image_path):
                    print(f"Base image not found: {base_image_path}")
                    continue
                break
            
            
            while True:
                try:
                    frames_input = input("Number of frames (2-30): ").strip()
                    if not frames_input:
                        print("Please provide number of frames")
                        continue
                    num_frames = int(frames_input)
                    if num_frames < 2 or num_frames > 30:
                        print("Number of frames should be between 2 and 30")
                        continue
                    break
                except ValueError:
                    print("Invalid number. Please enter a number between 2 and 30")
                    continue
            
            
            while True:
                animation_desc = input("Animation description: ").strip()
                if not animation_desc:
                    print("Please provide an animation description")
                    continue
                break
            
            
            frame_duration = 500  
            duration_input = input(f"Frame duration in milliseconds (default: {frame_duration}): ").strip()
            if duration_input:
                try:
                    frame_duration = int(duration_input)
                except ValueError:
                    print(f"Invalid duration, using default: {frame_duration}ms")
            
            
            gif_output_filename = None
            output_input = input("GIF filename (press Enter for auto-generated): ").strip()
            if output_input:
                gif_output_filename = output_input
                if not gif_output_filename.endswith('.gif'):
                    gif_output_filename += '.gif'
            
            
            cleanup_input = input("Delete individual frames after creating GIF? (y/n) [default: n]: ").strip().lower()
            cleanup_frames = cleanup_input in ['y', 'yes']
            
            print(f"\nCreating animation...")
            print(f"Base image: {base_image_path}")
            print(f"Frames: {num_frames}")
            print(f"Animation: {animation_desc}")
            print(f"Frame duration: {frame_duration}ms")
            print(f"Cleanup frames: {cleanup_frames}")
            print(f"Output location: {generator.session_folder}")
            
            
            result = generator.generate_complete_animation(
                base_image_path,
                animation_desc,
                num_frames,
                gif_output_filename,  
                frame_duration,
                cleanup_frames
            )
            
            if result["success"]:
                print(f"\n{result['message']}")
                print(f"Animation saved: {result['gif_path']}")
                if result.get('frames_directory') and not cleanup_frames:
                    print(f"Individual frames: {result['frames_directory']}")
                if result['failed_frames']:
                    print(f"Failed frames: {result['failed_frames']}")
                
                
                if generator.pixelate_images and result.get('palette_image'):
                    print("Color palette created and applied consistently across all frames")
                
                
                try:
                    gif_image = Image.open(result['gif_path'])
                    print("Opening animation preview...")
                    gif_image.show()
                except Exception as e:
                    print(f"Could not display animation: {e}")
                    
            else:
                print(f"Animation failed: {result['error']}")
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()