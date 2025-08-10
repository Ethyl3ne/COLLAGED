# COLLAGED

Games require thematic, visual, and tonal consistency across environments, sprites, items, and music. Maintaining this while creating assets can be challenging, thus this paper introduces COLLAGED, a generative AI pipeline for creating stylistically coherent game assets. COLLAGED targets pixel art games, which benefit from AI generation due to their low-resolution, highly editable nature. This enables developers to rapidly produce prototype media across different asset types. Given a reference image, audio file, or text prompt, COLLAGED generates matching scenes, characters, and music while preserving thematic unity. The pipeline uses Gemini 2.5 Flash for prompt enhancement, Stable Image Core or Gemini 2.0 for images, and MelodyFlow for music. This system produces draft assets that require less manual refinement than creating assets from scratch, accelerating development workflows. 

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ethyl3ne/COLLAGED.git
cd COLLAGED
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up env

Obtain a Google API Key from here: https://aistudio.google.com/app/apikey

Obtain a Stable Diffusion API Key from here: https://platform.stability.ai/account/keys

Edit the .env file

## Usage

Run the main script:
```bash
python main.py
```

## Requirements

See `requirements.txt` for all dependencies.

