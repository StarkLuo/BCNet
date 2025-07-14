# Quantitative Evaluation Framework for Edge-Guided Image Generation

## Documentation

### Research Background
This framework quantitatively evaluates edge-guided image generation using ControlNet+diffusion models. By extracting edges as guidance to generate new images, it calculates PSNR, SSIM and LPIPS metrics between generated and original images to assess the effectiveness of edge representations.

### File Structure
```
project_dir/
├── gt/                  # Original images (20 diverse styles)
│   ├── 1.jpg            
│   ├── 2.jpg            
│   └── ...              
├── edges/               # Edge extraction results (optional)
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── method1/             # Generation method1 results
│   ├── 1/               # 8 generated images per original
│   │   ├── gen_1.png    
│   │   ├── gen_2.png
│   │   └── ... (8 total)
│   ├── 2/
│   └── ...
└── eval.py              # Evaluation script
```

### Key Features
- Accounts for stylization differences (expect lower metric values)
- 8 generations per image to reduce randomness
- Reports mean±std of all quality metrics