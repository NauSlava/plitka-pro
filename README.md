# 🧱 Plitka Pro - AI-Powered Rubber Tile Generator

**Professional AI model for generating photorealistic rubber tile images with precise color control and geometric patterns.**

## 🚀 Features

- **🎨 Precise Color Control**: Generate tiles with exact color percentages (60% black, 40% white)
- **🔧 ControlNet Integration**: Geometric pattern control with Canny, Lineart, and SoftEdge
- **⚡ High Performance**: Optimized SDXL pipeline with LoRA fine-tuning
- **🔄 Flexible Architecture**: Support for both ControlNet and base SDXL generation
- **💾 Memory Efficient**: Lazy loading architecture for optimal GPU memory usage

## 🏗️ Architecture

- **Base Model**: Stable Diffusion XL (SDXL) 1.0
- **Fine-tuning**: LoRA (Low-Rank Adaptation) + Textual Inversion
- **Control**: ControlNet for geometric pattern generation
- **Optimization**: CUDA memory management and parallel processing

## 📦 Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- Docker (for containerized deployment)

### Local Setup
```bash
git clone https://github.com/papaandrey/plitka-pro.git
cd plitka-pro
pip install -r requirements.txt
```

### Docker Deployment
```bash
# Build the container
cog build

# Run locally
cog predict -f input.json

# Deploy to Replicate
cog push r8.im/username/plitka-pro
```

## 🎯 Usage

### API Request Format
```json
{
  "params_json": "{\"colors\":[{\"name\":\"black\",\"proportion\":60},{\"name\":\"white\",\"proportion\":40}],\"angle\":0,\"quality\":\"standard\",\"seed\":42}"
}
```

### Supported Parameters
- **colors**: Array of color objects with name and proportion
- **angle**: Rotation angle (0°, 45°, 90°, etc.)
- **quality**: "preview", "standard", "high"
- **overrides**: Custom parameters (use_controlnet, guidance_scale, etc.)

## 🔧 Configuration

### Model Files
- LoRA weights: `ohwx_rubber_tile_lora.safetensors`
- Textual Inversion: `ohwx_rubber_tile_ti.safetensors`
- ControlNet models: Canny, Lineart, SoftEdge

### Environment Variables
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_LAUNCH_BLOCKING=0
```

## 📊 Performance

- **Startup Time**: ~30-45 seconds (lazy loading)
- **Generation Time**: 15-60 seconds depending on quality
- **Memory Usage**: 40-50% reduction with lazy loading
- **VRAM Requirements**: 14GB+ for full pipeline

## 🧪 Testing

### Test Scenarios
1. **Color Accuracy**: Verify correct color percentages
2. **ControlNet Override**: Test pipeline switching
3. **Memory Management**: Validate VRAM optimization
4. **Quality Levels**: Compare preview/standard/high

### Run Tests
```bash
python test_parsing.py
python test_app.py
```

## 📈 Version History

- **v4.2.9**: Lazy Loading Memory Optimization
- **v4.2.8**: Optimized Dual Pipeline Architecture
- **v4.2.7**: Initial ControlNet Integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/papaandrey/plitka-pro/issues)
- **Documentation**: [docs/](docs/) folder
- **Testing Guide**: [TESTING_GUIDE_v4.2.8.md](TESTING_GUIDE_v4.2.8.md)

## 🏆 Acknowledgments

- Stable Diffusion XL team
- ControlNet developers
- ComfyUI community
- Replicate platform

---

**Made with ❤️ for professional tile design and AI innovation**
