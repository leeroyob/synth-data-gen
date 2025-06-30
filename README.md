# Synthetic Data Generator for Engineering Symbols

A powerful tool for generating synthetic training data from engineering symbols. Takes PNG symbol images and creates realistic synthetic datasets by placing them randomly on engineering sheets with various distortions to simulate hand-drawn/scanned conditions.

## Features

- **Multi-format Support**: Generates images with JSON, YOLO, and COCO format annotations
- **Realistic Distortions**: Applies rotation, scaling, blur, and other effects to simulate real-world conditions
- **Collision Detection**: Prevents symbol overlaps with configurable padding
- **Flexible Configuration**: JSON-based configuration for all parameters
- **Batch Processing**: Generate hundreds of images efficiently
- **Progress Tracking**: Real-time progress updates and statistics

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Images**:
   ```bash
   python main.py --num-images 10
   ```

3. **Check Output**:
   Generated images and annotations will be in the `output/` directory.

## Usage

### Basic Commands

```bash
# Generate default number of images (100)
python main.py

# Generate specific number of images
python main.py --num-images 50

# Use custom configuration
python main.py --config my_config.json

# Validate setup without generating
python main.py --validate-only

# Enable verbose output
python main.py --verbose
```

### Configuration

Edit `config/config.json` to customize:

- **Image count and DPI**: Control output resolution and quantity
- **Symbol placement**: Density, collision detection, margins
- **Distortions**: Rotation, scaling, blur, noise effects
- **Output format**: PNG quality, annotation metadata

### Project Structure

```
synth-data-gen/
├── src/                    # Core components
├── symbols/               # Input symbol images (PNG)
├── backgrounds/           # Optional background images  
├── output/               # Generated images and annotations
├── config/               # Configuration files
├── main.py              # CLI entry point
└── requirements.txt     # Dependencies
```

## Generated Output

Each run produces:
- **Images**: High-resolution PNG files (default: 6600x10200 pixels)
- **Annotations**: JSON files with bounding boxes, classes, and metadata
- **Statistics**: Symbol distribution and placement metrics

## Symbol Format

Symbols should be:
- PNG format (transparent or non-transparent backgrounds supported)
- Reasonable resolution (not too small/large)
- Placed in the `symbols/images/` directory

**Background Handling**:
- Transparent backgrounds are preserved
- Non-transparent backgrounds: anything that is not black (or very close to black) is automatically removed and made transparent
- This allows symbols with white or colored backgrounds to be processed correctly

The system automatically:
- Extracts class names from filenames
- Handles transparency correctly (removes non-black backgrounds)
- Validates symbol integrity
- Applies realistic distortions including rotation, scaling, blur, and noise effects

## Technical Details

### Sheet Specifications
- **Size**: 34" × 22" (landscape orientation)
- **Resolution**: 300 DPI (configurable)
- **Placement Area**: Bottom half only with ¾" margins
- **Output Dimensions**: 10200 × 6600 pixels

### Distortion Engine
Applies realistic effects to simulate real-world conditions:
- Random rotation (0-360°)
- Scaling variations (0.4-0.6x for smaller symbols)
- Gaussian blur effects
- Optional perspective warping
- Noise and compression artifacts

### Annotation Format
```json
{
  "image": {
    "filename": "synthetic_000001.png",
    "width": 10200,
    "height": 6600
  },
  "annotations": [
    {
      "class": "hydrant",
      "bbox": {"x": 1234, "y": 5678, "width": 100, "height": 120},
      "center": {"x": 1284, "y": 5738},
      "rotation": 45.2,
      "scale": 1.05,
      "symbol_name": "hydrant_type_a"
    }
  ]
}
```

## Advanced Features

- **Background Images**: Place symbols on custom engineering sheet backgrounds
- **Debug Mode**: Add grid lines and margin indicators
- **Multiple Annotation Formats**: JSON, YOLO, COCO support
- **Distortion Control**: Fine-tune realism vs. training effectiveness
- **Collision Detection**: Configurable symbol spacing and overlap prevention

## Requirements

- Python 3.8+
- PIL/Pillow for image processing
- OpenCV for advanced transformations
- NumPy for array operations
- scikit-image for additional filters

## Project Status

✅ **Completed Features**:
- Core pipeline with all major components
- Symbol loading and management (142 symbols loaded)
- Sheet generation with configurable dimensions
- Placement engine with collision detection
- Basic distortion engine (rotation, scaling, blur)
- Annotation generation (JSON, YOLO, COCO formats)
- CLI interface with validation
- Comprehensive configuration system

🔄 **Future Enhancements**:
- Advanced distortion effects (perspective warping, noise, line roughness)
- Visualization tools for bounding box display
- Additional annotation format support
- Performance optimizations for large batches

## License

MIT License - see LICENSE file for details.