"""
Main pipeline for synthetic data generation.
Orchestrates all components to generate training data.
"""

import random
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import os

from config_loader import ConfigLoader
from sheet_generator import SheetGenerator
from symbol_manager import SymbolManager
from placement_engine import PlacementEngine, PlacementInfo
from distortion_engine import DistortionEngine
from annotation_generator import AnnotationGenerator


class SyntheticDataPipeline:
    """Main pipeline for generating synthetic training data."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        
        # Initialize components
        self.sheet_generator = None
        self.symbol_manager = None
        self.placement_engine = None
        self.distortion_engine = None
        self.annotation_generator = None
        
        # Statistics
        self.generation_stats = {
            'total_images': 0,
            'total_symbols': 0,
            'failed_placements': 0,
            'class_distribution': {}
        }
    
    def initialize_components(self) -> None:
        """Initialize all pipeline components."""
        print("Initializing pipeline components...")
        
        # Get sheet dimensions
        sheet_width, sheet_height = self.config_loader.get_sheet_size_pixels()
        dpi = self.config['generation']['dpi']
        
        # Initialize sheet generator
        self.sheet_generator = SheetGenerator(sheet_width, sheet_height, dpi)
        self.sheet_generator.set_backgrounds_directory(self.config['paths']['backgrounds_dir'])
        
        # Initialize symbol manager
        self.symbol_manager = SymbolManager(self.config['paths']['symbols_dir'])
        self.symbol_manager.load_symbols()
        
        # Initialize placement engine
        placement_bounds = self.config_loader.get_placement_bounds()
        self.placement_engine = PlacementEngine(placement_bounds)
        self.placement_engine.set_collision_padding(
            self.config['symbol_placement'].get('collision_padding_pixels', 10)
        )
        
        # Initialize distortion engine
        self.distortion_engine = DistortionEngine(self.config['distortions'])
        
        # Initialize annotation generator
        self.annotation_generator = AnnotationGenerator(sheet_width, sheet_height)
        
        print("✅ All components initialized successfully")
    
    def select_symbols_for_image(self) -> List[Dict[str, Any]]:
        """
        Select symbols to place on a single image.
        
        Returns:
            List of symbol data dictionaries
        """
        density_range = self.config['symbol_placement']['density_range']
        num_symbols = random.randint(density_range[0], density_range[1])
        
        symbols_data = []
        all_symbols = self.symbol_manager.get_all_symbols()
        
        for _ in range(num_symbols):
            symbol = random.choice(all_symbols)
            symbol_size = symbol.get_size()
            
            symbols_data.append({
                'name': symbol.name,
                'class_name': symbol.get_class_name(),
                'size': symbol_size,
                'symbol_obj': symbol
            })
        
        return symbols_data
    
    def place_symbols_on_sheet(self, symbols_data: List[Dict[str, Any]]) -> List[PlacementInfo]:
        """
        Place symbols on the sheet with distortions.
        
        Args:
            symbols_data: List of symbol data
            
        Returns:
            List of successful placements
        """
        self.placement_engine.clear_placements()
        
        successful_placements = []
        
        for symbol_data in symbols_data:
            # Load the original symbol image
            symbol_img = symbol_data['symbol_obj'].load_image()
            
            # Generate distortion parameters
            distortion_params = self.distortion_engine.generate_distortion_params()
            
            # Apply distortions step by step
            distorted_img = symbol_img.copy()
            
            # Apply geometric transforms (rotation, scaling, flipping)
            distorted_img = self.distortion_engine.apply_geometric_transforms(distorted_img, distortion_params)
            
            # Apply blur effects
            distorted_img = self.distortion_engine.apply_blur_effects(distorted_img, distortion_params)
            
            # Apply noise effects
            distorted_img = self.distortion_engine.apply_noise_effects(distorted_img, distortion_params)
            
            # Apply line effects
            distorted_img = self.distortion_engine.apply_line_effects(distorted_img, distortion_params)
            
            # Update symbol data with distorted image and parameters
            symbol_data['distorted_size'] = distorted_img.size
            symbol_data['distorted_image'] = distorted_img
            symbol_data['distortion_params'] = distortion_params
            
            # Place the symbol (no additional transformations needed since they're already applied)
            placement = self.placement_engine.place_symbol_random(
                symbol_name=symbol_data['name'],
                class_name=symbol_data['class_name'],
                symbol_size=symbol_data['distorted_size'],
                scale_range=(1.0, 1.0),  # Scaling already applied in distortion
                rotation_range=(0.0, 0.0),  # Rotation already applied in distortion
                allow_flip=False,  # Flipping already applied in distortion
                check_collisions=not self.config['symbol_placement']['allow_overlaps']
            )
            
            if placement:
                # Update placement with distortion info for annotations
                placement.rotation = distortion_params.rotation
                placement.scale = distortion_params.scale
                placement.flipped_h = distortion_params.flip_horizontal
                placement.flipped_v = distortion_params.flip_vertical
                
                successful_placements.append(placement)
                symbol_data['placement'] = placement
            else:
                self.generation_stats['failed_placements'] += 1
        
        return successful_placements
    
    def render_final_image(self, symbols_data: List[Dict[str, Any]], 
                          placements: List[PlacementInfo]) -> Image.Image:
        """
        Render the final image with all placed symbols.
        
        Args:
            symbols_data: List of symbol data with placements
            placements: List of placement information
            
        Returns:
            Final rendered image
        """
        # Create base sheet
        use_background = len(self.sheet_generator.available_backgrounds) > 0
        sheet = self.sheet_generator.generate_sheet(
            use_background=use_background,
            add_debug_elements=self.config['output'].get('include_debug_info', False)
        )
        
        # Place each symbol on the sheet
        for symbol_data in symbols_data:
            if 'placement' not in symbol_data:
                continue  # Skip symbols that couldn't be placed
            
            placement = symbol_data['placement']
            distorted_img = symbol_data['distorted_image']
            
            # Paste the symbol onto the sheet
            if distorted_img.mode == 'RGBA':
                sheet.paste(distorted_img, (placement.x, placement.y), distorted_img)
            else:
                sheet.paste(distorted_img, (placement.x, placement.y))
        
        return sheet
    
    def generate_single_image(self, image_index: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generate a single synthetic image with annotations.
        
        Args:
            image_index: Index of the image being generated
            
        Returns:
            Tuple of (image, annotation_dict)
        """
        # Select symbols for this image
        symbols_data = self.select_symbols_for_image()
        
        # Place symbols on sheet
        placements = self.place_symbols_on_sheet(symbols_data)
        
        # Render final image
        final_image = self.render_final_image(symbols_data, placements)
        
        # Generate annotations
        image_filename = f"synthetic_{image_index:06d}.png"
        annotation = self.annotation_generator.create_annotation(
            image_filename, placements, 
            include_metadata=self.config['output'].get('include_debug_info', False)
        )
        
        # Update statistics
        self.generation_stats['total_images'] += 1
        self.generation_stats['total_symbols'] += len(placements)
        
        for placement in placements:
            class_name = placement.class_name
            self.generation_stats['class_distribution'][class_name] = \
                self.generation_stats['class_distribution'].get(class_name, 0) + 1
        
        return final_image, annotation
    
    def save_outputs(self, image: Image.Image, annotation: Dict[str, Any], 
                    image_index: int) -> None:
        """
        Save generated image and annotation to disk.
        
        Args:
            image: Generated image
            annotation: Annotation dictionary
            image_index: Index of the image
        """
        output_dir = Path(self.config['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        image_filename = f"synthetic_{image_index:06d}.png"
        image_path = output_dir / image_filename
        image.save(image_path, format='PNG')
        
        # Save annotation
        annotation_filename = f"synthetic_{image_index:06d}.json"
        annotation_path = output_dir / annotation_filename
        
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
    
    def generate_batch(self, num_images: int = None) -> None:
        """
        Generate a batch of synthetic images.
        
        Args:
            num_images: Number of images to generate. If None, uses config value.
        """
        if num_images is None:
            num_images = self.config['generation']['num_images']
        
        print(f"Generating {num_images} synthetic images...")
        
        # Initialize components if not already done
        if self.sheet_generator is None:
            self.initialize_components()
        
        # Generate images
        for i in range(num_images):
            try:
                # Generate single image
                image, annotation = self.generate_single_image(i)
                
                # Save outputs
                self.save_outputs(image, annotation, i)
                
                # Progress update
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"Generated {i + 1}/{num_images} images")
                
            except Exception as e:
                print(f"Error generating image {i}: {e}")
                continue
        
        print(f"✅ Batch generation completed!")
        self.print_generation_statistics()
    
    def print_generation_statistics(self) -> None:
        """Print statistics about the generation process."""
        stats = self.generation_stats
        
        print("\n📊 Generation Statistics:")
        print(f"   Total images generated: {stats['total_images']}")
        print(f"   Total symbols placed: {stats['total_symbols']}")
        print(f"   Failed placements: {stats['failed_placements']}")
        
        if stats['total_symbols'] > 0:
            avg_symbols = stats['total_symbols'] / stats['total_images']
            print(f"   Average symbols per image: {avg_symbols:.1f}")
        
        # Show top classes
        if stats['class_distribution']:
            print("\n🏷️  Top symbol classes:")
            sorted_classes = sorted(stats['class_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:10]:
                print(f"   {class_name}: {count}")
    
    def validate_setup(self) -> List[str]:
        """
        Validate the pipeline setup and return any issues.
        
        Returns:
            List of validation issues
        """
        issues = []
        
        try:
            # Check if components can be initialized
            self.initialize_components()
            
            # Check symbol manager
            if len(self.symbol_manager.get_all_symbols()) == 0:
                issues.append("No symbols loaded")
            
            # Check output directory
            output_dir = Path(self.config['paths']['output_dir'])
            if not output_dir.exists():
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create output directory: {e}")
            
            # Validate symbol loading
            symbol_issues = self.symbol_manager.validate_symbols()
            issues.extend(symbol_issues[:5])  # Limit to first 5 issues
            
        except Exception as e:
            issues.append(f"Component initialization failed: {e}")
        
        return issues


def main():
    """Main entry point for the pipeline."""
    print("🚀 Starting Synthetic Data Generation Pipeline")
    
    # Create pipeline
    pipeline = SyntheticDataPipeline()
    
    # Validate setup
    issues = pipeline.validate_setup()
    if issues:
        print("⚠️  Setup issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return
    
    # Generate batch
    pipeline.generate_batch()


if __name__ == "__main__":
    main() 