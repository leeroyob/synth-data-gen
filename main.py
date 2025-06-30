#!/usr/bin/env python3
"""
Main CLI entry point for the Synthetic Data Generator.
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.main_pipeline import SyntheticDataPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from engineering symbols",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Generate default number of images
  python main.py --num-images 50          # Generate 50 images
  python main.py --config custom.json     # Use custom configuration
  python main.py --validate-only          # Only validate setup, don't generate
        """
    )
    
    parser.add_argument(
        '--num-images', '-n',
        type=int,
        help='Number of images to generate (overrides config)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (default: config/config.json)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate the setup without generating images'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    print("🎨 Synthetic Data Generator for Engineering Symbols")
    print("=" * 55)
    
    try:
        # Create pipeline
        if args.verbose:
            print(f"Loading configuration from: {args.config or 'config/config.json'}")
        
        pipeline = SyntheticDataPipeline(args.config)
        
        # Validate setup
        print("Validating setup...")
        issues = pipeline.validate_setup()
        
        if issues:
            print("❌ Setup validation failed:")
            for issue in issues:
                print(f"   • {issue}")
            print("\nPlease fix these issues before running the generator.")
            return 1
        
        print("✅ Setup validation passed!")
        
        # If only validating, exit here
        if args.validate_only:
            print("Validation complete. Use without --validate-only to generate images.")
            return 0
        
        # Generate images
        if args.num_images:
            print(f"Generating {args.num_images} images (overriding config)...")
            pipeline.generate_batch(args.num_images)
        else:
            print("Generating images using configuration settings...")
            pipeline.generate_batch()
        
        print("\n🎉 Generation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 