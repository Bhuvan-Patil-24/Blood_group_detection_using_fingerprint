"""
Unified Pipeline for Fingerprint Blood Group Detection
Complete end-to-end system: Image → Validation → Blood Group Prediction
"""

import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from preprocessing.image_processor import ImageProcessor
from feature_extraction.feature_extractor import FingerprintFeatureExtractor
from feature_extraction.fingerprint_classifier import FingerprintClassifier
from model.cnn_model import BloodGroupCNN
# from model.resNet_model import BloodGroupResNet50


class FingerprintBloodGroupPipeline:
    """
    Production-ready pipeline integrating:
    1. Image Preprocessing
    2. Feature Extraction
    3. Fingerprint Validation (ML Classifier)
    4. Blood Group Prediction (CNN)
    """
    
    def __init__(self, 
                 validator_model_path='saved_models/fingerprint_validator.joblib',
                 cnn_model_path='saved_models/bloodgroup_cnn.keras',
                 target_size=(128, 128),
                 confidence_threshold=0.5):
        """
        Initialize pipeline with pre-trained models
        
        Args:
            validator_model_path (str): Path to fingerprint validator .joblib model
            cnn_model_path (str): Path to blood group CNN .h5 model
            target_size (tuple): Image preprocessing size (width, height)
            confidence_threshold (float): Minimum confidence for valid predictions
        """
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        
        print("=" * 70)
        print("INITIALIZING FINGERPRINT BLOOD GROUP PIPELINE")
        print("=" * 70)
        
        # Initialize preprocessor
        print("\n[1/4] Initializing Image Preprocessor...")
        self.preprocessor = ImageProcessor(target_size=target_size)
        print("  ✓ Preprocessor ready")
        
        # Initialize feature extractor
        print("\n[2/4] Initializing Feature Extractor...")
        self.feature_extractor = FingerprintFeatureExtractor(target_size=target_size)
        print("  ✓ Feature extractor ready")
        
        # Load validator model
        print("\n[3/4] Loading Fingerprint Validator Model...")
        self.validator = FingerprintClassifier()
        try:
            self.validator.load_model(validator_model_path)
            print(f"  ✓ Validator loaded: {self.validator.best_model_name}")
        except Exception as e:
            print(f"  ✗ Warning: Could not load validator: {e}")
            print(f"    Pipeline will skip validation step")
            self.validator = None
        
        # Load CNN model
        print("\n[4/4] Loading Blood Group CNN Model...")
        self.cnn = BloodGroupCNN(input_shape=(target_size[0], target_size[1], 1))
        # self.cnn = BloodGroupResNet50(input_shape=(224,224,3))
        try:
            self.cnn.load_model(cnn_model_path)
            print(f"  ✓ CNN loaded successfully")
        except Exception as e:
            print(f"  ✗ Warning: Could not load CNN: {e}")
            print(f"    Blood group prediction will not be available")
            self.cnn = None
        
        print("\n" + "=" * 70)
        print("✓ PIPELINE INITIALIZATION COMPLETE")
        print("=" * 70)
    
    def predict(self, image_path, return_detailed=False):
        """
        Complete prediction pipeline
        
        Args:
            image_path (str): Path to input fingerprint image
            return_detailed (bool): If True, return detailed step-by-step results
            
        Returns:
            dict: Prediction result in JSON format
        """
        print("\n" + "=" * 70)
        print(f"PROCESSING: {Path(image_path).name}")
        print("=" * 70)
        
        result = {
            'status': False,
            'message': '',
            'image_path': str(image_path),
            'blood_group': None,
            'confidence': None
        }
        
        detailed_info = {
            'preprocessing': None,
            'feature_extraction': None,
            'validation': None,
            'blood_group_prediction': None
        }
        
        try:
            # Step 1: Validate input file
            print("\n[Step 1/5] Validating input file...")
            if not self._validate_input(image_path):
                result['message'] = "Invalid image file path or format"
                print(f"  ✗ {result['message']}")
                return result
            print("  ✓ Input file validated")
            
            # Step 2: Preprocess image
            print("\n[Step 2/5] Preprocessing fingerprint image...")
            try:
                preprocessed_image = self.preprocessor.preprocess_fingerprint(image_path)
                print(f"  ✓ Preprocessed: {preprocessed_image.shape}, "
                      f"dtype={preprocessed_image.dtype}, "
                      f"range=[{preprocessed_image.min():.3f}, {preprocessed_image.max():.3f}]")
                detailed_info['preprocessing'] = {
                    'success': True,
                    'shape': preprocessed_image.shape,
                    'dtype': str(preprocessed_image.dtype)
                }
            except Exception as e:
                result['message'] = f"Preprocessing failed: {str(e)}"
                print(f"  ✗ {result['message']}")
                detailed_info['preprocessing'] = {'success': False, 'error': str(e)}
                return self._format_result(result, detailed_info, return_detailed)
            
            # Step 3: Extract features for validation
            print("\n[Step 3/5] Extracting features for validation...")
            try:
                features = self.feature_extractor.extract_features(preprocessed_image)
                print(f"  ✓ Features extracted: {len(features)} dimensions")
                
                # Check for NaN or Inf
                if np.isnan(features).any() or np.isinf(features).any():
                    print("  ⚠ Warning: Features contain NaN/Inf values, cleaning...")
                    features = self._clean_features(features)
                
                detailed_info['feature_extraction'] = {
                    'success': True,
                    'feature_count': len(features),
                    'has_nan': bool(np.isnan(features).any()),
                    'has_inf': bool(np.isinf(features).any())
                }
            except Exception as e:
                result['message'] = f"Feature extraction failed: {str(e)}"
                print(f"  ✗ {result['message']}")
                detailed_info['feature_extraction'] = {'success': False, 'error': str(e)}
                return self._format_result(result, detailed_info, return_detailed)
            
            # Step 4: Validate fingerprint (if validator is loaded)
            if self.validator is not None:
                print("\n[Step 4/5] Validating fingerprint authenticity...")
                try:
                    is_fingerprint = self.validator.predict(features)[0]
                    fingerprint_confidence = self.validator.predict_proba(features)[0]
                    
                    detailed_info['validation'] = {
                        'is_valid': bool(is_fingerprint == 1),
                        'confidence': float(fingerprint_confidence),
                        'threshold': self.confidence_threshold
                    }
                    
                    if is_fingerprint == 0:
                        result['message'] = "Invalid fingerprint image - not a fingerprint"
                        result['validation_confidence'] = float(1 - fingerprint_confidence)
                        print(f"  ✗ Not a fingerprint (confidence: {(1-fingerprint_confidence)*100:.2f}%)")
                        return self._format_result(result, detailed_info, return_detailed)
                    
                    if fingerprint_confidence < self.confidence_threshold:
                        result['message'] = f"Low confidence fingerprint (confidence: {fingerprint_confidence:.2f})"
                        result['validation_confidence'] = float(fingerprint_confidence)
                        print(f"  ✗ Low confidence: {fingerprint_confidence*100:.2f}%")
                        return self._format_result(result, detailed_info, return_detailed)
                    
                    print(f"  ✓ Valid fingerprint (confidence: {fingerprint_confidence*100:.2f}%)")
                    
                except Exception as e:
                    print(f"  ⚠ Validation error: {e}, continuing to prediction...")
                    detailed_info['validation'] = {'success': False, 'error': str(e)}
            else:
                print("\n[Step 4/5] Skipping validation (validator not loaded)...")
                detailed_info['validation'] = {'skipped': True}
            
            # Step 5: Predict blood group with CNN
            print("\n[Step 5/5] Predicting blood group...")
            if self.cnn is None:
                result['message'] = "CNN model not loaded, cannot predict blood group"
                print(f"  ✗ {result['message']}")
                return self._format_result(result, detailed_info, return_detailed)
            
            try:
                blood_group_result = self.cnn.predict_blood_group(preprocessed_image)
                
                result['status'] = True
                result['blood_group'] = blood_group_result['blood_group']
                result['confidence'] = blood_group_result['confidence']
                result['all_probabilities'] = blood_group_result['all_probabilities']
                result['message'] = "Prediction successful"
                
                detailed_info['blood_group_prediction'] = {
                    'success': True,
                    'predicted_class': blood_group_result['blood_group'],
                    'confidence': blood_group_result['confidence'],
                    'all_probabilities': blood_group_result['all_probabilities']
                }
                
                print(f"  ✓ Blood Group: {result['blood_group']}")
                print(f"  ✓ Confidence: {result['confidence']*100:.2f}%")
                
            except Exception as e:
                result['message'] = f"Blood group prediction failed: {str(e)}"
                print(f"  ✗ {result['message']}")
                detailed_info['blood_group_prediction'] = {'success': False, 'error': str(e)}
                return self._format_result(result, detailed_info, return_detailed)
            
        except Exception as e:
            result['message'] = f"Pipeline error: {str(e)}"
            print(f"\n✗ Unexpected error: {e}")
            return self._format_result(result, detailed_info, return_detailed)
        
        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETE")
        print("=" * 70)
        
        return self._format_result(result, detailed_info, return_detailed)
    
    def _validate_input(self, image_path):
        """Validate input file exists and has valid extension"""
        path = Path(image_path)
        
        if not path.exists():
            return False
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        if path.suffix.lower() not in valid_extensions:
            return False
        
        return True
    
    def _clean_features(self, features):
        """Clean feature vector by replacing NaN/Inf with 0"""
        features = np.array(features)
        features[np.isnan(features)] = 0.0
        features[np.isinf(features)] = 0.0
        return features
    
    def _format_result(self, result, detailed_info, return_detailed):
        """Format final result based on detail level"""
        if return_detailed:
            result['detailed_info'] = detailed_info
        
        return result
    
    def predict_batch(self, image_paths, save_results=None):
        """
        Process multiple images in batch
        
        Args:
            image_paths (list): List of image file paths
            save_results (str, optional): Path to save JSON results
            
        Returns:
            list: List of prediction results
        """
        print("\n" + "=" * 70)
        print(f"BATCH PROCESSING: {len(image_paths)} images")
        print("=" * 70)
        
        results = []
        
        for i, path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {Path(path).name}")
            result = self.predict(path, return_detailed=False)
            results.append(result)
        
        # Save results if requested
        if save_results:
            with open(save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {save_results}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 70)
        
        successful = sum(1 for r in results if r['status'])
        failed = len(results) - successful
        
        print(f"Total processed: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if successful > 0:
            blood_groups = [r['blood_group'] for r in results if r['status']]
            from collections import Counter
            counts = Counter(blood_groups)
            print("\nBlood Group Distribution:")
            for group, count in sorted(counts.items()):
                print(f"  {group}: {count}")
        
        return results
    
    def get_pipeline_info(self):
        """Get information about loaded models and configuration"""
        info = {
            'preprocessor': {
                'target_size': self.target_size,
                'status': 'loaded'
            },
            'feature_extractor': {
                'target_size': self.target_size,
                'status': 'loaded'
            },
            'validator': {
                'status': 'loaded' if self.validator else 'not loaded',
                'model_type': self.validator.best_model_name if self.validator else None
            },
            'cnn': {
                'status': 'loaded' if self.cnn and self.cnn.is_trained else 'not loaded',
                'input_shape': self.cnn.input_shape if self.cnn else None,
                'blood_groups': self.cnn.blood_groups if self.cnn else None
            },
            'confidence_threshold': self.confidence_threshold
        }
        
        return info


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fingerprint Blood Group Detection Pipeline'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to fingerprint image'
    )
    parser.add_argument(
        '--validator',
        type=str,
        default='models/validator.joblib',
        help='Path to validator model'
    )
    parser.add_argument(
        '--cnn',
        type=str,
        default='models/bloodgroup_cnn.h5',
        help='Path to CNN model'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Return detailed step-by-step results'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save result to JSON file'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FingerprintBloodGroupPipeline(
        validator_model_path=args.validator,
        cnn_model_path=args.cnn
    )
    
    # Process image
    result = pipeline.predict(args.image_path, return_detailed=args.detailed)
    
    # Print result
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(json.dumps(result, indent=2))
    
    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Result saved to: {args.output}")