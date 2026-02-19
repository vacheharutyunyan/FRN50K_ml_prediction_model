#!/usr/bin/env python3
"""
Training script for baseline models.

This script demonstrates how to structure a complete ML pipeline
from data loading to model evaluation. Students can use this as a
template for their own experiments.

Usage:
    python scripts/train_baseline.py --model linear
    python scripts/train_baseline.py --model random_forest --config config/custom_config.yaml
"""

import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path so we can import our modules
import sys
sys.path.append('src')

from data.data_loader import FreshRetailDataLoader
from data.feature_engineering import FeatureEngineer
from models.baseline.linear_models import LinearForecastingModel
from models.baseline.tree_models import TreeForecastingModel
from models.baseline.naive_models import NaiveForecaster
from evaluate.metrics import ForecastingMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(description='Train baseline forecasting models')
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['naive', 'linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting'],
        default='linear',
        help='Model type to train'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/trained',
        help='Directory to save trained model'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size for faster experimentation (default: use all data)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for this experiment (default: auto-generate)'
    )
    
    return parser.parse_args()

def load_and_prepare_data(config: dict, sample_size = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data for training.
    
    Teaching note: This function demonstrates the complete data preparation pipeline
    that students should understand and be able to modify.
    """
    logger.info("Loading data...")
    
    # Initialize data loader with provided config
    data_loader = FreshRetailDataLoader(config=config)
    train_data, eval_data = data_loader.load_data()
    
    # Sample data if requested (useful for quick experiments)
    if sample_size and len(train_data) > sample_size:
        logger.info(f"Sampling {sample_size} rows from training data for faster experimentation")
        train_data = train_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Print data summary for students
    summary = data_loader.get_data_summary()
    logger.info("Data Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    # Feature engineering
    logger.info("Starting feature engineering...")
    feature_engineer = FeatureEngineer(config['preprocessing'])
    
    # Apply feature engineering to both train and eval data
    train_data_fe = feature_engineer.engineer_all_features(train_data)
    eval_data_fe = feature_engineer.engineer_all_features(eval_data)
    
    logger.info(f"Feature engineering complete. Features: {train_data_fe.shape[1]}")
    
    return train_data_fe, eval_data_fe

def create_model(model_name: str, config: dict):
    """
    Factory function to create the specified model.
    
    Teaching insight: Factory patterns make it easy to switch between
    different model types while keeping the training pipeline consistent.
    """
    model_config = config['models'].get('baseline', {})
    
    if model_name == 'naive':
        return NaiveForecaster(
            strategy='seasonal',
            seasonal_period=24,
            config=model_config.get('naive', {})
        )
    
    elif model_name in ['linear', 'ridge', 'lasso']:
        return LinearForecastingModel(
            model_type=model_name,
            config=model_config.get('linear_regression', {})
        )
    
    elif model_name in ['random_forest', 'gradient_boosting']:
        return TreeForecastingModel(
            model_type=model_name,
            config=model_config.get('random_forest', {})
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_and_evaluate_model(model, train_data: pd.DataFrame, eval_data: pd.DataFrame, config: dict):
    """
    Complete training and evaluation pipeline.
    
    Teaching value: This shows students how to structure a complete ML experiment
    with proper train/validation splits, evaluation, and result logging.
    """
    target_col = config['data']['target_column']
    
    # Define features (exclude target and metadata columns)
    metadata_cols = ['store_id', 'product_id', 'city_id', 'dt', target_col]
    feature_cols = [col for col in train_data.columns if col not in metadata_cols]
    
    logger.info(f"Using {len(feature_cols)} features for training")
    logger.info(f"Feature sample: {feature_cols[:10]}...")
    
    # Prepare training data
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    
    # Prepare evaluation data
    X_eval = eval_data[feature_cols]
    y_eval = eval_data[target_col]
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Evaluation data shape: {X_eval.shape}")
    
    # Train the model
    logger.info(f"Training {model.model_name}...")
    start_time = datetime.now()
    
    try:
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Generate predictions
    logger.info("Generating predictions...")
    try:
        train_predictions = model.predict(X_train)
        eval_predictions = model.predict(X_eval)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
    
    # Evaluate performance
    logger.info("Evaluating model performance...")
    
    # Training metrics
    train_metrics = ForecastingMetrics.calculate_all_metrics(y_train.values, train_predictions)
    logger.info("Training Metrics:")
    for metric, value in train_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Evaluation metrics
    eval_metrics = ForecastingMetrics.calculate_all_metrics(y_eval.values, eval_predictions, y_train.values)
    logger.info("Evaluation Metrics:")
    for metric, value in eval_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Feature importance (if available)
    feature_importance = model.get_feature_importance()
    if feature_importance:
        logger.info("Top 10 Most Important Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            logger.info(f"  {feature}: {importance:.4f}")
    
    # Create results summary
    results = {
        'model_name': model.model_name,
        'training_time_seconds': training_time,
        'train_metrics': train_metrics,
        'eval_metrics': eval_metrics,
        'feature_importance': feature_importance,
        'model_config': model.config,
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_eval_samples': len(X_eval)
    }
    
    return model, results

def save_results(model, results: dict, output_dir: str, experiment_name: str):
    """
    Save model and results to disk.
    
    Teaching point: Always save your experiments! This enables reproducibility
    and comparison between different approaches.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"{experiment_name}_{timestamp}.joblib"
    model_path = output_path / model_filename
    model.save_model(str(model_path))
    
    # Save results
    results_filename = f"{experiment_name}_{timestamp}_results.yaml"
    results_path = output_path / results_filename
    
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Results saved to: {results_path}")
    
    return str(model_path), str(results_path)

def main():
    """Main training pipeline."""
    args = parse_arguments()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"baseline_{args.model}_{timestamp}"
    
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Configuration: {args.config}")
    
    try:
        # Load and prepare data
        train_data, eval_data = load_and_prepare_data(config, args.sample_size)
        
        # Create model
        model = create_model(args.model, config)
        
        # Train and evaluate
        trained_model, results = train_and_evaluate_model(model, train_data, eval_data, config)
        
        # Save results
        model_path, results_path = save_results(trained_model, results, args.output_dir, args.experiment_name)
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Key Results:")
        logger.info(f"  Evaluation MAE: {results['eval_metrics']['MAE']:.4f}")
        logger.info(f"  Evaluation RMSE: {results['eval_metrics']['RMSE']:.4f}")
        logger.info(f"  Evaluation MAPE: {results['eval_metrics']['MAPE']:.2f}%")
        logger.info(f"  Model Bias: {results['eval_metrics']['Bias']:.4f}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()