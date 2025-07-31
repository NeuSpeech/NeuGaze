# =============================================================================
# NeuSpeech Institute, NeuGaze Project
# Copyright (c) 2024 Yiqian Yang
#
# This code is part of the NeuGaze project developed at NeuSpeech Institute.
# Author: Yiqian Yang
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 
# International License. To view a copy of this license, visit:
# http://creativecommons.org/licenses/by-nc/4.0/
# =============================================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
from calculate_error_map import ErrorMapCalculator


class ProgressiveTrainingEvaluator:
    def __init__(self, screen_width_px=3072, screen_height_px=1920,
                 screen_width_mm=344.6, screen_height_mm=215.4,
                 regression_model_type='lassocv', feature_type='basic'):
        """
        Progressive Training Evaluator
        
        Args:
            screen_width_px: Screen width in pixels
            screen_height_px: Screen height in pixels
            screen_width_mm: Screen width in millimeters
            screen_height_mm: Screen height in millimeters
            regression_model_type: Regression model type
            feature_type: Feature type
        """
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px
        self.screen_width_mm = screen_width_mm
        self.screen_height_mm = screen_height_mm
        self.regression_model_type = regression_model_type
        self.feature_type = feature_type
        
        self.results = []  # 存储每次训练的结果
        
    def run_progressive_training(self, train_folders, test_folder, output_dir, include_kalman_analysis=False):
        """
        Run progressive training evaluation
        
        Args:
            train_folders: List of training data folders in order
            test_folder: Test data folder
            output_dir: Output directory
            include_kalman_analysis: Whether to include Kalman filtering analysis
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("Progressive Training Error Analysis Started")
        print("="*80)
        print(f"Training data folders: {train_folders}")
        print(f"Test data folder: {test_folder}")
        print(f"Output directory: {output_dir}")
        print(f"Regression model: {self.regression_model_type}")
        print(f"Feature type: {self.feature_type}")
        print()
        
        # Load test data
        test_calculator = ErrorMapCalculator(
            screen_width_px=self.screen_width_px,
            screen_height_px=self.screen_height_px,
            screen_width_mm=self.screen_width_mm,
            screen_height_mm=self.screen_height_mm,
            regression_model_type=self.regression_model_type,
            feature_type=self.feature_type
        )
        
        print("Loading test data...")
        test_data = test_calculator.load_calibration_data([test_folder])
        if not test_data:
            raise ValueError(f"Cannot load test data from {test_folder}")
        print(f"Test data loaded: {len(test_data)} data points")
        print()
        
        # Progressive training
        for i in range(len(train_folders)):
            current_train_folders = train_folders[:i+1]
            
            print(f"{'='*60}")
            print(f"Training {i+1} - Using {len(current_train_folders)} training datasets")
            print(f"{'='*60}")
            print(f"Current training folders: {current_train_folders}")
            
            # Create current training calculator
            calculator = ErrorMapCalculator(
                screen_width_px=self.screen_width_px,
                screen_height_px=self.screen_height_px,
                screen_width_mm=self.screen_width_mm,
                screen_height_mm=self.screen_height_mm,
                regression_model_type=self.regression_model_type,
                feature_type=self.feature_type
            )
            
            # Load current training data
            print("Loading training data...")
            train_data = calculator.load_calibration_data(current_train_folders)
            if not train_data:
                print(f"Warning: Cannot load training data from {current_train_folders}, skipping this training")
                continue
            
            # Train model
            print("Training model...")
            train_scores = calculator.train_model(train_data, test_data)
            
            # Calculate test errors
            print("Calculating test errors...")
            error_results = calculator.calculate_errors(test_data)
            
            # Create output directory for this training iteration
            iteration_output_dir = output_dir / f"iteration_{i+1:02d}_datasets_{len(current_train_folders)}"
            iteration_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate error maps and statistics plots
            print("Generating error distribution maps...")
            calculator.create_error_map(error_results, iteration_output_dir, unit='px')
            calculator.create_error_map(error_results, iteration_output_dir, unit='mm')
            
            # Save results
            calculator.save_results(error_results, iteration_output_dir, train_scores)
            
            # Save detailed information for this training iteration
            iteration_info = {
                'iteration': i + 1,
                'num_datasets': len(current_train_folders),
                'train_folders': current_train_folders,
                'test_folder': test_folder,
                'num_train_samples': len(train_data),
                'num_test_samples': len(test_data),
                'model_config': {
                    'regression_model_type': self.regression_model_type,
                    'feature_type': self.feature_type,
                },
                'train_score': train_scores[0] if train_scores else None,
                'test_score': train_scores[1] if train_scores else None,
                'error_statistics_px': error_results['statistics_px'],
                'error_statistics_mm': error_results['statistics_mm'],
                'output_dir': str(iteration_output_dir)
            }
            
            # Save iteration information
            with open(iteration_output_dir / 'iteration_info.json', 'w', encoding='utf-8') as f:
                json.dump(iteration_info, f, indent=4, ensure_ascii=False)
            
            # Store results for subsequent summarization
            self.results.append(iteration_info)
            
            # Print summary for this training
            calculator.print_summary(error_results, train_scores, "test")
            
            print(f"Training {i+1} completed, results saved to: {iteration_output_dir}")
            print()
        
        # Generate comprehensive analysis report
        self.generate_comprehensive_report(output_dir)
        
        # Run Kalman filtering analysis if requested
        if include_kalman_analysis:
            print("\nStarting Kalman filtering analysis...")
            self.run_kalman_analysis(test_folder, output_dir)
        
        print("="*80)
        print("Progressive Training Error Analysis Completed")
        print(f"All results saved to: {output_dir}")
        print("="*80)
    
    def generate_comprehensive_report(self, output_dir):
        """Generate comprehensive analysis report"""
        print("Generating comprehensive analysis report...")
        
        # Create comprehensive comparison plots
        self.create_comparison_plots(output_dir)
        
        # Create comprehensive report document
        self.create_summary_report(output_dir)
        
        # Save all results to JSON
        with open(output_dir / 'comprehensive_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
    
    def create_comparison_plots(self, output_dir):
        """Create comparison plots"""
        if not self.results:
            return
        
        # Extract data for plotting
        iterations = [r['iteration'] for r in self.results]
        train_scores = [r['train_score'] for r in self.results if r['train_score'] is not None]
        test_scores = [r['test_score'] for r in self.results if r['test_score'] is not None]
        
        # Pixel error statistics
        med_px = [r['error_statistics_px']['med'] for r in self.results]
        mae_x_px = [r['error_statistics_px']['mae_x'] for r in self.results]
        mae_y_px = [r['error_statistics_px']['mae_y'] for r in self.results]
        std_px = [r['error_statistics_px']['std'] for r in self.results]
        max_px = [r['error_statistics_px']['max'] for r in self.results]
        
        # Millimeter error statistics
        med_mm = [r['error_statistics_mm']['med'] for r in self.results]
        mae_x_mm = [r['error_statistics_mm']['mae_x'] for r in self.results]
        mae_y_mm = [r['error_statistics_mm']['mae_y'] for r in self.results]
        std_mm = [r['error_statistics_mm']['std'] for r in self.results]
        max_mm = [r['error_statistics_mm']['max'] for r in self.results]
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Progressive Training Results Comparison Analysis', fontsize=16, fontweight='bold')
        
        # R² Score Comparison
        ax = axes[0, 0]
        if train_scores and test_scores:
            ax.plot(iterations[:len(train_scores)], train_scores, 'o-', label='Training R²', linewidth=2, markersize=8)
            ax.plot(iterations[:len(test_scores)], test_scores, 's-', label='Testing R²', linewidth=2, markersize=8)
            ax.set_xlabel('Training Iteration (Number of Datasets)')
            ax.set_ylabel('R² Score')
            ax.set_title('Model Performance Comparison (R² Score)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(iterations)
        
        # Pixel Error - Average Error
        ax = axes[0, 1]
        ax.plot(iterations, med_px, 'o-', label='Mean Error (MED)', linewidth=2, markersize=8, color='red')
        ax.plot(iterations, mae_x_px, 's-', label='X-axis MAE', linewidth=2, markersize=6, color='blue')
        ax.plot(iterations, mae_y_px, '^-', label='Y-axis MAE', linewidth=2, markersize=6, color='green')
        ax.set_xlabel('Training Iteration (Number of Datasets)')
        ax.set_ylabel('Error (pixels)')
        ax.set_title('Pixel Error Trend Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(iterations)
        
        # Pixel Error - Standard Deviation and Maximum Error
        ax = axes[0, 2]
        ax.plot(iterations, std_px, 'o-', label='Standard Deviation', linewidth=2, markersize=8, color='orange')
        ax.plot(iterations, max_px, 's-', label='Maximum Error', linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('Training Iteration (Number of Datasets)')
        ax.set_ylabel('Error (pixels)')
        ax.set_title('Pixel Error Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(iterations)
        
        # Millimeter Error - Average Error
        ax = axes[1, 0]
        ax.plot(iterations, med_mm, 'o-', label='Mean Error (MED)', linewidth=2, markersize=8, color='red')
        ax.plot(iterations, mae_x_mm, 's-', label='X-axis MAE', linewidth=2, markersize=6, color='blue')
        ax.plot(iterations, mae_y_mm, '^-', label='Y-axis MAE', linewidth=2, markersize=6, color='green')
        ax.set_xlabel('Training Iteration (Number of Datasets)')
        ax.set_ylabel('Error (millimeters)')
        ax.set_title('Millimeter Error Trend Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(iterations)
        
        # Millimeter Error - Standard Deviation and Maximum Error
        ax = axes[1, 1]
        ax.plot(iterations, std_mm, 'o-', label='Standard Deviation', linewidth=2, markersize=8, color='orange')
        ax.plot(iterations, max_mm, 's-', label='Maximum Error', linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('Training Iteration (Number of Datasets)')
        ax.set_ylabel('Error (millimeters)')
        ax.set_title('Millimeter Error Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(iterations)
        
        # Training Data Size vs Performance Relationship
        ax = axes[1, 2]
        num_datasets = [r['num_datasets'] for r in self.results]
        train_samples = [r['num_train_samples'] for r in self.results]
        
        # Create dual y-axis
        ax2 = ax.twinx()
        line1 = ax.plot(iterations, med_px, 'o-', label='Pixel Mean Error', linewidth=2, markersize=8, color='red')
        line2 = ax2.plot(iterations, train_samples, 's-', label='Training Samples', linewidth=2, markersize=8, color='blue')
        
        ax.set_xlabel('Training Iteration (Number of Datasets)')
        ax.set_ylabel('Error (pixels)', color='red')
        ax2.set_ylabel('Training Samples', color='blue')
        ax.set_title('Training Data Size vs Error Relationship')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(iterations)
        
        # Merge legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create error improvement analysis
        self.create_improvement_analysis(output_dir)
    
    def run_kalman_analysis(self, test_folder, output_dir):
        """
        Run Kalman filtering analysis on test data
        
        Args:
            test_folder: Test data folder
            output_dir: Output directory for analysis results
        """
        if not self.results:
            print("No progressive training results available. Please run progressive training first.")
            return
        
        # Use the best performing model for Kalman analysis
        # best_iteration_idx = min(range(len(self.results)), 
        #                        key=lambda i: self.results[i]['error_statistics_px']['med'])
        # best_result = self.results[best_iteration_idx]
        first_result = self.results[0]  # 用第一次校准的结果
        print(f"\nRunning Kalman filtering analysis using best model from iteration {first_result['iteration']}...")
        
        # Create calculator with best configuration
        calculator = ErrorMapCalculator(
            screen_width_px=self.screen_width_px,
            screen_height_px=self.screen_height_px,
            screen_width_mm=self.screen_width_mm,
            screen_height_mm=self.screen_height_mm,
            regression_model_type=self.regression_model_type,
            feature_type=self.feature_type
        )
        
        # Load test data
        test_data = calculator.load_calibration_data([test_folder])
        if not test_data:
            raise ValueError(f"Cannot load test data from {test_folder}")
        
        # Load the best trained model
        best_model_path = Path(first_result['output_dir']) / 'trained_model.pkl'
        if best_model_path.exists():
            calculator.load_model(str(best_model_path))
        else:
            raise ValueError(f"Cannot find trained model at {best_model_path}")
        
        # Create Kalman analysis output directory
        kalman_output_dir = Path(output_dir) / 'kalman_filtering_analysis'
        
        # Run Kalman filtering analysis with configuration from config file
        calculator.analyze_kalman_filtering(
            calibration_data=test_data,
            output_dir=kalman_output_dir,
            dt=0.04,  # From config
            kalman_filter_std_measurement=4.0,  # Reduced for more stable predictions
            Q_coef=0.01,  # Reduced for smoother state transitions
            num_predictions_per_point=15  # Match the actual data points per calibration point
        )
        
        print(f"Kalman filtering analysis completed and saved to: {kalman_output_dir}")
    
    def create_improvement_analysis(self, output_dir):
        """Create error improvement analysis plots"""
        if len(self.results) < 2:
            return
        
        # Calculate relative improvement rates
        baseline_med_px = self.results[0]['error_statistics_px']['med']
        baseline_med_mm = self.results[0]['error_statistics_mm']['med']
        
        improvement_px = []
        improvement_mm = []
        iterations = []
        
        for i, result in enumerate(self.results):
            if i == 0:
                improvement_px.append(0)  # Baseline is 0 improvement
                improvement_mm.append(0)
            else:
                current_med_px = result['error_statistics_px']['med']
                current_med_mm = result['error_statistics_mm']['med']
                
                imp_px = (baseline_med_px - current_med_px) / baseline_med_px * 100
                imp_mm = (baseline_med_mm - current_med_mm) / baseline_med_mm * 100
                
                improvement_px.append(imp_px)
                improvement_mm.append(imp_mm)
            
            iterations.append(result['iteration'])
        
        # Plot improvement rate charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Error Improvement Rate Analysis', fontsize=16, fontweight='bold')
        
        # Pixel error improvement rate
        ax1.bar(iterations, improvement_px, color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.set_xlabel('Training Iteration (Number of Datasets)')
        ax1.set_ylabel('Error Improvement Rate (%)')
        ax1.set_title('Pixel Error Improvement Rate\n(Relative to Single Dataset Training)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(iterations)
        
        # Add value labels
        for i, v in enumerate(improvement_px):
            if v != 0:
                ax1.text(iterations[i], v + max(improvement_px) * 0.01, f'{v:.1f}%', 
                        ha='center', va='bottom', fontweight='bold')
        
        # Millimeter error improvement rate
        ax2.bar(iterations, improvement_mm, color='lightcoral', alpha=0.7, edgecolor='darkred')
        ax2.set_xlabel('Training Iteration (Number of Datasets)')
        ax2.set_ylabel('Error Improvement Rate (%)')
        ax2.set_title('Millimeter Error Improvement Rate\n(Relative to Single Dataset Training)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(iterations)
        
        # Add value labels
        for i, v in enumerate(improvement_mm):
            if v != 0:
                ax2.text(iterations[i], v + max(improvement_mm) * 0.01, f'{v:.1f}%', 
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, output_dir):
        """Create text summary report"""
        if not self.results:
            return
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("Progressive Training Error Analysis Comprehensive Report")
        report_lines.append("="*80)
        report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model configuration: {self.regression_model_type} ({self.feature_type} features)")
        report_lines.append(f"Screen configuration: {self.screen_width_px}×{self.screen_height_px}px")
        report_lines.append(f"Physical size: {self.screen_width_mm:.1f}×{self.screen_height_mm:.1f}mm")
        report_lines.append("")
        
        # Training configuration summary
        report_lines.append("Training Configuration Summary:")
        report_lines.append("-" * 40)
        for i, result in enumerate(self.results):
            report_lines.append(f"Iteration {result['iteration']}: Using {result['num_datasets']} datasets, "
                              f"{result['num_train_samples']} training samples")
        report_lines.append(f"Test data: {self.results[0]['num_test_samples']} samples")
        report_lines.append("")
        
        # Performance summary
        report_lines.append("Performance Summary:")
        report_lines.append("-" * 40)
        
        best_pixel_idx = min(range(len(self.results)), 
                           key=lambda i: self.results[i]['error_statistics_px']['med'])
        best_mm_idx = min(range(len(self.results)), 
                        key=lambda i: self.results[i]['error_statistics_mm']['med'])
        
        report_lines.append(f"Best pixel error: Iteration {self.results[best_pixel_idx]['iteration']} "
                          f"({self.results[best_pixel_idx]['error_statistics_px']['med']:.2f}px)")
        report_lines.append(f"Best millimeter error: Iteration {self.results[best_mm_idx]['iteration']} "
                          f"({self.results[best_mm_idx]['error_statistics_mm']['med']:.2f}mm)")
        
        # Calculate improvement situation
        if len(self.results) > 1:
            final_px = self.results[-1]['error_statistics_px']['med']
            initial_px = self.results[0]['error_statistics_px']['med']
            px_improvement = (initial_px - final_px) / initial_px * 100
            
            final_mm = self.results[-1]['error_statistics_mm']['med']
            initial_mm = self.results[0]['error_statistics_mm']['med']
            mm_improvement = (initial_mm - final_mm) / initial_mm * 100
            
            report_lines.append(f"Overall improvement rate (pixels): {px_improvement:.1f}%")
            report_lines.append(f"Overall improvement rate (millimeters): {mm_improvement:.1f}%")
        
        report_lines.append("")
        
        # Detailed results
        report_lines.append("Detailed Results:")
        report_lines.append("-" * 40)
        for result in self.results:
            report_lines.append(f"\nIteration {result['iteration']} (using {result['num_datasets']} datasets):")
            if result['train_score'] is not None:
                report_lines.append(f"  Training R² score: {result['train_score']:.4f}")
            if result['test_score'] is not None:
                report_lines.append(f"  Testing R² score: {result['test_score']:.4f}")
            
            px_stats = result['error_statistics_px']
            mm_stats = result['error_statistics_mm']
            
            report_lines.append(f"  Pixel error - MED: {px_stats['med']:.2f}px, "
                              f"X-MAE: {px_stats['mae_x']:.2f}px, "
                              f"Y-MAE: {px_stats['mae_y']:.2f}px")
            report_lines.append(f"  Millimeter error - MED: {mm_stats['med']:.2f}mm, "
                              f"X-MAE: {mm_stats['mae_x']:.2f}mm, "
                              f"Y-MAE: {mm_stats['mae_y']:.2f}mm")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # 保存报告
        with open(output_dir / 'comprehensive_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 同时打印到控制台
        print('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(description='Progressive training error analysis')
    parser.add_argument('--train_folders', nargs='+', required=True,
                       help='List of training data folder paths in order')
    parser.add_argument('--test_folder', required=True,
                       help='Test data folder path')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory')
    parser.add_argument('--regression_model', type=str, default='lassocv',
                       choices=['ridge', 'lasso', 'lassocv', 'random_forest', 'gradient_boosting', 'mlp'],
                       help='Regression model type (default: lassocv)')
    parser.add_argument('--feature_type', type=str, default='basic',
                       choices=['basic', 'mediapipe', 'combined'],
                       help='Feature type (default: basic)')
    parser.add_argument('--screen_width_px', type=int, default=3072,
                       help='Screen width in pixels (default: 3072)')
    parser.add_argument('--screen_height_px', type=int, default=1920,
                       help='Screen height in pixels (default: 1920)')
    parser.add_argument('--screen_width_mm', type=float, default=344.6,
                       help='Screen width in millimeters (default: 344.6)')
    parser.add_argument('--screen_height_mm', type=float, default=215.4,
                       help='Screen height in millimeters (default: 215.4)')
    parser.add_argument('--include_kalman_analysis', action='store_true',
                       help='Include Kalman filtering analysis (default: False)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ProgressiveTrainingEvaluator(
        screen_width_px=args.screen_width_px,
        screen_height_px=args.screen_height_px,
        screen_width_mm=args.screen_width_mm,
        screen_height_mm=args.screen_height_mm,
        regression_model_type=args.regression_model,
        feature_type=args.feature_type
    )
    
    # Run progressive training evaluation
    evaluator.run_progressive_training(
        train_folders=args.train_folders,
        test_folder=args.test_folder,
        output_dir=args.output_dir,
        include_kalman_analysis=args.include_kalman_analysis
    )


if __name__ == "__main__":
    main() 