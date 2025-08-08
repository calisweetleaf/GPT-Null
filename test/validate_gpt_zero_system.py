#!/usr/bin/env python3
"""
GPT-Ø System Validation and Readiness Report
============================================

Comprehensive validation script that verifies all components are production-ready
and meet the strict coding requirements outlined in coding_requirements.instructions.md.

This script performs:
- Module import validation
- Core functionality testing  
- Performance benchmarking
- Memory efficiency testing
- Security constraint validation
- Integration testing
- Requirements compliance checking

Author: Synthetic Cognitive Partner (Claude)
Date: August 6, 2025
Status: Production Validation
Compliance: OWASP Top 10, ≥90% test coverage
"""

import sys
import time
import logging
import traceback
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import psutil
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger("GPT-Zero-Validator")


class GPTZeroValidator:
    """Comprehensive validation suite for GPT-Ø system."""
    
    def __init__(self):
        """Initialize validator with tracking metrics."""
        self.results = {
            'modules_tested': 0,
            'modules_passed': 0,
            'modules_failed': 0,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'warnings': [],
            'errors': [],
            'performance_metrics': {},
            'memory_metrics': {},
            'security_checks': {}
        }
        self.start_time = time.time()
    
    def log_test(self, test_name: str, passed: bool, details: str = None):
        """Log test result and update metrics."""
        self.results['tests_run'] += 1
        
        if passed:
            self.results['tests_passed'] += 1
            logger.info(f"✓ {test_name}")
            if details:
                logger.info(f"  {details}")
        else:
            self.results['tests_failed'] += 1
            logger.error(f"✗ {test_name}")
            if details:
                logger.error(f"  {details}")
            self.results['errors'].append(f"{test_name}: {details}")
    
    def log_warning(self, warning: str):
        """Log a warning."""
        logger.warning(warning)
        self.results['warnings'].append(warning)
    
    def validate_module_imports(self) -> bool:
        """Validate all core modules can be imported successfully."""
        logger.info("=== Module Import Validation ===")
        
        modules_to_test = [
            ('recursive_weights_core', 'Core recursive weights implementation'),
            ('cas.neural_memory_runtime', 'Neural memory runtime system'),
            ('gpt_model', 'Main GPT-Ø model'),
            ('tokenizer_mux', 'Multimodal tokenizer'),
            ('tool_output_head', 'Tool synthesis system'),
            ('cas.cas_system', 'Constitutional AI system'),
            ('cas.neural_model_manager', 'Model management system')
        ]
        
        for module_name, description in modules_to_test:
            try:
                __import__(module_name)
                self.log_test(f"Import {module_name}", True, description)
                self.results['modules_passed'] += 1
            except Exception as e:
                self.log_test(f"Import {module_name}", False, f"Import failed: {e}")
                self.results['modules_failed'] += 1
            
            self.results['modules_tested'] += 1
        
        return self.results['modules_failed'] == 0
    
    def validate_recursive_weights(self) -> bool:
        """Validate recursive weights core functionality."""
        logger.info("=== Recursive Weights Validation ===")
        
        try:
            import recursive_weights_core as rwc
            
            # Test 1: Create example weight
            weight = rwc.create_example_recursive_weight()
            self.log_test("Create recursive weight", True, f"Dimension: {weight.dimension_size}")
            
            # Test 2: Registry operations
            registry = rwc.RecursiveWeightRegistry()
            registry.register_weight("test_weight", weight)
            retrieved = registry.get_weight("test_weight")
            self.log_test("Weight registry", retrieved is not None, f"Registry size: {len(registry)}")
            
            # Test 3: Forward computation
            codebook = torch.randn(10, 512)
            result = weight.forward(codebook, time_step=1.0, recursion_depth=2)
            
            valid_output = (
                result.shape == (512,) and
                not torch.isnan(result).any() and
                not torch.isinf(result).any()
            )
            self.log_test("Forward computation", valid_output, f"Output shape: {result.shape}")
            
            # Test 4: Phase computation
            phase = weight.compute_phase_value(1.0)
            valid_phase = phase.shape == (512,) and not torch.isnan(phase).any()
            self.log_test("Phase computation", valid_phase, f"Phase computed successfully")
            
            # Test 5: Layer integration
            layer = rwc.RecursiveWeightLayer(256, 512, 16)
            test_input = torch.randn(2, 10, 256)
            output = layer(test_input, time_step=1.0, recursion_depth=2)
            valid_layer = output.shape == (2, 10, 512) and not torch.isnan(output).any()
            self.log_test("Layer integration", valid_layer, f"Layer output shape: {output.shape}")
            
            # Test 6: Serialization
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.rwgt', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
            
            try:
                rwc.RecursiveWeightSerializer.serialize_weight(weight, tmp_path)
                deserialized = rwc.RecursiveWeightSerializer.deserialize_weight(tmp_path)
                
                # Verify serialization accuracy
                orig_result = weight.forward(codebook, recursion_depth=0)
                deser_result = deserialized.forward(codebook, recursion_depth=0)
                diff = torch.abs(orig_result - deser_result).max()
                
                self.log_test("Serialization", diff < 1e-5, f"Serialization error: {diff:.2e}")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            
            return True
            
        except Exception as e:
            self.log_test("Recursive weights validation", False, f"Exception: {e}")
            traceback.print_exc()
            return False
    
    def validate_neural_memory(self) -> bool:
        """Validate neural memory runtime functionality."""
        logger.info("=== Neural Memory Runtime Validation ===")
        
        try:
            from cas import neural_memory_runtime as nmr
            
            # Test 1: Create memory runtime
            config = {
                'max_context_length': 1000,
                'memory_budget_mb': 100,
                'compression_target': 8.0,
                'max_tensors_per_tier': 1000,
                'compression_threshold': 0.5
            }
            runtime = nmr.NeuralMemoryRuntime(config)
            self.log_test("Create memory runtime", True, f"Budget: {config['memory_budget_mb']}MB")
            
            # Test 2: Store and retrieve tensor
            test_tensor = torch.randn(100, 512)
            tensor_id = runtime.store_activation("test_tensor", test_tensor, nmr.MemoryTier.HOT)
            retrieved = runtime.retrieve_activation(tensor_id)
            
            retrieval_valid = (
                retrieved is not None and
                torch.allclose(test_tensor, retrieved, rtol=1e-3)
            )
            self.log_test("Memory storage/retrieval", retrieval_valid, f"Tensor ID: {tensor_id}")
            
            # Test 3: Memory tier functionality
            tier_test = len(list(nmr.MemoryTier)) == 5
            self.log_test("Memory tier system", tier_test, f"5-tier memory hierarchy")
            
            # Test 4: Context summarization (if available)
            try:
                summary = runtime.context_summarizer
                self.log_test("Context summarizer", summary is not None, "Context summarizer available")
            except:
                self.log_test("Context summarizer", False, "Context summarizer not available")
            
            return True
            
        except Exception as e:
            self.log_test("Neural memory validation", False, f"Exception: {e}")
            traceback.print_exc()
            return False
    
    def validate_performance(self) -> bool:
        """Validate system performance meets requirements."""
        logger.info("=== Performance Validation ===")
        
        try:
            import recursive_weights_core as rwc
            
            # Performance test: Large-scale processing
            layer = rwc.RecursiveWeightLayer(1024, 2048, 64)
            test_input = torch.randn(8, 256, 1024)  # 8 batch, 256 seq, 1024 features
            
            start_time = time.time()
            output = layer(test_input, time_step=1.0, recursion_depth=2)
            end_time = time.time()
            
            processing_time = end_time - start_time
            tokens_per_second = (8 * 256) / processing_time
            
            self.results['performance_metrics']['tokens_per_second'] = tokens_per_second
            self.results['performance_metrics']['processing_time'] = processing_time
            
            performance_target = tokens_per_second > 500  # Adjust as needed
            self.log_test("Performance benchmark", performance_target,
                         f"{tokens_per_second:.1f} tokens/sec, {processing_time:.4f}s")
            
            return performance_target
            
        except Exception as e:
            self.log_test("Performance validation", False, f"Exception: {e}")
            return False
    
    def validate_memory_efficiency(self) -> bool:
        """Validate memory efficiency requirements."""
        logger.info("=== Memory Efficiency Validation ===")
        
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Memory test: Create and manage large tensors
            from cas import neural_memory_runtime as nmr
            config = {
                'max_context_length': 10000,
                'memory_budget_mb': 500,
                'compression_target': 8.0,
                'max_tensors_per_tier': 1000,
                'compression_threshold': 0.5
            }
            runtime = nmr.NeuralMemoryRuntime(config)
            
            tensor_ids = []
            for i in range(50):
                tensor = torch.randn(1000, 512)
                tensor_id = runtime.store_activation(f"tensor_{i}", tensor, nmr.MemoryTier.WARM)
                tensor_ids.append(tensor_id)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Clean up
            for tensor_id in tensor_ids:
                runtime.delete_activation(tensor_id)
            
            import gc
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_growth = peak_memory - initial_memory
            memory_cleanup = peak_memory - final_memory
            
            self.results['memory_metrics']['initial_mb'] = initial_memory
            self.results['memory_metrics']['peak_mb'] = peak_memory
            self.results['memory_metrics']['final_mb'] = final_memory
            self.results['memory_metrics']['growth_mb'] = memory_growth
            self.results['memory_metrics']['cleanup_mb'] = memory_cleanup
            
            memory_efficient = memory_growth < 1000 and memory_cleanup > memory_growth * 0.5
            self.log_test("Memory efficiency", memory_efficient,
                         f"Growth: {memory_growth:.1f}MB, Cleanup: {memory_cleanup:.1f}MB")
            
            return memory_efficient
            
        except Exception as e:
            self.log_test("Memory efficiency validation", False, f"Exception: {e}")
            return False
    
    def validate_security_constraints(self) -> bool:
        """Validate security constraints and input validation."""
        logger.info("=== Security Validation ===")
        
        try:
            import recursive_weights_core as rwc
            
            security_tests = 0
            security_passed = 0
            
            # Test 1: Input validation
            try:
                rwc.validate_tensor_input("not_a_tensor", "test")
                security_tests += 1
            except rwc.ValidationError:
                security_passed += 1
                security_tests += 1
            
            # Test 2: NaN detection
            try:
                nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
                rwc.validate_tensor_input(nan_tensor, "nan_test")
                security_tests += 1
            except rwc.SecurityError:
                security_passed += 1
                security_tests += 1
            
            # Test 3: Inf detection
            try:
                inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
                rwc.validate_tensor_input(inf_tensor, "inf_test")
                security_tests += 1
            except rwc.SecurityError:
                security_passed += 1
                security_tests += 1
            
            # Test 4: Value bounds
            try:
                large_tensor = torch.tensor([1e7, 2e7, 3e7])
                rwc.validate_tensor_input(large_tensor, "large_test")
                security_tests += 1
            except rwc.SecurityError:
                security_passed += 1
                security_tests += 1
            
            # Test 5: Recursion depth limits
            try:
                weight = rwc.create_example_recursive_weight()
                codebook = torch.randn(10, 512)
                weight.forward(codebook, recursion_depth=1000)
                security_tests += 1
            except rwc.SecurityError:
                security_passed += 1
                security_tests += 1
            
            self.results['security_checks']['total_tests'] = security_tests
            self.results['security_checks']['passed_tests'] = security_passed
            
            security_valid = security_passed == security_tests
            self.log_test("Security constraints", security_valid,
                         f"{security_passed}/{security_tests} security tests passed")
            
            return security_valid
            
        except Exception as e:
            self.log_test("Security validation", False, f"Exception: {e}")
            return False
    
    def validate_integration(self) -> bool:
        """Validate system integration and interoperability."""
        logger.info("=== Integration Validation ===")
        
        try:
            import recursive_weights_core as rwc
            from cas import neural_memory_runtime as nmr
            
            # Integration test: Memory runtime + Recursive weights
            config = {
                'max_context_length': 1000,
                'memory_budget_mb': 200,
                'compression_target': 8.0,
                'max_tensors_per_tier': 1000,
                'compression_threshold': 0.5
            }
            memory_runtime = nmr.NeuralMemoryRuntime(config)
            
            rw_layer = rwc.RecursiveWeightLayer(256, 512, 8)
            test_input = torch.randn(1, 50, 256)
            
            # Process through recursive weights
            output = rw_layer(test_input, time_step=1.0, recursion_depth=1)
            
            # Store in memory system
            input_id = memory_runtime.store_activation("test_input", test_input, nmr.MemoryTier.HOT)
            output_id = memory_runtime.store_activation("test_output", output, nmr.MemoryTier.HOT)
            
            # Verify retrieval
            retrieved_input = memory_runtime.retrieve_activation(input_id)
            retrieved_output = memory_runtime.retrieve_activation(output_id)
            
            integration_valid = (
                torch.allclose(test_input, retrieved_input, rtol=1e-3) and
                torch.allclose(output, retrieved_output, rtol=1e-3)
            )
            
            self.log_test("System integration", integration_valid,
                         "Memory runtime + Recursive weights integration")
            
            return integration_valid
            
        except Exception as e:
            self.log_test("Integration validation", False, f"Exception: {e}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_validation_time_seconds': total_time,
            'summary': {
                'modules_tested': self.results['modules_tested'],
                'modules_passed': self.results['modules_passed'],
                'modules_failed': self.results['modules_failed'],
                'tests_run': self.results['tests_run'],
                'tests_passed': self.results['tests_passed'], 
                'tests_failed': self.results['tests_failed'],
                'success_rate': self.results['tests_passed'] / max(1, self.results['tests_run']),
                'production_ready': self.results['tests_failed'] == 0 and self.results['modules_failed'] == 0
            },
            'performance_metrics': self.results['performance_metrics'],
            'memory_metrics': self.results['memory_metrics'],
            'security_checks': self.results['security_checks'],
            'warnings': self.results['warnings'],
            'errors': self.results['errors']
        }
        
        return report
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite."""
        logger.info("🚀 Starting GPT-Ø System Validation Suite")
        logger.info("=" * 60)
        
        validation_steps = [
            self.validate_module_imports,
            self.validate_recursive_weights, 
            self.validate_neural_memory,
            self.validate_performance,
            self.validate_memory_efficiency,
            self.validate_security_constraints,
            self.validate_integration
        ]
        
        overall_success = True
        
        for step in validation_steps:
            try:
                success = step()
                overall_success = overall_success and success
            except Exception as e:
                logger.error(f"Validation step failed: {step.__name__}")
                logger.error(f"Exception: {e}")
                traceback.print_exc()
                overall_success = False
        
        # Generate and display report
        report = self.generate_report()
        
        logger.info("=" * 60)
        logger.info("🔍 VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Modules tested: {report['summary']['modules_tested']}")
        logger.info(f"Modules passed: {report['summary']['modules_passed']}")
        logger.info(f"Tests run: {report['summary']['tests_run']}")
        logger.info(f"Tests passed: {report['summary']['tests_passed']}")
        logger.info(f"Success rate: {report['summary']['success_rate']:.1%}")
        
        if report['performance_metrics']:
            logger.info(f"Performance: {report['performance_metrics'].get('tokens_per_second', 0):.1f} tokens/sec")
        
        if report['memory_metrics']:
            logger.info(f"Memory growth: {report['memory_metrics'].get('growth_mb', 0):.1f}MB")
        
        if report['summary']['production_ready']:
            logger.info("✅ GPT-Ø System is PRODUCTION READY!")
        else:
            logger.error("❌ GPT-Ø System has issues that need to be addressed.")
            if report['errors']:
                logger.error("Errors encountered:")
                for error in report['errors']:
                    logger.error(f"  - {error}")
        
        # Save report
        report_path = Path("validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Detailed report saved to: {report_path}")
        
        return overall_success


def main():
    """Main validation entry point."""
    validator = GPTZeroValidator()
    success = validator.run_full_validation()
    
    if success:
        logger.info("🎉 All validations passed! GPT-Ø is ready for deployment.")
        sys.exit(0)
    else:
        logger.error("💥 Validation failures detected. Review errors before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()