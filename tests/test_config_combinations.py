#!/usr/bin/env python3
"""
Test different configuration combinations to identify the problematic settings
that cause evaluation freeze.
"""

import os
import sys
import yaml
import time
import logging
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_config(base_config_path: str, modifications: dict) -> str:
    """Create a test configuration with modifications."""
    
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply modifications
    for key_path, value in modifications.items():
        keys = key_path.split('.')
        current = config
        
        # Navigate to the nested key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    # Write to temporary file
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_config, default_flow_style=False)
    temp_config.close()
    
    return temp_config.name


def test_configuration(config_path: str, test_name: str, timeout_seconds: int = 180) -> dict:
    """Test a specific configuration."""
    logger.info(f"Testing configuration: {test_name}")
    
    import subprocess
    import signal
    
    # Create command to run training
    cmd = [
        "uv", "run", "train_grpo.py",
        "--config", config_path,
        "--profile", "test",
        "--training.epochs", "0.001",  # Very small training
        "--training.scheduling.eval_steps", "5",
        "--training.scheduling.save_steps", "5",
        "--training.scheduling.logging_steps", "1",
        "--monitoring.wandb.enabled", "false",
        "--callbacks.checkpoint_preservation.enabled", "false",
    ]
    
    start_time = time.time()
    
    try:
        # Run with timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            elapsed = time.time() - start_time
            
            if process.returncode == 0:
                return {
                    "success": True,
                    "elapsed": elapsed,
                    "returncode": process.returncode,
                    "stdout": stdout[-1000:],  # Last 1000 chars
                    "stderr": stderr[-1000:] if stderr else "",
                }
            else:
                return {
                    "success": False,
                    "elapsed": elapsed,
                    "returncode": process.returncode,
                    "stdout": stdout[-1000:],
                    "stderr": stderr[-1000:] if stderr else "",
                    "error": "Non-zero exit code"
                }
        except subprocess.TimeoutExpired:
            # Kill the process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait()
            elapsed = time.time() - start_time
            
            return {
                "success": False,
                "elapsed": elapsed,
                "error": "Timeout",
                "timeout": True
            }
            
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "elapsed": elapsed,
            "error": str(e),
            "exception": True
        }


def main():
    """Main test runner for different configurations."""
    logger.info("🧪 Testing different configuration combinations")
    
    base_config = "config.yaml"
    if not os.path.exists(base_config):
        logger.error(f"Base configuration file not found: {base_config}")
        return
    
    # Define test configurations
    test_configs = [
        {
            "name": "baseline_minimal",
            "description": "Minimal baseline configuration",
            "modifications": {
                "model.name": "Qwen/Qwen2.5-0.5B-Instruct",
                "training.batch_size.per_device_train": 2,
                "training.batch_size.per_device_eval": 2,
                "training.dataloader.num_workers": 0,
                "training.dataloader.pin_memory": False,
                "training.dataloader.persistent_workers": False,
                "grpo.vllm.enabled": False,
                "monitoring.wandb.enabled": False,
                "rewards.functions": ["format"],
                "dataset.split.test_size": 0.05,
            }
        },
        {
            "name": "vllm_enabled",
            "description": "Test with vLLM enabled",
            "modifications": {
                "model.name": "Qwen/Qwen2.5-0.5B-Instruct",
                "training.batch_size.per_device_train": 2,
                "training.batch_size.per_device_eval": 2,
                "training.dataloader.num_workers": 0,
                "grpo.vllm.enabled": True,
                "monitoring.wandb.enabled": False,
                "rewards.functions": ["format"],
                "dataset.split.test_size": 0.05,
            }
        },
        {
            "name": "high_dataloader_workers",
            "description": "Test with high dataloader workers",
            "modifications": {
                "model.name": "Qwen/Qwen2.5-0.5B-Instruct",
                "training.batch_size.per_device_train": 2,
                "training.batch_size.per_device_eval": 2,
                "training.dataloader.num_workers": 8,
                "training.dataloader.pin_memory": True,
                "training.dataloader.persistent_workers": True,
                "grpo.vllm.enabled": False,
                "monitoring.wandb.enabled": False,
                "rewards.functions": ["format"],
                "dataset.split.test_size": 0.05,
            }
        },
        {
            "name": "multiple_rewards",
            "description": "Test with multiple reward functions",
            "modifications": {
                "model.name": "Qwen/Qwen2.5-0.5B-Instruct",
                "training.batch_size.per_device_train": 2,
                "training.batch_size.per_device_eval": 2,
                "training.dataloader.num_workers": 0,
                "grpo.vllm.enabled": False,
                "monitoring.wandb.enabled": False,
                "rewards.functions": ["format", "equation", "accuracy"],
                "dataset.split.test_size": 0.05,
            }
        },
        {
            "name": "wandb_enabled",
            "description": "Test with WandB enabled",
            "modifications": {
                "model.name": "Qwen/Qwen2.5-0.5B-Instruct",
                "training.batch_size.per_device_train": 2,
                "training.batch_size.per_device_eval": 2,
                "training.dataloader.num_workers": 0,
                "grpo.vllm.enabled": False,
                "monitoring.wandb.enabled": True,
                "rewards.functions": ["format"],
                "dataset.split.test_size": 0.05,
            }
        },
        {
            "name": "large_batch_size",
            "description": "Test with larger batch size",
            "modifications": {
                "model.name": "Qwen/Qwen2.5-0.5B-Instruct",
                "training.batch_size.per_device_train": 8,
                "training.batch_size.per_device_eval": 8,
                "training.dataloader.num_workers": 0,
                "grpo.vllm.enabled": False,
                "monitoring.wandb.enabled": False,
                "rewards.functions": ["format"],
                "dataset.split.test_size": 0.05,
            }
        },
        {
            "name": "frequent_eval",
            "description": "Test with frequent evaluation",
            "modifications": {
                "model.name": "Qwen/Qwen2.5-0.5B-Instruct",
                "training.batch_size.per_device_train": 2,
                "training.batch_size.per_device_eval": 2,
                "training.dataloader.num_workers": 0,
                "grpo.vllm.enabled": False,
                "monitoring.wandb.enabled": False,
                "rewards.functions": ["format"],
                "dataset.split.test_size": 0.05,
                "training.scheduling.eval_steps": 2,  # Very frequent
            }
        },
        {
            "name": "current_config",
            "description": "Test with current problematic config",
            "modifications": {
                "model.name": "Qwen/Qwen2.5-3B-Instruct",
                "training.batch_size.per_device_train": 8,
                "training.batch_size.per_device_eval": 8,
                "training.dataloader.num_workers": 8,
                "training.dataloader.pin_memory": True,
                "training.dataloader.persistent_workers": True,
                "grpo.vllm.enabled": True,
                "monitoring.wandb.enabled": True,
                "rewards.functions": ["format", "equation"],
                "dataset.split.test_size": 0.1,
                "training.scheduling.eval_steps": 100,
            }
        }
    ]
    
    results = {}
    
    for config in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config['name']}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"{'='*60}")
        
        # Create test configuration
        try:
            test_config_path = create_test_config(base_config, config['modifications'])
            
            # Run test
            result = test_configuration(test_config_path, config['name'], timeout_seconds=300)
            results[config['name']] = result
            
            # Log result
            if result['success']:
                logger.info(f"✅ {config['name']}: SUCCESS ({result['elapsed']:.1f}s)")
            else:
                if result.get('timeout'):
                    logger.error(f"❌ {config['name']}: TIMEOUT ({result['elapsed']:.1f}s)")
                else:
                    logger.error(f"❌ {config['name']}: FAILED ({result['elapsed']:.1f}s) - {result.get('error', 'Unknown error')}")
                
                # Show stderr if available
                if result.get('stderr'):
                    logger.error(f"   Error output: {result['stderr'][:500]}...")
            
            # Clean up
            os.unlink(test_config_path)
            
        except Exception as e:
            logger.error(f"❌ {config['name']}: Setup failed: {e}")
            results[config['name']] = {"success": False, "error": f"Setup failed: {e}"}
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 SUMMARY")
    logger.info("="*60)
    
    successful_configs = []
    failed_configs = []
    timeout_configs = []
    
    for name, result in results.items():
        if result['success']:
            successful_configs.append(name)
        elif result.get('timeout'):
            timeout_configs.append(name)
        else:
            failed_configs.append(name)
    
    logger.info(f"✅ Successful configurations: {len(successful_configs)}")
    for name in successful_configs:
        logger.info(f"   - {name}")
    
    logger.info(f"❌ Failed configurations: {len(failed_configs)}")
    for name in failed_configs:
        logger.info(f"   - {name}")
    
    logger.info(f"⏱️ Timeout configurations: {len(timeout_configs)}")
    for name in timeout_configs:
        logger.info(f"   - {name}")
    
    # Analysis
    logger.info("\n🔍 ANALYSIS:")
    
    if timeout_configs:
        logger.warning(f"⚠️ The following configurations caused timeouts (likely freeze):")
        for name in timeout_configs:
            logger.warning(f"   - {name}")
    
    if successful_configs and (failed_configs or timeout_configs):
        logger.info("💡 Recommendations:")
        logger.info("   - Compare successful vs failed configurations")
        logger.info("   - Look for common patterns in timeout configurations")
        
        # Identify common issues
        vllm_timeouts = [name for name in timeout_configs if 'vllm' in name or name == 'current_config']
        if vllm_timeouts:
            logger.warning("   - vLLM might be causing freezes")
        
        dataloader_timeouts = [name for name in timeout_configs if 'dataloader' in name or name == 'current_config']
        if dataloader_timeouts:
            logger.warning("   - High dataloader workers might be causing issues")
        
        reward_timeouts = [name for name in timeout_configs if 'reward' in name or name == 'current_config']
        if reward_timeouts:
            logger.warning("   - Multiple reward functions might be causing slowdowns")
    
    # Save results
    import json
    with open("config_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\n📄 Detailed results saved to: config_test_results.json")


if __name__ == "__main__":
    main()