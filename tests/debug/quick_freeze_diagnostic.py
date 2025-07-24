#!/usr/bin/env python3
"""
Quick diagnostic script to identify the most likely cause of evaluation freeze.
This script runs the most common issue checks first.
"""

import os
import sys
import logging
import subprocess
import time
import signal
import tempfile
import psutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_with_monitoring(cmd, timeout=120, description=""):
    """Run command with resource monitoring."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )
        
        # Monitor process
        max_memory = 0
        max_cpu = 0
        
        try:
            ps_process = psutil.Process(process.pid)
            
            while process.poll() is None:
                try:
                    # Monitor resource usage
                    memory_info = ps_process.memory_info()
                    cpu_percent = ps_process.cpu_percent()
                    
                    current_memory = memory_info.rss / 1024 / 1024  # MB
                    max_memory = max(max_memory, current_memory)
                    max_cpu = max(max_cpu, cpu_percent)
                    
                    elapsed = time.time() - start_time
                    
                    if elapsed > timeout:
                        logger.warning(f"⏱️ Timeout reached ({timeout}s), terminating...")
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        process.wait(timeout=5)
                        return {
                            "success": False,
                            "timeout": True,
                            "elapsed": elapsed,
                            "max_memory_mb": max_memory,
                            "max_cpu_percent": max_cpu,
                            "error": "Timeout"
                        }
                    
                    time.sleep(0.1)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
            
            # Get final output
            stdout, stderr = process.communicate(timeout=10)
            elapsed = time.time() - start_time
            
            return {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "elapsed": elapsed,
                "max_memory_mb": max_memory,
                "max_cpu_percent": max_cpu,
                "stdout": stdout[-1000:],
                "stderr": stderr[-1000:] if stderr else "",
                "timeout": False
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "elapsed": elapsed,
                "max_memory_mb": max_memory,
                "max_cpu_percent": max_cpu,
                "error": str(e),
                "timeout": False
            }
            
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "elapsed": elapsed,
            "error": str(e),
            "timeout": False
        }


def test_minimal_training():
    """Test minimal training configuration."""
    logger.info("🧪 Testing minimal training configuration...")
    
    cmd = [
        "uv", "run", "train_grpo.py",
        "--profile", "test",
        "--model.name", "Qwen/Qwen2.5-0.5B-Instruct",
        "--training.batch_size.per_device_train", "2",
        "--training.batch_size.per_device_eval", "2",
        "--training.dataloader.num_workers", "0",
        "--training.dataloader.pin_memory", "false",
        "--training.dataloader.persistent_workers", "false",
        "--grpo.vllm.enabled", "false",
        "--monitoring.wandb.enabled", "false",
        "--rewards.functions", "format",
        "--dataset.split.test_size", "0.05",
        "--training.epochs", "0.001",
        "--training.scheduling.eval_steps", "3",
        "--training.scheduling.save_steps", "3",
        "--training.scheduling.logging_steps", "1",
        "--callbacks.checkpoint_preservation.enabled", "false",
    ]
    
    result = run_with_monitoring(cmd, timeout=120, description="Minimal training")
    
    if result["success"]:
        logger.info(f"✅ Minimal training: SUCCESS ({result['elapsed']:.1f}s)")
        logger.info(f"   Memory: {result['max_memory_mb']:.1f}MB, CPU: {result['max_cpu_percent']:.1f}%")
        return True
    else:
        if result.get("timeout"):
            logger.error(f"❌ Minimal training: TIMEOUT after {result['elapsed']:.1f}s")
        else:
            logger.error(f"❌ Minimal training: FAILED after {result['elapsed']:.1f}s")
        logger.error(f"   Memory: {result['max_memory_mb']:.1f}MB, CPU: {result['max_cpu_percent']:.1f}%")
        if result.get("stderr"):
            logger.error(f"   Error: {result['stderr'][:500]}...")
        return False


def test_vllm_issue():
    """Test if vLLM is causing the freeze."""
    logger.info("🧪 Testing vLLM issue...")
    
    cmd = [
        "uv", "run", "train_grpo.py",
        "--profile", "test",
        "--model.name", "Qwen/Qwen2.5-0.5B-Instruct",
        "--training.batch_size.per_device_train", "2",
        "--training.batch_size.per_device_eval", "2",
        "--training.dataloader.num_workers", "0",
        "--grpo.vllm.enabled", "true",  # Enable vLLM
        "--monitoring.wandb.enabled", "false",
        "--rewards.functions", "format",
        "--dataset.split.test_size", "0.05",
        "--training.epochs", "0.001",
        "--training.scheduling.eval_steps", "3",
        "--training.scheduling.save_steps", "3",
        "--callbacks.checkpoint_preservation.enabled", "false",
    ]
    
    result = run_with_monitoring(cmd, timeout=120, description="vLLM enabled")
    
    if result["success"]:
        logger.info(f"✅ vLLM test: SUCCESS ({result['elapsed']:.1f}s)")
        return True
    else:
        if result.get("timeout"):
            logger.error(f"❌ vLLM test: TIMEOUT after {result['elapsed']:.1f}s - vLLM likely causing freeze")
        else:
            logger.error(f"❌ vLLM test: FAILED after {result['elapsed']:.1f}s")
        return False


def test_dataloader_issue():
    """Test if high dataloader workers are causing the freeze."""
    logger.info("🧪 Testing dataloader workers issue...")
    
    cmd = [
        "uv", "run", "train_grpo.py",
        "--profile", "test",
        "--model.name", "Qwen/Qwen2.5-0.5B-Instruct",
        "--training.batch_size.per_device_train", "2",
        "--training.batch_size.per_device_eval", "2",
        "--training.dataloader.num_workers", "8",  # High workers
        "--training.dataloader.pin_memory", "true",
        "--training.dataloader.persistent_workers", "true",
        "--grpo.vllm.enabled", "false",
        "--monitoring.wandb.enabled", "false",
        "--rewards.functions", "format",
        "--dataset.split.test_size", "0.05",
        "--training.epochs", "0.001",
        "--training.scheduling.eval_steps", "3",
        "--training.scheduling.save_steps", "3",
        "--callbacks.checkpoint_preservation.enabled", "false",
    ]
    
    result = run_with_monitoring(cmd, timeout=120, description="High dataloader workers")
    
    if result["success"]:
        logger.info(f"✅ Dataloader test: SUCCESS ({result['elapsed']:.1f}s)")
        return True
    else:
        if result.get("timeout"):
            logger.error(f"❌ Dataloader test: TIMEOUT after {result['elapsed']:.1f}s - High workers likely causing freeze")
        else:
            logger.error(f"❌ Dataloader test: FAILED after {result['elapsed']:.1f}s")
        return False


def test_reward_functions_issue():
    """Test if multiple reward functions are causing the freeze."""
    logger.info("🧪 Testing multiple reward functions issue...")
    
    cmd = [
        "uv", "run", "train_grpo.py",
        "--profile", "test",
        "--model.name", "Qwen/Qwen2.5-0.5B-Instruct",
        "--training.batch_size.per_device_train", "2",
        "--training.batch_size.per_device_eval", "2",
        "--training.dataloader.num_workers", "0",
        "--grpo.vllm.enabled", "false",
        "--monitoring.wandb.enabled", "false",
        "--rewards.functions", "format,equation,accuracy",  # Multiple rewards
        "--dataset.split.test_size", "0.05",
        "--training.epochs", "0.001",
        "--training.scheduling.eval_steps", "3",
        "--training.scheduling.save_steps", "3",
        "--callbacks.checkpoint_preservation.enabled", "false",
    ]
    
    result = run_with_monitoring(cmd, timeout=120, description="Multiple reward functions")
    
    if result["success"]:
        logger.info(f"✅ Multiple rewards test: SUCCESS ({result['elapsed']:.1f}s)")
        return True
    else:
        if result.get("timeout"):
            logger.error(f"❌ Multiple rewards test: TIMEOUT after {result['elapsed']:.1f}s - Reward functions likely causing freeze")
        else:
            logger.error(f"❌ Multiple rewards test: FAILED after {result['elapsed']:.1f}s")
        return False


def test_current_config():
    """Test the current problematic configuration."""
    logger.info("🧪 Testing current problematic configuration...")
    
    cmd = [
        "uv", "run", "train_grpo.py",
        "--training.epochs", "0.001",
        "--training.scheduling.eval_steps", "5",
        "--training.scheduling.save_steps", "5",
        "--callbacks.checkpoint_preservation.enabled", "false",
        "--dataset.split.test_size", "0.05",
    ]
    
    result = run_with_monitoring(cmd, timeout=180, description="Current config")
    
    if result["success"]:
        logger.info(f"✅ Current config test: SUCCESS ({result['elapsed']:.1f}s)")
        return True
    else:
        if result.get("timeout"):
            logger.error(f"❌ Current config test: TIMEOUT after {result['elapsed']:.1f}s - Current config causes freeze")
        else:
            logger.error(f"❌ Current config test: FAILED after {result['elapsed']:.1f}s")
        return False


def main():
    """Run quick diagnostic tests."""
    logger.info("🚀 Running Quick Freeze Diagnostic")
    logger.info("="*60)
    
    # Check system resources
    logger.info("💻 System Resources:")
    logger.info(f"   CPU cores: {psutil.cpu_count()}")
    logger.info(f"   Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    
    if hasattr(psutil, 'nvidia_smi_info'):
        logger.info("   GPU: Available")
    else:
        logger.info("   GPU: Not detected by psutil")
    
    tests = [
        ("Minimal Training", test_minimal_training),
        ("vLLM Issue", test_vllm_issue),
        ("Dataloader Issue", test_dataloader_issue),
        ("Reward Functions Issue", test_reward_functions_issue),
        ("Current Config", test_current_config),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"❌ {test_name}: Exception: {e}")
            results[test_name] = False
    
    # Summary and recommendations
    logger.info("\n" + "="*60)
    logger.info("🔍 DIAGNOSTIC SUMMARY")
    logger.info("="*60)
    
    successful_tests = [name for name, success in results.items() if success]
    failed_tests = [name for name, success in results.items() if not success]
    
    logger.info(f"✅ Successful tests: {len(successful_tests)}")
    for name in successful_tests:
        logger.info(f"   - {name}")
    
    logger.info(f"❌ Failed tests: {len(failed_tests)}")
    for name in failed_tests:
        logger.info(f"   - {name}")
    
    # Generate recommendations
    logger.info("\n💡 RECOMMENDATIONS:")
    
    if "Minimal Training" in successful_tests:
        logger.info("✅ Basic training works - issue is in specific configuration")
        
        if "vLLM Issue" in failed_tests:
            logger.warning("⚠️ vLLM is likely causing the freeze")
            logger.info("   → Set grpo.vllm.enabled: false in config.yaml")
        
        if "Dataloader Issue" in failed_tests:
            logger.warning("⚠️ High dataloader workers are likely causing the freeze")
            logger.info("   → Set training.dataloader.num_workers: 0 in config.yaml")
        
        if "Reward Functions Issue" in failed_tests:
            logger.warning("⚠️ Multiple reward functions are likely causing slow evaluation")
            logger.info("   → Reduce rewards.functions to just ['format'] for testing")
        
        if "Current Config" in failed_tests:
            logger.warning("⚠️ Current configuration has issues")
            logger.info("   → Apply the fixes identified above")
    
    else:
        logger.error("❌ Even minimal training failed - check:")
        logger.error("   - GPU memory availability")
        logger.error("   - Model download status")
        logger.error("   - Dataset accessibility")
        logger.error("   - Dependencies installation")
    
    logger.info("\n📋 NEXT STEPS:")
    logger.info("1. Apply the recommended configuration changes")
    logger.info("2. Run training with the fixed configuration")
    logger.info("3. If still having issues, run the comprehensive test suite:")
    logger.info("   uv run python test_evaluation_freeze.py")


if __name__ == "__main__":
    main()