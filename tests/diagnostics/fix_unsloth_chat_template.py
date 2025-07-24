#!/usr/bin/env python3
"""
Example of how to properly fix the chat template in train_grpo.py using Unsloth's methods.
"""

def _configure_tokenizer_unsloth_way(tokenizer, model_name):
    """Configure tokenizer using Unsloth's proper chat template method."""
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template using Unsloth's method
    if tokenizer.chat_template is None:
        try:
            from unsloth.chat_templates import get_chat_template, CHAT_TEMPLATES
            
            # Check available templates
            available_templates = list(CHAT_TEMPLATES.keys())
            print(f"Available Unsloth chat templates: {available_templates}")
            
            # For Qwen models, use appropriate template
            if "qwen" in model_name.lower():
                template_name = "qwen-2.5"  # or "chatml" for general ChatML format
            elif "gemma" in model_name.lower():
                template_name = "gemma-3"
            elif "llama" in model_name.lower():
                template_name = "llama-3.1"
            else:
                template_name = "chatml"  # Default ChatML format
            
            print(f"Using chat template: {template_name}")
            
            # Apply the chat template using Unsloth's method
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=template_name,
                mapping={"role": "role", "content": "content"},  # Standard mapping
                map_eos_token=True
            )
            
            print("✅ Successfully applied Unsloth chat template")
            
        except ImportError:
            print("⚠️ Unsloth chat_templates not available, using ChatML fallback")
            # Fallback to proper ChatML format
            tokenizer.chat_template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
        
        except Exception as e:
            print(f"⚠️ Failed to apply Unsloth chat template: {e}")
            # Fallback to proper ChatML format
            tokenizer.chat_template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
    
    return tokenizer

# Example of how to modify the train_grpo.py file:
def modified_configure_tokenizer_example():
    """
    This is how the _configure_tokenizer function should be modified in train_grpo.py
    """
    return """
def _configure_tokenizer(tokenizer, model_name=None):
    '''Configure tokenizer settings for GRPO training.'''
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template using Unsloth's method if available
    if tokenizer.chat_template is None:
        try:
            from unsloth.chat_templates import get_chat_template
            
            # Determine appropriate template based on model
            if model_name and "qwen" in model_name.lower():
                template_name = "qwen-2.5"
            else:
                template_name = "chatml"  # Default ChatML format
            
            logger.info(f"Applying chat template: {template_name}")
            
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=template_name,
                mapping={"role": "role", "content": "content"},
                map_eos_token=True
            )
            
            logger.info("✅ Successfully applied Unsloth chat template")
            
        except ImportError:
            logger.warning("Unsloth chat_templates not available, using ChatML fallback")
            tokenizer.chat_template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\\n{{ message['content'] }}<|im_end|>\\n{% endfor %}"
        
        except Exception as e:
            logger.warning(f"Failed to apply Unsloth chat template: {e}")  
            tokenizer.chat_template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\\n{{ message['content'] }}<|im_end|>\\n{% endfor %}"
    
    return tokenizer
"""

if __name__ == "__main__":
    print("🔧 Proper Unsloth Chat Template Setup")
    print("=" * 60)
    
    print("1. Use Unsloth's get_chat_template() function")
    print("2. Choose appropriate template based on model type")
    print("3. Apply proper mapping and EOS token handling")
    print("4. Have fallback to ChatML format")
    
    print("\n📝 Available Qwen templates in Unsloth:")
    print("   • qwen-2.5 - For Qwen 2.5 models")
    print("   • chatml - General ChatML format")
    
    print("\n🎯 This approach ensures:")
    print("   • Proper ChatML tokens are preserved")
    print("   • Model-specific optimizations are applied")
    print("   • Consistent behavior between standard and unsloth models")
    
    print(f"\n📄 Modified function:\n{modified_configure_tokenizer_example()}")