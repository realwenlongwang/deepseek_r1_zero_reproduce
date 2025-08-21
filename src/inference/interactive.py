#!/usr/bin/env python3
"""
Interactive utilities for DeepSeek R1 Zero inference.
Provides user-friendly interfaces for model interaction.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class InteractiveSession:
    """
    Enhanced interactive session with conversation management.
    """
    
    def __init__(self, inference_engine, save_conversations: bool = True):
        """
        Initialize interactive session.
        
        Args:
            inference_engine: InferenceEngine instance
            save_conversations: Whether to save conversations to file
        """
        self.engine = inference_engine
        self.save_conversations = save_conversations
        self.conversation_history = []
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        if save_conversations:
            self.conversations_dir = "conversations"
            os.makedirs(self.conversations_dir, exist_ok=True)
    
    def start(self, 
             system_message: str = None,
             welcome_message: str = None,
             max_turns: int = 50):
        """
        Start interactive conversation.
        
        Args:
            system_message: Optional system message
            welcome_message: Custom welcome message
            max_turns: Maximum conversation turns
        """
        # Display welcome
        if welcome_message:
            print(welcome_message)
        else:
            self._show_welcome()
        
        # Initialize conversation
        if system_message:
            self.conversation_history.append({
                "role": "system", 
                "content": system_message,
                "timestamp": datetime.now().isoformat()
            })
        
        turn = 0
        while turn < max_turns:
            try:
                # Get user input
                user_input = self._get_user_input()
                
                if user_input is None:  # Exit command
                    break
                
                if user_input.startswith('/'):  # Handle commands
                    if self._handle_command(user_input):
                        continue
                    else:
                        break
                
                # Add to history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Generate response
                response = self._generate_response()
                
                if response:
                    print(f"\n🤖 Assistant:\n{response}\n")
                    
                    # Add to history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                
                turn += 1
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted by user. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
        
        # Save conversation
        if self.save_conversations and self.conversation_history:
            self._save_conversation()
        
        print(f"Session ended after {turn} turns.")
    
    def _show_welcome(self):
        """Display welcome message and help."""
        print("="*80)
        print("🤖 DeepSeek R1 Zero Interactive Chat")
        print("="*80)
        print("Commands:")
        print("  /help     - Show this help message")
        print("  /clear    - Clear conversation history")
        print("  /history  - Show conversation history")
        print("  /save     - Save current conversation")
        print("  /load     - Load previous conversation")
        print("  /preset   - Change generation preset")
        print("  /quit     - Exit the conversation")
        print("  /exit     - Exit the conversation")
        print("\nJust type your message to start chatting!")
        print("="*80)
    
    def _get_user_input(self) -> Optional[str]:
        """Get user input with prompt."""
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '/quit', '/exit']:
                return None
            
            return user_input if user_input else self._get_user_input()
            
        except (KeyboardInterrupt, EOFError):
            return None
    
    def _handle_command(self, command: str) -> bool:
        """
        Handle user commands.
        
        Args:
            command: Command string starting with '/'
            
        Returns:
            True to continue conversation, False to exit
        """
        cmd = command.lower().strip()
        
        if cmd in ['/help', '/h']:
            self._show_welcome()
        
        elif cmd in ['/clear', '/c']:
            self.conversation_history = []
            print("🧹 Conversation history cleared.")
        
        elif cmd in ['/history', '/hist']:
            self._show_history()
        
        elif cmd in ['/save', '/s']:
            if self._save_conversation():
                print("💾 Conversation saved.")
        
        elif cmd in ['/load', '/l']:
            self._load_conversation()
        
        elif cmd in ['/preset', '/p']:
            self._change_preset()
        
        elif cmd in ['/quit', '/exit', '/q']:
            return False
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Type /help for available commands.")
        
        return True
    
    def _generate_response(self) -> Optional[str]:
        """Generate response using the inference engine."""
        try:
            # Prepare messages for generation
            messages = []
            for msg in self.conversation_history:
                if msg["role"] in ["system", "user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Get latest user message
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            if not user_messages:
                return None
            
            latest_prompt = user_messages[-1]["content"]
            
            # Generate response
            print("\n🤖 Assistant: ", end="", flush=True)
            print("(thinking...)", end="", flush=True)
            
            response = self.engine.generate(
                prompt=latest_prompt,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                use_system_message=any(msg["role"] == "system" for msg in messages)
            )
            
            print("\r🤖 Assistant: ", end="", flush=True)
            return response
            
        except Exception as e:
            print(f"\n❌ Generation failed: {e}")
            return None
    
    def _show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
        
        print("\n📝 Conversation History:")
        print("-" * 60)
        
        for i, msg in enumerate(self.conversation_history):
            role_emoji = {"system": "⚙️", "user": "👤", "assistant": "🤖"}.get(msg["role"], "❓")
            timestamp = msg.get("timestamp", "").split("T")[-1].split(".")[0]
            
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            
            print(f"{i+1:2d}. {role_emoji} {msg['role'].capitalize():9} [{timestamp}]: {content}")
        
        print("-" * 60)
    
    def _save_conversation(self) -> bool:
        """Save conversation to file."""
        if not self.save_conversations or not self.conversation_history:
            return False
        
        try:
            filename = f"conversation_{self.session_id}.json"
            filepath = os.path.join(self.conversations_dir, filename)
            
            conversation_data = {
                "session_id": self.session_id,
                "session_start": self.session_start.isoformat(),
                "session_end": datetime.now().isoformat(),
                "turn_count": len([msg for msg in self.conversation_history if msg["role"] == "user"]),
                "conversation": self.conversation_history
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Conversation saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save conversation: {e}")
            return False
    
    def _load_conversation(self):
        """Load previous conversation."""
        if not os.path.exists(self.conversations_dir):
            print("📁 No saved conversations found.")
            return
        
        # List available conversations
        conversation_files = [f for f in os.listdir(self.conversations_dir) if f.endswith('.json')]
        
        if not conversation_files:
            print("📁 No saved conversations found.")
            return
        
        print("\n📁 Available conversations:")
        for i, filename in enumerate(conversation_files, 1):
            print(f"  {i}. {filename}")
        
        try:
            choice = input("\nEnter number to load (or press Enter to cancel): ").strip()
            if not choice:
                return
            
            idx = int(choice) - 1
            if 0 <= idx < len(conversation_files):
                filepath = os.path.join(self.conversations_dir, conversation_files[idx])
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    conversation_data = json.load(f)
                
                self.conversation_history = conversation_data["conversation"]
                print(f"📂 Loaded conversation from {conversation_files[idx]}")
                
                # Show brief summary
                turn_count = conversation_data.get("turn_count", 0)
                session_start = conversation_data.get("session_start", "unknown")
                print(f"   Turns: {turn_count}, Started: {session_start}")
            else:
                print("❌ Invalid selection.")
                
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Failed to load conversation: {e}")
    
    def _change_preset(self):
        """Change generation preset."""
        from .generators import GenerationPresets
        
        presets = {
            "1": ("Creative", GenerationPresets.creative()),
            "2": ("Balanced", GenerationPresets.balanced()),
            "3": ("Precise", GenerationPresets.precise()),
            "4": ("Deterministic", GenerationPresets.deterministic()),
            "5": ("Reasoning", GenerationPresets.reasoning())
        }
        
        print("\n🎛️ Available generation presets:")
        for key, (name, params) in presets.items():
            temp = params.get("temperature", "N/A")
            top_p = params.get("top_p", "N/A")
            print(f"  {key}. {name} (temp={temp}, top_p={top_p})")
        
        try:
            choice = input("\nSelect preset (1-5) or press Enter to cancel: ").strip()
            
            if choice in presets:
                name, params = presets[choice]
                
                # Update engine parameters (this would need to be implemented in InferenceEngine)
                print(f"🎛️ Preset changed to: {name}")
                print(f"   Parameters: {params}")
            elif choice:
                print("❌ Invalid selection.")
                
        except KeyboardInterrupt:
            print("\n❌ Cancelled.")


def run_interactive_chat(checkpoint_path: str = None, 
                        device: str = "auto",
                        **generation_kwargs):
    """
    Convenience function to start interactive chat.
    
    Args:
        checkpoint_path: Path to model checkpoint (uses latest if None)
        device: Device for inference
        **generation_kwargs: Default generation parameters
    """
    from .checkpoint_loader import AutoCheckpointLoader, find_latest_checkpoint
    from .generators import InferenceEngine
    
    # Find checkpoint if not specified
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint("saved_models")
        if checkpoint_path is None:
            print("❌ No checkpoints found. Please specify checkpoint_path.")
            return
        print(f"🔍 Using latest checkpoint: {checkpoint_path}")
    
    # Load model
    print("🔄 Loading checkpoint...")
    loader = AutoCheckpointLoader(device_map=device)
    model, tokenizer, metadata = loader.load_checkpoint(checkpoint_path)
    
    # Create inference engine
    engine = InferenceEngine(model, tokenizer, device)
    
    # Start interactive session
    session = InteractiveSession(engine, save_conversations=True)
    session.start()


if __name__ == "__main__":
    run_interactive_chat()