import json
import re
from typing import Dict, List, Optional, Tuple
import datasets
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

# DeepSeek system prompt for GRPO based training
SYSTEM_PROMPT = (
  f"""A conversation between User and Assistant. The user asks a question, 
      and the Assistant solves it. The assistant
      first thinks about the reasoning process in the mind and 
      then provides the user with the answer. The reasoning
      process and answer are enclosed within <think> </think> 
      and <answer> </answer> tags, respectively, i.e., 
      <think> reasoning process here </think><answer> answer here </answer>
   """
)

def make_conversation(example):
    """Convert dataset examples into conversation format for training."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

def make_conversation_bespoke(example):
    """Convert Bespoke-Stratos dataset into conversation format."""
    # Extract user message from conversations
    user_message = ""
    for conv in example["conversations"]:
        if conv["from"] == "user":
            user_message = conv["value"]
            break
    
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }

def convert_bespoke_tags(text: str) -> str:
    """Convert Bespoke-Stratos tags to our format."""
    # Convert <|begin_of_thought|> to <think>
    text = re.sub(r'<\|begin_of_thought\|>', '<think>', text)
    # Convert <|end_of_solution|> to </think>\n\n<answer> and add closing </answer>
    text = re.sub(r'<\|end_of_solution\|>', '</think>\n\n<answer>', text)
    # Add closing answer tag at the end if it doesn't exist
    if '<answer>' in text and not text.strip().endswith('</answer>'):
        text = text.strip() + '\n</answer>'
    
    return text

def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1]  # Return the last boxed answer
    return ""

class ReasoningDataset:
    def __init__(
        self,
        dataset_name: str = "AI-MO/NuminaMath-TIR",
        split: str = "train",
        max_length: int = 2048,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        self.dataset = self._load_and_preprocess()
    
    def _load_and_preprocess(self) -> Dataset:
        """Load and preprocess the dataset based on its type."""
        raw_dataset = load_dataset(self.dataset_name, split=self.split)
        
        processed_data = []
        
        # Determine dataset type and process accordingly
        if "Bespoke-Stratos" in self.dataset_name:
            for item in raw_dataset:
                processed_item = self._process_bespoke_item(item)
                if processed_item:
                    processed_data.append(processed_item)
        else:
            # Assume NuminaMath-TIR or similar format
            for item in raw_dataset:
                processed_item = self._process_numina_item(item)
                if processed_item:
                    processed_data.append(processed_item)
        
        return Dataset.from_list(processed_data)
    
    def _process_bespoke_item(self, item: Dict) -> Optional[Dict]:
        """Process Bespoke-Stratos dataset items."""
        try:
            conversations = item.get("conversations", [])
            if not conversations:
                return None
            
            # Get user message
            user_message = ""
            assistant_response = ""
            
            for conv in conversations:
                if conv["from"] == "user":
                    user_message = conv["value"]
                elif conv["from"] == "assistant":
                    assistant_response = conv["value"]
            
            if not user_message:
                return None
            
            # Convert to our conversation format
            conversation_data = make_conversation_bespoke(item)
            
            # Process assistant response if available (for reference/evaluation)
            reference_response = ""
            if assistant_response:
                reference_response = convert_bespoke_tags(assistant_response)
            
            return {
                "prompt": conversation_data["prompt"],
                "problem": user_message,
                "reference_response": reference_response,
                "dataset_type": "bespoke"
            }
        
        except Exception as e:
            print(f"Error processing Bespoke item: {e}")
            return None
    
    def _process_numina_item(self, item: Dict) -> Optional[Dict]:
        """Process NuminaMath-TIR dataset items."""
        try:
            problem = item.get("problem", "")
            solution = item.get("solution", "")
            
            if not problem:
                return None
            
            # Create conversation data
            conversation_data = make_conversation(item)
            
            # Extract answer for reference if solution exists
            reference_answer = ""
            if solution:
                reference_answer = extract_boxed_answer(solution)
            
            return {
                "prompt": conversation_data["prompt"],
                "problem": problem,
                "reference_solution": solution,
                "reference_answer": reference_answer,
                "dataset_type": "numina"
            }
        
        except Exception as e:
            print(f"Error processing NuminaMath item: {e}")
            return None
    
    def tokenize_prompt(self, prompt: List[Dict]) -> Dict:
        """Tokenize the prompt for training."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not provided")
        
        # Apply chat template to prompt only (no assistant response)
        text = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True  # This adds the assistant prompt
        )
        
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "text": text
        }
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        
        if self.tokenizer:
            tokenized = self.tokenize_prompt(item["prompt"])
            item.update(tokenized)
        
        return item
    
    def get_batch(self, batch_size: int = 1) -> List[Dict]:
        """Get a random batch of items."""
        import random
        indices = random.sample(range(len(self.dataset)), min(batch_size, len(self.dataset)))
        return [self[idx] for idx in indices]

def create_dataset(
    dataset_name: str = "AI-MO/NuminaMath-TIR",
    split: str = "train",
    max_length: int = 2048,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> ReasoningDataset:
    """Create a reasoning dataset."""
    return ReasoningDataset(
        dataset_name=dataset_name,
        split=split,
        max_length=max_length,
        tokenizer=tokenizer
    )