import torch
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForMaskedLM, EsmTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
import gc
from tqdm import tqdm

# Configuration
INPUT_DIR = "/scratch/akj/DMS_ProteinGym_substitutions"
OUTPUT_DIR = "/scratch/akj/DMS_ProteinGym_embeddings"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # ESM-2 650M model
BATCH_SIZE = 4
MAX_LENGTH = 1024
TEST_SIZE = 0.2
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
LORA_RANK = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset class for protein mutations
class ProteinMutationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        mutation_sequence = item['mutated_sequence']
        fitness_score = item['DMS_score']
        
        # Tokenize sequence
        inputs = self.tokenizer(
            mutation_sequence,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(fitness_score, dtype=torch.float)
        }

# Function to extract embeddings from a model
def extract_embeddings(model, tokenizer, dataloader):
    model.eval()
    all_embeddings = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            # Get the model's hidden states
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get the last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            
            # Average the embeddings over the sequence length (excluding padding)
            # This gives a single vector per sequence
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            masked_hidden = last_hidden_state * mask_expanded
            sum_embeddings = torch.sum(masked_hidden, dim=1)
            seq_lengths = torch.sum(attention_mask, dim=1, keepdim=True)
            mean_embeddings = sum_embeddings / seq_lengths
            
            all_embeddings.append(mean_embeddings.cpu().numpy())
            
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    return embeddings

# Define a simplified trainer that only optimizes the LoRA parameters
class FitnessTrainer(Trainer):
    def compute_loss(self, model, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use the last hidden state to predict fitness
        last_hidden = outputs.hidden_states[-1]
        
        # Average pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        masked_hidden = last_hidden * mask_expanded
        sum_hidden = torch.sum(masked_hidden, dim=1)
        seq_lengths = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled_output = sum_hidden / seq_lengths
        
        # Simple prediction head (just a linear layer)
        prediction = self.model.lm_head(pooled_output).squeeze(-1)
        
        # MSE Loss for regression
        loss = torch.nn.functional.mse_loss(prediction, labels)
        
        return loss

def main():
    # Load tokenizer
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    
    # Get all CSV files
    all_csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    # Filter out files that already have embeddings and their original counterparts
    processed_files = set()
    for file in all_csv_files:
        basename = os.path.basename(file)
        if "_with_embeddings" in basename:
            # Add both the embedding file and its original counterpart to exclude list
            processed_files.add(basename)
            processed_files.add(basename.replace("_with_embeddings", ""))
    
    # Files to process
    files_to_process = [f for f in all_csv_files if os.path.basename(f) not in processed_files]
    
    print(f"Found {len(all_csv_files)} total CSV files")
    print(f"Already processed {len(processed_files) // 2} files")
    print(f"Processing {len(files_to_process)} remaining files")
    
    # Process each CSV file
    for csv_file in tqdm(files_to_process, desc="Processing files"):
        filename = os.path.basename(csv_file)
        protein_name = os.path.splitext(filename)[0]
        output_file = os.path.join(OUTPUT_DIR, f"{protein_name}_with_embeddings.csv")
        
        # Check if output already exists
        if os.path.exists(output_file):
            print(f"Embeddings already exist for {protein_name}, skipping...")
            continue
        
        print(f"\n--- Processing {protein_name} ---")
        
        try:
            # Load data
            data = pd.read_csv(csv_file)
            
            # Check if we have the required columns
            if 'mutated_sequence' not in data.columns or 'DMS_score' not in data.columns:
                print(f"Missing required columns in {csv_file}, skipping...")
                continue
            
            # Split data into train and test sets
            train_data, test_data = train_test_split(
                data, test_size=TEST_SIZE, random_state=42
            )
            
            print(f"Train set size: {len(train_data)}, Test set size: {len(test_data)}")
            
            # Load base model with minimal memory footprint
            base_model = EsmForMaskedLM.from_pretrained(
                MODEL_NAME,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=["query", "value"],
                bias="none",
            )
            
            # Create LoRA model
            model = get_peft_model(base_model, peft_config)
            model.to(DEVICE)
            
            # Enable gradient checkpointing to save memory
            model.gradient_checkpointing_enable()
            
            # Create datasets and dataloaders
            train_dataset = ProteinMutationDataset(train_data, tokenizer)
            test_dataset = ProteinMutationDataset(test_data, tokenizer)
            
            # Fine-tune the model
            training_args = TrainingArguments(
                output_dir=f"./results/{protein_name}",
                num_train_epochs=NUM_EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                weight_decay=0.01,
                save_strategy="no",
                report_to="none",
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                fp16=True,
                logging_steps=10
            )
            
            # Custom trainer
            trainer = FitnessTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )
            
            print(f"Fine-tuning model for {protein_name}...")
            trainer.train()
            
            # Create a dataloader for the full dataset (to generate embeddings)
            full_dataset = ProteinMutationDataset(data, tokenizer)
            full_dataloader = DataLoader(
                full_dataset, 
                batch_size=BATCH_SIZE,
                shuffle=False
            )
            
            # Extract embeddings
            print(f"Generating embeddings for {protein_name}...")
            embeddings = extract_embeddings(model, tokenizer, full_dataloader)
            
            # Create a DataFrame with embeddings
            embedding_df = pd.DataFrame(embeddings)
            embedding_columns = [f"embedding_{i}" for i in range(embeddings.shape[1])]
            embedding_df.columns = embedding_columns
            
            # Combine original data with embeddings
            result_df = pd.concat([data.reset_index(drop=True), embedding_df], axis=1)
            
            # Save to CSV
            result_df.to_csv(output_file, index=False)
            print(f"Saved embeddings to {output_file}")
            
            # Clean up to free memory
            del model, base_model, train_dataset, test_dataset, full_dataset
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {protein_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()