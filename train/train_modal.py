# This trains on cloud, evaluates locally
import modal
import torch
import wandb
from accelerate import Accelerator
import os
import numpy as np
from huggingface_hub import HfApi, HfFolder
import transformers
from tqdm import tqdm
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from transformers import DataCollatorForSeq2Seq, BitsAndBytesConfig
from datasets import load_dataset
from evaluate import load
from transformers import get_scheduler
from datetime import datetime
from torch.amp import autocast

app = modal.App("Training")

image = modal.Image.debian_slim().pip_install([
    "transformers", "torch", "wandb", "evaluate", "huggingface_hub", "datasets",
    "bert_score", "evaluate", "numpy", "peft", "accelerate", "bitsandbytes", "torchvision"
])

def setup_environment():
    HfFolder.save_token("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")
    wandb.login(key="f216d281d44a13dc6d5b7c00942e66e6f02f4d53")
    seed = 1
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)
    np.random.seed(seed)
    
    torch.cuda.empty_cache()

def load_model_and_tokenizer(lora_path):
    model_name = "AbdulmohsenA/Faseeh"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                  quantization_config=quantization_config,
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
    prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang="arb_Arab")
    
    # Apply LoRA
    print(f"Using path: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
        
    return model, tokenizer

def compute_metrics(preds, labels, tokenizer):
    metric = load("bertscore")
    
    labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    bertscore_results = metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        lang="ar",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    prediction_lengths = [(pred != tokenizer.pad_token_id).sum().item() for pred in preds]
    
    return {
        "precision": round(torch.tensor(bertscore_results['precision']).mean().item(), 4),
        "recall": round(torch.tensor(bertscore_results['recall']).mean().item(), 4),
        "f1": round(torch.tensor(bertscore_results['f1']).mean().item(), 4),
        "gen_len": round(np.mean(prediction_lengths), 4)
    }

def load_and_preprocess_data(tokenizer):
    dataset = load_dataset("Abdulmohsena/Classic-Arabic-English-Language-Pairs-Downsampled")['train']
    preprocess_function = lambda examples: tokenizer(
        examples['source'], text_target=examples['target'], max_length=256, truncation=True, padding=True, return_tensors='pt')
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['source', 'target'])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.20)
    return tokenized_dataset

@app.function(gpu="A10G", image=image, timeout=3600)
def train_model(wandb_run_id):
    setup_environment()

    wandb.init(project="Faseeh", id=wandb_run_id, resume="must")

    try:
        artifact = wandb.use_artifact('lora-adapter:latest', type='model')
        lora_dir = artifact.download()
    except Exception:
        import warnings
        warnings.warn("No WandB logged model found. Using AbdulmohsenA/Faseeh_LoRA...")
        lora_dir = "AbdulmohsenA/Faseeh_LoRA"

    model, tokenizer = load_model_and_tokenizer(lora_dir)
    BATCH_SIZE = 8

    tokenized_dataset = load_and_preprocess_data(tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors='pt')
    train_dataloader = torch.utils.data.DataLoader(tokenized_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)

    accelerator = Accelerator(mixed_precision="fp16")
    model = model.to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    lr_scheduler = get_scheduler(name="cosine",
                                optimizer=optimizer,
                                num_warmup_steps = len(train_dataloader) * 0.10,
                                num_training_steps = len(train_dataloader))

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    model.gradient_checkpointing_disable()
    model.gradient_checkpointing_kwargs = {"use_reentrant": False}

    model.train()
    train_loss = 0
    progress_bar = tqdm(train_dataloader)

    for step, batch in enumerate(progress_bar):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()

        # Logging
        if step % 100:
            progress_bar.set_postfix({"loss": loss.item()})
            wandb.log({"train_loss": loss.item()})

    model = accelerator.unwrap_model(model)

    lora_path = "./logs/lora_weights"
    model.save_pretrained(lora_path)

    # model.push_to_hub("Abdulmohsena/Faseeh_LoRA", token=True, max_shard_size="5GB", safe_serialization=True)

    artifact = wandb.Artifact('lora-adapter', type='model')
    artifact.add_dir(lora_path)
    wandb.log_artifact(artifact)

    wandb.finish()

def evaluate_model_locally(wandb_run_id):
    setup_environment()
    
    wandb.init(project="Faseeh", id=wandb_run_id, resume="must")
    artifact = wandb.use_artifact('lora-adapter:latest', type='model')
    lora_dir = artifact.download()

    model, tokenizer = load_model_and_tokenizer(lora_dir)
    tokenized_dataset = load_and_preprocess_data(tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors='pt')
    eval_dataloader = torch.utils.data.DataLoader(tokenized_dataset["test"], batch_size=4, collate_fn=data_collator)
    
    accelerator = Accelerator(mixed_precision="fp16")
    model = model.to(accelerator.device)
    
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )
    
    model.eval()
    eval_metrics = {"eval_loss": 0, "bertscore_f1": 0, "gen_length": 0}
    for batch in tqdm(eval_dataloader):
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        with torch.no_grad():
            outputs = model(**batch)
            
            predictions = torch.argmax(outputs.logits, dim=2)
            labels = batch['labels']
            metrics = compute_metrics(predictions, labels, tokenizer)
            
        eval_metrics["eval_loss"] += outputs.loss.detach().item()
        eval_metrics["bertscore_f1"] += metrics['f1']
        eval_metrics["gen_length"] += metrics['gen_len']


    eval_metrics["eval_loss"] /= len(eval_dataloader)
    eval_metrics["bertscore_f1"] /= len(eval_dataloader)
    eval_metrics["gen_length"] /= len(eval_dataloader)
    print(f"Evaluation metrics: {eval_metrics}")

    wandb.log(eval_metrics)
    wandb.finish()

@app.local_entrypoint()
def main():
    epochs = 1
    wandb_run = wandb.init(project="Faseeh", name=f"Faseeh @ {datetime.now()}")
    
    
    # Commit the starting run to enable multi environment runs
    wandb_run.finish()
    
    for _ in range(epochs):
        # train_model.remote(wandb_run.id)
        evaluate_model_locally(wandb_run.id)
        
    wandb_run.finish()