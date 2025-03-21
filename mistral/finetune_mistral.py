from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("csv", data_files="data/augmented_dataset.csv")["train"]

def preprocess(example):
    prompt = f"Tu es un agent intelligent. Génère une requête SPARQL pour répondre à la question suivante : {example['question']}\nRequête SPARQL : "
    full_text = prompt + example['sparql']
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir="./models/mistral_sparql_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_total_limit=2,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,
    optim="paged_adamw_8bit"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

model.save_pretrained("./models/mistral_sparql_lora")
tokenizer.save_pretrained("./models/mistral_sparql_lora")