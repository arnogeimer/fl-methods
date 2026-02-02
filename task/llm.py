import random
import torch
import json
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging
)
from peft import LoraConfig, get_peft_model
from typing import List, OrderedDict, Tuple
import tqdm
import numpy as np
from collections import OrderedDict
from datasets import load_dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.disable()

# -------------------------
# 1. Dataset generation
# -------------------------

logging.set_verbosity_error()


def format_example(ex):
    return (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n{ex['output']}"
    )


# -------------------------
# 2. Torch dataset
# -------------------------

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.texts = [format_example(ex) for ex in data]
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item


def load_data(
        num_datasets: int = 3,
        dataset_size: int = 100,
        percs: List[List[float]] = None,
    ) -> Tuple[List[Dataset], Dataset]:

    datasets = []
    trainset = load_dataset("tatsu-lab/alpaca", split="train")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    for i in range(num_datasets):
        raw_data = trainset.shuffle().select(range(dataset_size))
        dataset = InstructionDataset(raw_data, tokenizer)
        datasets.append(dataset)

    testset = InstructionDataset(
        trainset.shuffle().select(range(1000)),
        tokenizer
    )

    return (datasets, testset)

def train(
    model,
    trainset: Dataset,
    epochs: int = 5,
    ):


    train_size = int(0.8 * len(trainset))
    test_size = len(trainset) - train_size
    train_ds, test_ds = random_split(trainset, [train_size, test_size])

    optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
)
    
    # Training args
    args = TrainingArguments(
        output_dir="./toy_lora_out",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        logging_steps=50,
        do_eval=True,
        save_steps=10**9,   # effectively disables saving
        report_to="none",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        #optimizers=(optimizer, None),
    )

    trainer.train()

def test(
    model,
    testset,
) -> float:

    # Training args
    args = TrainingArguments(
        output_dir="./toy_lora_out",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=50,
        do_eval=True,
        save_steps=10**9,   # effectively disables saving
        report_to="none",
        fp16=True,
        dataloader_num_workers=4,   # or 4
        dataloader_pin_memory=True,
        prediction_loss_only=True,
    )
    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=None,
        eval_dataset=testset,
    )
    with torch.inference_mode():
        loss = trainer.evaluate()['eval_loss']
    return float(loss)


def get_model():
    model_name = "distilgpt2"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model



def ndarrays_from_model(model: AutoModelForCausalLM) -> dict:
    return OrderedDict(
        (k, v.detach().cpu().numpy())
        for k, v in model.state_dict().items()
        if "lora_" in k
    )

def weighted_average_lora(param_dicts, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    avg = OrderedDict()
    for key in param_dicts[0]:
        if "lora_" in key:
            stacked = np.stack([p[key] for p in param_dicts], axis=0)
            avg[key] = np.tensordot(weights, stacked, axes=(0, 0))
    return avg

def ndarrays_to_model(model, params):
    state = model.state_dict()
    with torch.no_grad():
        for k in params:
            state[k].copy_(
                torch.from_numpy(params[k]).to(
                    device=state[k].device,
                    dtype=state[k].dtype
                )
            )
    model.load_state_dict(state)

