import argparse
import os

import torch
from tqdm.auto import tqdm
from transformers import AdamW, AutoModelForMaskedLM, AutoTokenizer

from dataset import make_dataloader
from utils import set_seed


def train(model, tokenizer, data, labels, optimizer):
    device = model.device
    optimizer.zero_grad()
    tokens = tokenizer(data, return_tensors="pt", truncation=True, padding=True).to(
        device
    )
    labels = tokenizer(labels, return_tensors="pt", truncation=True, padding=True)[
        "input_ids"
    ].to(device)
    loss = model(**tokens, labels=labels).loss
    loss.backward()
    optimizer.step()


def main(args):
    set_seed()
    model_name = args.model_name
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name).to(device)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    optimizer = AdamW(model.parameters())
    mlm_dataloader = make_dataloader(os.path.join("data", "masked.txt"), batch_size)
    color_dataloader = make_dataloader(os.path.join("data", "color.txt"), batch_size)
    for i in range(args.num_epochs):
        # adaptive fine-tuning on color
        for data, labels in color_dataloader:
            train(model, tokenizer, data, labels, device)
        # continued pre-training on mlm
        for data, labels in mlm_dataloader:
            train(model, tokenizer, data, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_size", type=int, help="size of batch")
    parser.add_argument(
        "model_name", type=str, help="name of pretrained language model"
    )
    parser.add_argument("num_epochs", type=int, help="number of training epochs")
    args = parser.parse_args()
    main(args)
