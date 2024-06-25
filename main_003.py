"""
transformers ViLT

qrsh -g $GROUP -l rt_G.small=1 -l h_rt=1:00:00
module load python/3.12/3.12.2 cuda/12.1/12.1.1
cd ~/prj/dl_lecture_competition_pub
tmux new -s s0
source venv/bin/activate
python main_003.py
"""

import os
import pickle
import random
import re
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor
from transformers import (
    ViltForQuestionAnswering,
    ViltProcessor,
)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)

    # 冠詞の削除
    text = re.sub(r"\b(a|an|the)\b", "", text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't",
        "isnt": "isn't",
        "arent": "aren't",
        "wont": "won't",
        "cant": "can't",
        "wouldnt": "wouldn't",
        "couldnt": "couldn't",
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", " ", text)

    # 句読点をスペースに変換
    text = re.sub(r"\s+,", ",", text)

    # 連続するスペースを1つに変換
    text = re.sub(r"\s+", " ", text).strip()

    return text


class VQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_json_file_path: str,
        image_hdf5_file_path: str,
        answer2idx: dict,
        idx2answer: dict,
        include_answers=True,
    ):
        self.df = pd.read_json(ann_json_file_path)
        self.image_hdf5_file_path = image_hdf5_file_path
        self.answer2idx = answer2idx
        self.idx2answer = idx2answer
        self.include_answers = include_answers

        self.processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
        self.processor.image_processor.do_rescale = False
        self.processor.image_processor.do_resize = False

        self.all_images: np.ndarray | None = None  # (N_all, height, width, 3)

    def load_all_images(self):
        print(f"load all images from {self.image_hdf5_file_path}")
        with h5py.File(self.image_hdf5_file_path, "r") as f:
            self.all_images = f["images"][:, :, :, :]

    def clear_all_images(self):
        print(f"clear all images which were loaded from {self.image_hdf5_file_path}")
        self.all_images = None

    def __getitem__(self, idx):
        if self.all_images is None:
            with h5py.File(self.image_hdf5_file_path, "r") as f:
                image = f["images"][idx, :, :, :]
        else:
            image = self.all_images[idx]

        image = to_tensor(image)
        question = process_text(self.df["question"][idx])

        if not self.include_answers:
            return (image, question)

        answer_indexes = []
        answer_distribution = torch.zeros(
            size=(len(self.answer2idx),), dtype=torch.float32
        )

        for answer_dict in self.df["answers"][idx]:
            answer = process_text(answer_dict["answer"])
            answer_idx = self.answer2idx[answer]
            answer_indexes.append(answer_idx)
            answer_distribution[answer_idx] += 1.0

        answer_indexes = torch.tensor(answer_indexes, dtype=torch.int64)
        answer_distribution /= answer_distribution.sum()

        return (
            image,
            question,
            answer_distribution,
            answer_indexes,
        )

    def __len__(self):
        return len(self.df)

    def collate_fn(self, samples: list[tuple]) -> tuple:
        image = [sample[0] for sample in samples]
        question = [sample[1] for sample in samples]

        inputs = self.processor(
            image, question, padding=True, truncation=True, return_tensors="pt"
        )

        batch_size = len(samples)

        if not self.include_answers:
            return (batch_size, inputs)

        answer_distribution = torch.stack([sample[2] for sample in samples])
        answer_indexes = torch.stack([sample[3] for sample in samples])

        return (batch_size, inputs, answer_distribution, answer_indexes)


def calc_vqa_metric(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.0

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.0
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


def load_corpus():
    print("start loading corpus")

    with open("data/answer_corpus.pkl", "rb") as f:
        answer2idx, idx2answer = pickle.load(f)

    print("finish loading corpus")

    return answer2idx, idx2answer


def main():
    fix_seed(42)

    device = "cuda"

    answer2idx, idx2answer = load_corpus()

    train_dataset = VQADataset(
        ann_json_file_path="./data/train.json",
        image_hdf5_file_path="data/train_384x384x3_uint8.hdf5",
        answer2idx=answer2idx,
        idx2answer=idx2answer,
        include_answers=True,
    )
    test_dataset = VQADataset(
        ann_json_file_path="./data/valid.json",
        image_hdf5_file_path="data/valid_384x384x3_uint8.hdf5",
        answer2idx=answer2idx,
        idx2answer=idx2answer,
        include_answers=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=5,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn,
    )

    print("load vilt")

    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    model.config.num_labels = len(idx2answer)
    model.config.id2label = idx2answer
    model.config.label2id = answer2idx

    model.classifier = nn.Sequential(
        nn.Linear(model.config.hidden_size, model.config.hidden_size * 2),
        nn.LayerNorm(model.config.hidden_size * 2),
        nn.GELU(),
        nn.Linear(model.config.hidden_size * 2, model.config.num_labels),
    )

    num_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model.to(device)
    model.train()

    print("start training")

    for epoch in range(num_epochs):
        train_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        train_vqa_metric = torch.tensor(0.0, dtype=torch.float32, device=device)
        train_start_time = time.time()

        for batch_size, inputs, answer_distribution, answer_indexes in train_dataloader:
            answer_distribution = answer_distribution.to(device)
            answer_indexes = answer_indexes.to(device)

            for key in inputs.keys():
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(device)

            outputs = model(**inputs)
            logits = outputs.logits

            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits, dim=1),
                answer_distribution,
                reduction="batchmean",
                log_target=False,
            )
            vqa_metric = calc_vqa_metric(logits.argmax(1), answer_indexes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss * batch_size / len(train_dataset)
            train_vqa_metric += vqa_metric * batch_size / len(train_dataset)

        train_elapsed_time = time.time() - train_start_time

        print(
            f"【{epoch+1:02d} / {num_epochs:02d}】, train_loss : {train_loss:.4f}, train_vqa_metric : {vqa_metric:.4f}, elapsed_time : train_elapsed_time : {train_elapsed_time:.4f}"
        )

    model.eval()
    predict_step_outputs = []

    print("start inference")

    for batch_size, inputs in test_dataloader:
        for key in inputs.keys():
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)

        outputs = model(**inputs)
        logits = outputs.logits

        predict_step_outputs.append(logits.argmax(dim=1).cpu())

    submission = torch.cat(predict_step_outputs, dim=0)
    submission = [idx2answer[id] for id in submission.tolist()]
    submission = np.array(submission)

    print("create outputs directory")

    os.mkdir("outputs/003", mode=0o700)
    torch.save(model.state_dict(), "outputs/003/model.pth")
    np.save("outputs/003/submission.npy", submission)


if __name__ == "__main__":
    main()
