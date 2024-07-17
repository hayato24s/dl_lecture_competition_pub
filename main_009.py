"""
transformers ViLT

compared to 004, trannval ratio is [0.95, 0.05].

qrsh -g $GROUP -l rt_G.large=1 -l h_rt=1:00:00
tmux new -s s0
source venv/bin/activate
python main_009.py train

qrsh -g $GROUP -l rt_G.small=1 -l h_rt=1:00:00
tmux new -s s0
source venv/bin/activate
python main_009.py predict
"""

import pickle
import random
import re
import sys
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
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
            return {
                "image": image,
                "question": question,
            }

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

        return {
            "image": image,
            "question": question,
            "answer_indexes": answer_indexes,
            "answer_distribution": answer_distribution,
        }

    def __len__(self):
        return len(self.df)

    def collate_fn(self, samples: list[tuple]) -> dict:
        image = [sample["image"] for sample in samples]
        question = [sample["question"] for sample in samples]

        inputs = self.processor(
            image, question, padding=True, truncation=True, return_tensors="pt"
        )

        batch_size = len(samples)

        if not self.include_answers:
            return {
                "batch_size": batch_size,
                "inputs": inputs,
            }

        answer_distribution = torch.stack(
            [sample["answer_distribution"] for sample in samples]
        )
        answer_indexes = torch.stack([sample["answer_indexes"] for sample in samples])

        return {
            "batch_size": batch_size,
            "inputs": inputs,
            "answer_distribution": answer_distribution,
            "answer_indexes": answer_indexes,
        }

    @classmethod
    def load_corpus(cls):
        with open("data/answer_corpus.pkl", "rb") as f:
            answer2idx, idx2answer = pickle.load(f)
        return answer2idx, idx2answer


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


class LitModule(L.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=[
                {"params": self.model.parameters()},
            ],
            lr=1e-4,
            weight_decay=1e-5,
        )
        return optimizer

    def training_step(self, batch: dict, batch_idx: int):
        batch_size = batch["batch_size"]
        inputs = batch["inputs"]
        answer_distribution = batch["answer_distribution"]
        answer_indexes = batch["answer_indexes"]

        outputs = self.model(**inputs)
        logits = outputs.logits

        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits, dim=1),
            answer_distribution,
            reduction="batchmean",
            log_target=False,
        )
        vqa_metric = calc_vqa_metric(logits.argmax(1), answer_indexes)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_vqa_metric",
            vqa_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        batch_size = batch["batch_size"]
        inputs = batch["inputs"]
        answer_distribution = batch["answer_distribution"]
        answer_indexes = batch["answer_indexes"]

        outputs = self.model(**inputs)
        logits = outputs.logits

        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits, dim=1),
            answer_distribution,
            reduction="batchmean",
            log_target=False,
        )
        vqa_metric = calc_vqa_metric(logits.argmax(1), answer_indexes)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_vqa_metric",
            vqa_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def predict_step(self, batch: dict, batch_idx: int):
        inputs = batch["inputs"]

        outputs = self.model(**inputs)
        logits = outputs.logits
        pred = logits.argmax(dim=1)

        return pred.cpu()


def main_train():
    fix_seed(42)

    output_dir_path = Path("outputs/009")
    output_dir_path.mkdir(mode=0o700, parents=True, exist_ok=True)

    answer2idx, idx2answer = VQADataset.load_corpus()

    train_valid_dataset = VQADataset(
        ann_json_file_path="./data/train.json",
        image_hdf5_file_path="data/train_384x384x3_uint8.hdf5",
        answer2idx=answer2idx,
        idx2answer=idx2answer,
        include_answers=True,
    )
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_valid_dataset, lengths=[0.95, 0.05]
    )
    setattr(train_dataset, "collate_fn", train_dataset.dataset.collate_fn)
    setattr(valid_dataset, "collate_fn", valid_dataset.dataset.collate_fn)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=48,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=48,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        collate_fn=valid_dataset.collate_fn,
    )

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

    model.train()

    litmodule = LitModule(model=model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir_path.joinpath("checkpoints"),
        filename="{epoch:03d}-{val_vqa_metric:.4f}",
        monitor="val_vqa_metric",
        save_top_k=5,
        mode="max",
    )

    trainer = L.Trainer(
        accelerator="gpu",
        logger=TensorBoardLogger(save_dir=output_dir_path.joinpath("tensorboard")),
        callbacks=[checkpoint_callback],
        max_epochs=30,
        deterministic=True,
        detect_anomaly=False,
    )

    trainer.fit(
        model=litmodule,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


def main_predict():
    fix_seed(42)

    output_dir_path = Path("outputs/009")
    output_dir_path.mkdir(mode=0o700, parents=True, exist_ok=True)

    answer2idx, idx2answer = VQADataset.load_corpus()

    predict_dataset = VQADataset(
        ann_json_file_path="./data/valid.json",
        image_hdf5_file_path="data/valid_384x384x3_uint8.hdf5",
        answer2idx=answer2idx,
        idx2answer=idx2answer,
        include_answers=False,
    )

    predict_dataloader = torch.utils.data.DataLoader(
        predict_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        collate_fn=predict_dataset.collate_fn,
    )

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

    litmodule = LitModule(model=model)

    trainer = L.Trainer(
        accelerator="gpu",
        deterministic=True,
        detect_anomaly=False,
    )

    litmodule = LitModule.load_from_checkpoint(
        checkpoint_path="outputs/009/checkpoints/epoch=022-val_vqa_metric=0.6006.ckpt",
        map_location="cpu",
        model=model,
    )
    torch.save(litmodule.model.state_dict(), output_dir_path.joinpath("model.pt"))

    predict_step_outputs = trainer.predict(
        model=litmodule,
        dataloaders=predict_dataloader,
        return_predictions=True,
    )

    submission = torch.cat(predict_step_outputs, dim=0)
    submission = [idx2answer[id] for id in submission.tolist()]
    submission = np.array(submission)

    np.save(output_dir_path.joinpath("submission.npy"), submission)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        main_train()
    elif sys.argv[1] == "predict":
        main_predict()
