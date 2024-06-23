"""
qrsh -g $GROUP -l rt_G.small=1 -l h_rt=1:00:00
module load python/3.12/3.12.2 cuda/12.1/12.1.1
cd ~/prj/dl_lecture_competition_pub
source venv/bin/activate
python main_002.py
"""

import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor
from transformers import BertModel, BertTokenizer


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
        image_npy_file_path: str,
        include_answers=True,
    ):
        self.df = pd.read_json(ann_json_file_path)
        self.image_npy_file_path = image_npy_file_path
        self.include_answers = include_answers

        self.answer2idx = {}
        self.idx2answer = {}

        if self.include_answers:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

        self.all_images: np.ndarray | None = None  # (N_all, 224, 224, 3)
        self.image_npy_file_path = image_npy_file_path

    def load_all_images(self):
        self.all_images = np.load(self.image_npy_file_path)

    def clear_all_images(self):
        self.all_images = None

    def update_dict(self, dataset):
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        assert self.all_images is not None

        image = to_tensor(self.all_images[idx])
        question = process_text(self.df["question"][idx])

        if not self.include_answers:
            return image, question

        answer_idxs = []
        answer_distribution = torch.zeros(
            size=(len(self.answer2idx),), dtype=torch.float32
        )
        for answer in self.df["answers"][idx]:
            answer_idx = self.answer2idx[process_text(answer["answer"])]
            answer_idxs.append(answer_idx)
            answer_distribution[answer_idx] += 1.0
        answer_idxs = torch.tensor(answer_idxs, dtype=torch.int64)
        answer_distribution /= answer_distribution.sum()

        return (
            image,
            question,
            answer_distribution,
            answer_idxs,
        )

    def __len__(self):
        return len(self.df)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
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


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, stride=1
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


class VQAModel(nn.Module):
    def __init__(self, num_answers: int):
        super().__init__()

        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained(
            "bert-base-uncased", torch_dtype=torch.float32, attn_implementation="sdpa"
        )
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.resnet = ResNet18()
        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_answers),
        )

    def forward(self, image, question):
        N = image.shape[0]
        image_feature = self.resnet(image)
        assert image_feature.shape == (N, 512)
        with torch.no_grad():
            question = self.bert_tokenizer(
                question,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(image.device)
            question_feature = self.bert_model(**question).last_hidden_state[
                :, 0, :
            ]  # (N, 768)
            assert question_feature.shape == (N, 768)
        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


def train(model, dataloader, optimizer, device):
    model.train()

    total_loss = 0
    total_acc = 0

    start = time.time()
    for i, (image, question, answer_distribution, answer_idxs) in enumerate(dataloader):
        image, answer_distribution, answer_idxs = (
            image.to(device),
            answer_distribution.to(device),
            answer_idxs.to(device),
        )

        pred_distribution = model(image, question)
        loss = torch.nn.functional.kl_div(
            pred_distribution.log(),
            answer_distribution,
            reduction="batchmean",
            log_target=False,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = VQA_criterion(pred_distribution.argmax(1), answer_idxs)

        # print(f"{i:03d}, train loss : {loss:.4f}, train acc : {acc:.4f}")

        total_loss += loss.item()
        total_acc += acc

    return (
        total_loss / len(dataloader),
        total_acc / len(dataloader),
        time.time() - start,
    )


def main():
    fix_seed(42)

    # deviceの設定
    device = "cuda"

    # dataloader / model
    train_dataset = VQADataset(
        ann_json_file_path="./data/train.json",
        image_npy_file_path="data/train_224x224x3_uint8.npy",
        include_answers=True,
    )
    test_dataset = VQADataset(
        ann_json_file_path="./data/valid.json",
        image_npy_file_path="data/valid_224x224x3_uint8.npy",
        include_answers=False,
    )
    test_dataset.update_dict(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=5,
        pin_memory=True,
    )

    model = VQAModel(
        num_answers=len(train_dataset.answer2idx),
    ).to(device)

    # optimizer / criterion
    num_epoch = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # train model
    train_dataset.load_all_images()
    model.train()
    for epoch in range(num_epoch):
        train_loss, train_acc, train_time = train(
            model, train_dataloader, optimizer, device
        )
        print(
            f"【{epoch + 1}/{num_epoch}】\n"
            f"train time: {train_time:.2f} [s]\n"
            f"train loss: {train_loss:.4f}\n"
            f"train acc: {train_acc:.4f}\n"
        )

    # inference
    train_dataset.clear_all_images()
    test_dataset.load_all_images()
    model.eval()
    predict_step_outputs = []
    for image, question in test_dataloader:
        image = image.to(device)
        pred = model(image, question)
        predict_step_outputs.append(pred.argmax(dim=1).cpu())
    submission = torch.cat(predict_step_outputs, dim=0)
    submission = [train_dataset.idx2answer[id] for id in submission.tolist()]
    submission = np.array(submission)

    os.mkdir("outputs/002", mode=0o700)
    torch.save(model.state_dict(), "outputs/002/model.pth")
    np.save("outputs/002/submission.npy", submission)


if __name__ == "__main__":
    main()
