import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index


def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs[
        'question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)

    # change label to -100 for question tokens
    for i in range(num_question_tokens):
        label[i] = -100
    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)


class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split="forget10", loss_type="idk"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ⭐ 自动推导路径
        base_dir = os.path.dirname(data_path)
        forget_file = os.path.join(base_dir, f"{split}.json")
        retain_ratio = 100 - int(split.replace("forget", ""))
        retain_file = os.path.join(base_dir, f"retain{str(retain_ratio).zfill(2)}.json")

        print(f"[DATA] forget_file: {forget_file}")
        print(f"[DATA] retain_file: {retain_file}")

        self.forget_data = datasets.load_dataset(
            'json', data_files=forget_file, split='train'
        )
        self.retain_data = datasets.load_dataset(
            'json', data_files=retain_file, split='train'
        )

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            # 修正 idontknow.jsonl 路径（原反斜杠错误）
            self.idontknowfile = os.path.join(base_dir, "idontknow.jsonl")
            self.idk = open(self.idontknowfile, "r", encoding="utf-8").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            # use questions from forget set if split is idk or forget
            if data_type == "retain":
                # 修复：避免覆盖 idx，使用独立变量
                retain_idx = (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
                data = self.retain_data
                cur_idx = retain_idx
            else:
                data = self.forget_data
                cur_idx = idx

            question = data[cur_idx]['question']
            answer = data[cur_idx]['answer']

            if data_type == "idk":
                # get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length, question, answer, self.model_configs
            )
            rets.append(converted_data)
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split="forget10", ):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ⭐ 自动推导路径（修复原错误：不能都用 data_path）
        base_dir = os.path.dirname(data_path)
        forget_file = os.path.join(base_dir, f"{split}.json")
        retain_ratio = 100 - int(split.replace("forget", ""))
        retain_file = os.path.join(base_dir, f"retain{str(retain_ratio).zfill(2)}.json")

        print(f"[DATA] forget_file: {forget_file}")
        print(f"[DATA] retain_file: {retain_file}")

        self.forget_data = datasets.load_dataset('json', data_files=forget_file, split='train')
        self.retain_data = datasets.load_dataset('json', data_files=retain_file, split='train')

        self.idontknowfile = os.path.join(base_dir, "idontknow.jsonl")  # 修正路径
        self.idk = open(self.idontknowfile, "r", encoding="utf-8").readlines()

        self.model_configs = get_model_identifiers_from_yaml(model_family)

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in ["idk", "forget", "retain"]:
            if data_type == "retain":
                retain_idx = (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
                data = self.retain_data
                cur_idx = retain_idx
            else:
                data = self.forget_data
                cur_idx = idx

            question = data[cur_idx]['question']
            if data_type != "idk":
                answer = data[cur_idx]['answer']
            else:
                # get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length, question, answer, self.model_configs
            )
            rets.append(converted_data)
        return rets


# 以下类基本保持不变，仅修复明显路径/加载问题（如果有）
class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split=None, question_key='question',
                 answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = datasets.load_dataset(
            'json',
            data_files=data_path,
            split='train'
        )
        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]
        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []
        for answer in answers:
            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length, question, answer, self.model_configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])
        return torch.stack(pad_input_ids_list).squeeze(), \
            torch.stack(label_list).squeeze(), \
            torch.stack(pad_attention_mask_list).squeeze(), \
            torch.tensor(indices)


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss