from torch.utils.data import Dataset
from transformers import T5Tokenizer


class GrammarDataset(Dataset):
    def __init__(self, dataset, tokenizer: T5Tokenizer,
                 max_len: int = 64, print_text: bool = False):
        self.dataset = dataset
        self.pad_to_max_length = False
        self.tokenizer = tokenizer
        self.print_text = print_text
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def tokenize_data(self, example):
        input_, target_ = example['input'], example['output']

        tokenized_inputs = self.tokenizer(input_, pad_to_max_length=self.pad_to_max_length,
                                          max_length=self.max_len, return_attention_mask=True)
        tokenized_targets = self.tokenizer(target_, pad_to_max_length=self.pad_to_max_length,
                                           max_length=self.max_len, return_attention_mask=True)
        inputs = {"inputs_ids": tokenized_inputs['inputs_ids'],
                  "attention_mask": tokenized_inputs['attention_mask'],
                  "labels": tokenized_targets['input_ids']}
        return inputs

    def __getitem__(self, index):
        inputs = self.tokenize_data(self.dataset[index])

        if self.print_text:
            for k in inputs.keys():
                print(k, len(inputs[k]))
        return inputs
