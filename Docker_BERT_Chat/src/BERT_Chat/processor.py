from torch.utils.data import DataLoader, RandomSampler, TensorDataset, WeightedRandomSampler
from transformers import BertTokenizer, SingleSentenceClassificationProcessor
import torch

############################
max_length = 43
batch_size = 16
############################

processor = SingleSentenceClassificationProcessor()

examples = processor.create_from_csv("data.csv")
labels = examples.labels
import pickle

with open("labels.pickle", "wb") as f:
    pickle.dump(labels, f)

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased",
                                          do_lower_case=False)
features = examples.get_features(tokenizer=tokenizer, max_length=max_length)
all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_attention_mask = torch.tensor([f.attention_mask for f in features],
                                  dtype=torch.long)
# all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
#dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
weight = torch.sum(torch.zeros(len(all_labels), len(labels)).scatter_(
    1, all_labels.unsqueeze(dim=1), 1.0),
                   dim=0).double() / len(all_labels)
weight = 1 / (weight + 1e-8)

samples_weight = [weight[l] for l in all_labels.numpy()]
train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
#train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset,
                              sampler=train_sampler,
                              batch_size=batch_size)
