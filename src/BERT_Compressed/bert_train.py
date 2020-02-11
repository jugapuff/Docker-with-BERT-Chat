import logging
from processor import train_dataloader, dataset, batch_size
from tqdm import tqdm, trange
from transformers import BertConfig, BertForSequenceClassification, AdamW
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import json
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)


logging.basicConfig(level=logging.INFO)

####################
epochs = 14
lr = 2e-5
logging_steps = 50
saving_steps = 2000

out_dir = "./out"
####################

train_iterator = trange(0, epochs, desc="Epoch")
config = BertConfig.from_pretrained("bert-base-multilingual-cased",
                                    num_labels=len(labels))
model = BertForSequenceClassification.from_pretrained(
    "./multi_cased_L-12_H-768_A-12", config=config)
model.to("cuda")

optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

print("***** 학습 과정이 시작됩니다! *****")
print("  Num examples =", len(dataset))
print("  Num Epochs =", epochs)
print("  Batchsize =", batch_size)

tr_loss, logging_loss = 0.0, 0.0
global_step = 0
model.zero_grad()
total_preds = []
total_out_label_ids = []

for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        model.train()
        batch = tuple(t.to("cuda") for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()

        tr_loss += loss.item()

        optimizer.step()
        model.zero_grad()
        global_step += 1

        if global_step % logging_steps == 0:
            with torch.no_grad():
                logits = outputs[1]
                preds = logits.detach().cpu().numpy()

                preds = np.argmax(preds, axis=1)

                out_label_ids = inputs["labels"].detach().cpu().numpy()

                total_preds.extend(preds)
                total_out_label_ids.extend(out_label_ids)
                #print(total_preds, total_out_label_ids)
            """
            c_matrix = confusion_matrix(total_out_label_ids,
                                        total_preds,
                                        labels=list(range(len(labels))))
            df_cm = pd.DataFrame(c_matrix, columns=labels)
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True)
            plt.savefig('./foo.png')
            """

            accuracy = np.sum(
                np.array([
                    int(total_out_label_ids[k] == total_preds[k])
                    for k in range(len(total_preds))
                ])) / len(total_preds)
            logs = {}
            loss_scalar = (tr_loss - logging_loss) / logging_steps
            logging_loss = tr_loss
            total_preds = []
            total_out_label_ids = []
            logs["loss"] = loss_scalar
            logs["accuracy"] = accuracy
            print(json.dumps({**logs, **{"step": global_step}}))
        if global_step % saving_steps == 0:
            model.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
