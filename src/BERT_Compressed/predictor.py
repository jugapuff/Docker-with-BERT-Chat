from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, SingleSentenceClassificationProcessor
import torch
import numpy as np
import pickle
import random

with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)

###################
out_dir = "./out"
max_length = 43
##################


class Predictor:
    def __init__(self):
        config = BertConfig.from_pretrained("bert-base-multilingual-cased",
                                            num_labels=len(labels))
        self.model = BertForSequenceClassification.from_pretrained(
            out_dir, config=config)
        self.model.to("cuda")
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased", do_lower_case=False)
        self.processor = SingleSentenceClassificationProcessor()
        self.responses, self.alternatives = self.get_responses()

    def predict(self, input_message):
        inputs = self.process_input(input_message)
        outputs = self.model(**inputs)
        logits = outputs[0]
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)[0]

        return random.choice(self.alternatives[self.responses[labels[preds]]])

    def process_input(self, input_message):
        input_message = input_message.strip().strip(".")
        examples = self.processor.create_from_examples([input_message])
        features = examples.get_features(tokenizer=self.tokenizer,
                                         max_length=max_length)
        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features],
                                          dtype=torch.long)
        return {
            "input_ids": all_input_ids.to("cuda"),
            "attention_mask": all_attention_mask.to("cuda")
        }

    def get_responses(self):
        responses = {}
        alternatives = {}
        with open("domain.yml", "r") as f:
            s = f.read()
        for paragraph in s.strip().split("\n\n"):
            paragraph = paragraph.strip().split("\n")
            if "intents" in paragraph[0]:
                key = ""
                for line in paragraph[1:]:
                    if "  - " in line:
                        key = line[4:-1]
                    elif "triggers: utter_" in line:
                        responses[key] = line.split(
                            "triggers: utter_")[1].strip()
            if "templates" in paragraph[0]:
                key = ""
                value = []
                for line in paragraph[1:]:
                    if "utter_" in line:

                        if key:
                            alternatives[key] = value
                            value = []
                        key = line.split("utter_")[1][:-1]
                    elif "- text: " in line:
                        value.append(line.split("- text: \"")[1][:-1])

        return responses, alternatives
