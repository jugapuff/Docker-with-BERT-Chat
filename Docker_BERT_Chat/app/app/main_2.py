from flask import Flask
from flask_restful import Resource, Api, reqparse
from transformers import BertForSequenceClassification, GPT2LMHeadModel, AutoTokenizer
from kogpt2_transformers import get_kogpt2_tokenizer
import torch.nn.functional as F
from time import time
import torch
import random
import csv
import re
import pickle
from time import sleep


app = Flask(__name__)
api = Api(app)
print("Load Reranker model & tokenizer")
reranker_model = BertForSequenceClassification.from_pretrained("/app/app/models/reranker/checkpoint-920", num_labels=2)
reranker_tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
reranker_tokenizer.add_special_tokens({"additional_special_tokens":["[/]"]})
reranker_model.resize_token_embeddings(len(reranker_tokenizer))
reranker_model = reranker_model.to("cuda")
reranker_model.eval()

# Load Classifier model & tokenizer
print("Load Classifier model & tokenizer")
classifier_model = BertForSequenceClassification.from_pretrained("/app/app/models/classifier/checkpoint-190", num_labels=167)
classifier_tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
classifier_model = classifier_model.to("cuda")
classifier_model.eval()

# Load Generator model & tokenizer
print("Load Generator model & tokenizer")
generator_model = GPT2LMHeadModel.from_pretrained("/app/app/models/generator/checkpoint-851")
generator_tokenizer = get_kogpt2_tokenizer()
generator_tokenizer.add_special_tokens({"additional_special_tokens": ["<chatbot>"]})
generator_model.resize_token_embeddings(len(generator_tokenizer))
generator_model = generator_model.to("cuda")
generator_model.eval()


history = []
candidates = []


with open("/app/app/label_dic", 'rb') as f:
    temp_dic = pickle.load(f) 
labels = sorted(temp_dic.keys())
class Inference(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument(
        "query", 
        type=str, 
        required=True, 
        help='"Query" field cannot be left blank!'
        )

    def post(self):
        global history
        global candidates
        data = Inference.parser.parse_args()
        query = data["query"]
        #         history = data["history"].split(">><<")
        #         if history == ['']:
        #             history = []
        #         candidates = data["candidates"].split(">><<")
        #         if len(data["candidates"])<2:
        #             candidates = []

        if query == "일상 대화 초기화" or query == "일상대화 초기화":
            history = []
            candidates = []
            return {"sentence":"일상 대화 초기화 완료"}

        updated_history = []
        updated_candidates = []
        history.append(query)
        batch = classifier_tokenizer(query,
                                    add_special_tokens=True, 
                                    truncation = True,
                                    return_token_type_ids=True, 
                                    padding= True, 
                                    return_tensors="pt").to("cuda")
        
        classified = classifier_model(**batch)[0]
        classified = torch.topk(classified, k=1, dim=-1)


        if classified.values[0,0].item() > 5.6:
            final_res = random.choice(temp_dic[labels[classified.indices[0,0].item()]])

            if final_res not in history:
                
                updated_history = history + [final_res]
                updated_candidates = random.choices(candidates, k=int(len(candidates) * 0.7))
                sleep(0.3)
                
                
                #return {"updated_history":updated_history, "final_res":final_res, "updated_candidates":updated_candidates }
                history = updated_history
                candidates = updated_candidates
                return {"sentence":final_res }


        tokenized_query_for_generator = generator_tokenizer(query+"<chatbot>", 
                                                            add_special_tokens=False, 
                                                            return_tensors='pt').to("cuda")
        before_gen_beam2 = time()
