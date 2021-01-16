# coding: UTF-8
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

class Config(object):

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/train.txt'                                
        self.dev_path = dataset + '/val.txt'                                   
        self.test_path = dataset + '/test.txt'                                  
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt').readlines()]                    
        self.log_path = dataset + '/run/log/' + self.model_name
        self.save_path = dataset + '/run/saved_dict/' + self.model_name + '.ckpt'   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')              

        self.require_improvement = 1000   # early stop                                
        self.num_classes = len(self.class_list) 
        self.num_epochs = 20                                          
        self.batch_size = 16                                          
        self.pad_size = 512                                              
        self.learning_rate = 5e-6                                      
        self.bert_path = './pre_trained/BERT'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768                                          


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
#         self.bert = BertModel.from_pretrained(config.bert_path,num_labels = config.num_classes)
        self.bert = BertForSequenceClassification.from_pretrained(config.bert_path,num_labels = config.num_classes)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  
        mask = x[2]  
#         sequence_output, pooled_output = self.bert(context, attention_mask=mask, return_dict=False)
#         out = self.fc(pooled_output)
#         return out

        output = self.bert(context, attention_mask=mask)
        return output[0]
