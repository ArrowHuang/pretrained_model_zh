# coding: UTF-8
import torch
import torch.nn as nn
import transformers
from transformers import ElectraTokenizer, ElectraForSequenceClassification

class Config(object):

    def __init__(self, dataset):
        self.model_name = 'electra'
        self.train_path = dataset + '/train.txt'                                
        self.dev_path = dataset + '/val.txt'                                   
        self.test_path = dataset + '/test.txt'                                  
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt').readlines()]                    
        self.log_path = dataset + '/run/log/' + self.model_name
        self.save_path = dataset + '/run/saved_dict/' + self.model_name + '.ckpt'   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')              

        self.require_improvement = 1000                                   
        self.num_labels = len(self.class_list)                         
        self.num_epochs = 20                                          
        self.batch_size = 16                                          
        self.pad_size = 512                                              
        self.learning_rate = 1e-4                                      
        self.electra_path = './pre_trained/ELECTRA'
        self.tokenizer = ElectraTokenizer.from_pretrained(self.electra_path)
        self.hidden_size = 768                                          


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.electra = ElectraForSequenceClassification.from_pretrained(config.electra_path,num_labels=config.num_labels)
        for param in self.electra.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        context = x[0]  
        mask = x[2]  
        output = self.electra(context, attention_mask=mask)
        return output[0]
