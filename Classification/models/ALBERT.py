# coding: UTF-8
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForSequenceClassification, BertTokenizer

# ['voidful/albert_chinese_tiny', 'voidful/albert_chinese_small', 'voidful/albert_chinese_base', 'voidful/albert_chinese_large', 'voidful/albert_chinese_xlarge', 'voidful/albert_chinese_xxlarge']

class Config(object):

    def __init__(self, dataset, task):
        self.model_name = 'albert'
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
        self.batch_size = 32                                          
        self.pad_size = 512                                              
        self.learning_rate = 5e-5                                      
        self.albert_path = 'voidful/albert_chinese_base'
        self.tokenizer = BertTokenizer.from_pretrained(self.albert_path)
        self.hidden_size = 768                                          


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.albert = AutoModelForSequenceClassification.from_pretrained(config.albert_path,num_labels = config.num_classes)
        for param in self.albert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  
        mask = x[2]  
        output = self.albert(context, attention_mask=mask)
        return output[0]
