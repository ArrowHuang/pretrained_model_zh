# coding: UTF-8
import torch
import transformers
from transformers import XLNetTokenizer, XLNetForSequenceClassification

class Config(object):

    def __init__(self, dataset):
        self.model_name = 'xlnet'
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
        self.learning_rate = 5e-6                                      
        self.xlnet_path = './pre_trained/XLNet'
        self.tokenizer = XLNetTokenizer.from_pretrained(self.xlnet_path)
        self.hidden_size = 768                                          


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained(config.xlnet_path,num_labels=config.num_labels)
        for param in self.xlnet.parameters():
            param.requires_grad = True\

    def forward(self, x):
        context = x[0]  
        mask = x[2]  
        output = self.xlnet(context, attention_mask=mask)
        return output[0]
