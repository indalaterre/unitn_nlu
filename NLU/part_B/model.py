from typing import Tuple

import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel

class UncertaintyWeighedLoss(nn.Module):

    def __init__(self, learnable_tasks):
        super(UncertaintyWeighedLoss, self).__init__()

        ## Uncertainty weighting from Kendall/Gal/Cipolla https://arxiv.org/abs/1705.07115
        ## The goal is the learn the weight for each loss minimizing the overall one
        ## This class is generic (accounts for a variable number of tasks but let's take our example of 2)
        ## We have the loss for the intents and the loss for the slots
        ## Given var1 = variance of intents loss
        ##       var2 = variance of slots loss
        ##       learnable_w1 and learnable_w2 are learnable parameters (they act as regularizers)
        ## Uncertainty weighting states that the total loss is
        ## 1/var1 * intent_loss + 1/var2 * slots_loss + learnable_w1 + learnable_w2

        ## To avoid 0-divisions and ensure variance is positive we can use the log of the variances
        self.log_vars = nn.Parameter(torch.zeros(learnable_tasks))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            # 1/var = exp(-log_var)
            total_loss += torch.exp(-self.log_vars[i]) * loss + self.log_vars[i]

        return total_loss

    def get_losses_weights(self):
        return [torch.exp(-self.log_vars[i]).item() for i in range(len(self.log_vars))]

class NLUBertModel(BertPreTrainedModel):

    def __init__(self,
                 config,
                 out_dims: Tuple[int, int],
                 dropout=0.1,
                 freeze_bert=False):
        super(NLUBertModel, self).__init__(config)

        self.bert = BertModel(config)
        # NOTE: I tried to freeze the BERT model as it was a pre-trained one
        # but this impacted on the overall model performance
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        self.output_intents = nn.Linear(self.config.hidden_size, out_dims[0])

        self.output_slots = nn.Linear(self.config.hidden_size, out_dims[1])

    def forward(self, token_ids, attention_mask):

        bert_output = self.bert(token_ids, attention_mask=attention_mask)

        # get the last hidden states for slots and the pooled output for intents
        sequence_output = self.dropout(bert_output.last_hidden_state)
        pooled_output = self.dropout(bert_output.pooler_output)

        intent_out = self.output_intents(pooled_output)
        slot_out = self.output_slots(sequence_output).permute(0, 2, 1)

        return slot_out, intent_out