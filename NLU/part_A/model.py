import torch
import torch.nn as nn

from typing import Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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


class NLUModel(nn.Module):

    def __init__(self,
                 emb_dim,
                 hidden_dim,
                 vocab_size,
                 out_dims: Tuple[int, int],

                 n_layers=1,
                 pad_index=0,

                 dropout=-1,
                 use_bidirectional = False):
        super(NLUModel, self).__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_index)

        self.lstm = nn.LSTM(emb_dim,
                            hidden_dim,
                            n_layers,
                            batch_first=True,
                            bidirectional=use_bidirectional)

        if use_bidirectional:
            # Bi-Directionality implies doubling the LSTM passes (forward and backward)
            # This creates two internal hidden states that will feed the final output layers
            hidden_dim *= 2

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.output_intents = nn.Linear(hidden_dim, out_dims[0])
        self.output_slots = nn.Linear(hidden_dim, out_dims[1])

    def forward(self, inputs, seq_lengths):

        embeddings = self.dropout(self.emb(inputs))

        # We need to pack the inputs to avoid LSTM processing padded tokens. This will save useful resources
        packed_embeddings = pack_padded_sequence(embeddings, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, (last_hidden, last_cell) = self.lstm(packed_embeddings)

        # Since we worked on packed sequences, we need to unpack them to get the original shape
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        # To get the calculated intent we need to have a look to the final state of the LSTM
        # LSTM cell process words one by one updating what it think the intent is up to all the words encountered
        # This is because intent is a sentence level property
        # NOTE: In case of bi-directionality we need to consider both forward/backward states by concatenating them
        # For bidirectional LSTM: last_hidden shape is (num_layers*2, batch, hidden)
        # [-2] is the final forward state, [-1] is the final backward state
        if self.lstm.bidirectional:
            last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
        else:
            last_hidden = last_hidden[-1]

        slots = self.output_slots(self.dropout(outputs)).permute(0, 2, 1)
        intents = self.output_intents(self.dropout(last_hidden))

        return slots, intents