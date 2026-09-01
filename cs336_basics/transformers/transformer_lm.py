import torch
from torch import nn

from cs336_basics.transformers.embedding import Embedding
from cs336_basics.transformers.transformer_block import TransformerBlock
from cs336_basics.transformers.RMSNorm import RMSNorm
from cs336_basics.transformers.linear import Linear
from cs336_basics.transformers.softmax import softmax



class TransformerLM(nn.Module):
   def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
      super().__init__()

      self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
      self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype) for _ in range(num_layers)])
      self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
      self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
     
      

   def forward(self, in_indices):

      #Convert indices into embedding vectors (B, T) -> (B, T, C)
      x = self.token_embeddings(in_indices)
      #Pass though all the layers(blocks)
      for layer in self.layers:
         x = layer(x)
      # Final norm layer
      x = self.ln_final(x)
      #Final LM head
      x = self.lm_head(x)
      return x
      
      
      



      