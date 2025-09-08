import torch
from torch import nn
import math
from typing import Optional

from transformers.models.bert import (
    BertPreTrainedModel,
    BertConfig,
)
from transformers.models.bert.modeling_bert import BertEncoder


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class mlp_meta(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config["hid_dim"], config["hid_dim"]),
            GeLU(),
            BertLayerNorm(config["hid_dim"], eps=1e-12),
            nn.Dropout(config["dropout"]),
        )

    def forward(self, x):
        return self.mlp(x)


class Bert_Transformer_Layer(BertPreTrainedModel):
    def __init__(self, fusion_config: dict):
        super().__init__(BertConfig(**fusion_config))
        bertconfig_fusion = BertConfig(**fusion_config)
        self.encoder = BertEncoder(bertconfig_fusion)
        self.init_weights()

    def forward(self, input, mask=None):
        """
        input:(bs, 4, dim)
        """
        batch, feats, dim = input.size()
        if mask is not None:
            mask_ = torch.ones(size=(batch, feats), device=mask.device)
            mask_[:, 1:] = mask
            mask_ = torch.bmm(
                mask_.view(batch, 1, -1).transpose(1, 2), mask_.view(batch, 1, -1)
            )
            mask_ = mask_.unsqueeze(1)

        else:
            mask = torch.Tensor([1.0]).to(input.device)
            mask_ = mask.repeat(batch, 1, feats, feats)

        extend_mask = (1 - mask_) * -10000
        assert not extend_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        # Call encoder with explicit output format control
        enc_output = self.encoder(
            input, 
            attention_mask=extend_mask, 
            head_mask=head_mask,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=self.config.return_dict
        )
        
        # Handle different output formats
        if hasattr(enc_output, 'last_hidden_state'):
            # If it's a dict-like object
            output = enc_output.last_hidden_state
            all_attention = getattr(enc_output, 'attentions', None)
        else:
            # If it's a tuple
            output = enc_output[0]
            
            # Safely handle attention weights - they might not be returned depending on config
            if len(enc_output) > 1 and enc_output[1] is not None:
                all_attention = enc_output[1]
            else:
                # If no attention weights returned, create dummy attention
                all_attention = None

        return output, all_attention


class MMDBaseModel(nn.Module):
    def __init__(
        self,
        config: dict,
        mlp_flag=True,
    ):
        super(MMDBaseModel, self).__init__()
        self.num_mlp = config["num_mlp"]
        self.transformer_flag = config["transformer_flag"]
        self.mlp_flag = mlp_flag
        token_num = config.get("token_num", 31)
        self.mlp = nn.Sequential(
            nn.Linear(config["in_dim"], config["hid_dim"]),
            GeLU(),
            BertLayerNorm(config["hid_dim"], eps=1e-12),
            nn.Dropout(config["dropout"]),
        )
        self.fusion_config = {
            "hidden_size": config["in_dim"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": 4,
            "output_attentions": True,
            "output_hidden_states": False,
            "return_dict": False,
            "_attn_implementation": "eager"
        }
        if self.num_mlp > 0:
            self.mlp2 = nn.ModuleList([mlp_meta(config) for _ in range(self.num_mlp)])
        if self.transformer_flag:
            self.transformer = Bert_Transformer_Layer(self.fusion_config)
        self.feature = nn.Linear(config["hid_dim"] * token_num, config["out_dim"])

    def forward1(self, features):
        features = features
        if self.transformer_flag:
            features, _ = self.transformer(features)
        return features

    def forward2(self, features):
        features = self.mlp(features)
        if self.num_mlp > 0:
            for _ in range(1):
                for mlp in self.mlp2:
                    features = mlp(features)
        features = self.feature(features.view(features.shape[0], -1))
        return features

    def forward(self, features):
        """
        input: [batch, token_num, hidden_size], output: [batch, token_num * config.out_dim]
        """

        if self.transformer_flag:
            features, _ = self.transformer(features)
        if self.mlp_flag:
            features = self.mlp(features)

        if self.num_mlp > 0:
            for _ in range(1):
                for mlp in self.mlp2:
                    features = mlp(features)

        features = self.feature(features.view(features.shape[0], -1))
        return features
    

class MMDModel:
    def __init__(self, config: dict, device: str):
        self.sigma = torch.tensor(config.get("sigma", 30.0) ** 2).to(device, dtype=torch.float)
        self.sigma0_u = torch.tensor(config.get("sigma0_u", 45.0) ** 2).to(device, dtype=torch.float)
        self.ep = torch.tensor(config.get("ep", 10.0) ** 2).to(device, dtype=torch.float)
        self.coeff_xy = config.get("coeff_xy", 2)
        self.is_yy_zero = config.get("is_yy_zero", False)
        self.is_xx_zero = config.get("is_xx_zero", False)

        self.basemodel = MMDBaseModel(config=config).to(device)

    def load_state_dict(self, ckpt_path: str):
        """Load a trained model from checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint["net"]

        # Remove "module." prefix if it exists
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_key = key[7:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
            
        self.sigma = torch.tensor(checkpoint["sigma"]).to(self.device, dtype=torch.float)
        self.sigma0_u = torch.tensor(checkpoint["sigma0_u"]).to(self.device, dtype=torch.float)
        self.ep = torch.tensor(checkpoint["ep"]).to(self.device, dtype=torch.float)
        
        self.basemodel.load_state_dict(state_dict)

    def save_state_dict(self, ckpt_path: str):
        """Save the model state dictionary."""
        torch.save({
            "net": self.basemodel.state_dict(),
            "sigma": self.sigma.item(),
            "sigma0_u": self.sigma0_u.item(),
            "ep": self.ep.item()
        }, ckpt_path)

        
    def __call__(self, x):
        """Forward pass through the model."""
        return self.basemodel(x)