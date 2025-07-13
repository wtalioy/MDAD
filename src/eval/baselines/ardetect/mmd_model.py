import torch
from torch import nn
import math
from dataclasses import dataclass

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
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hid_dim, config.hid_dim),
            GeLU(),
            BertLayerNorm(config.hid_dim, eps=1e-12),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class Bert_Transformer_Layer(BertPreTrainedModel):
    def __init__(self, fusion_config):
        # Ensure the config has the required fields
        config_dict = {
            "hidden_size": fusion_config.get("hidden_size", 768),
            "num_hidden_layers": fusion_config.get("num_hidden_layers", 1),
            "num_attention_heads": fusion_config.get("num_attention_heads", 4),
            "output_attentions": fusion_config.get("output_attentions", True),
            "output_hidden_states": fusion_config.get("output_hidden_states", False),
            "return_dict": fusion_config.get("return_dict", False),
            "intermediate_size": fusion_config.get("intermediate_size", fusion_config.get("hidden_size", 768) * 4),
            "hidden_act": fusion_config.get("hidden_act", "gelu"),
            "hidden_dropout_prob": fusion_config.get("hidden_dropout_prob", 0.1),
            "attention_probs_dropout_prob": fusion_config.get("attention_probs_dropout_prob", 0.1),
        }
        
        bertconfig_fusion = BertConfig(**config_dict)
        super().__init__(bertconfig_fusion)
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
    

@dataclass
class MMDConfig:
    in_dim: int = 1920
    hid_dim: int = 512
    out_dim: int = 300
    token_num: int = 31
    dropout: float = 0.2
    num_mlp: int = 0
    transformer_flag: bool = True
    num_hidden_layers: int = 1
default_config = MMDConfig()


class mmdModel(nn.Module):
    def __init__(
        self,
        config: MMDConfig,
        mlp_flag=True,
    ):
        super(mmdModel, self).__init__()
        self.num_mlp = config.num_mlp
        self.transformer_flag = config.transformer_flag
        self.mlp_flag = mlp_flag
        token_num = config.token_num
        self.mlp = nn.Sequential(
            nn.Linear(config.in_dim, config.hid_dim),
            GeLU(),
            BertLayerNorm(config.hid_dim, eps=1e-12),
            nn.Dropout(config.dropout),
        )
        self.fusion_config = {
            "hidden_size": config.in_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": 4,
            "output_attentions": True,
            "output_hidden_states": False,
            "return_dict": False,  # Ensure tuple output for backward compatibility
        }
        if self.num_mlp > 0:
            self.mlp2 = nn.ModuleList([mlp_meta(config) for _ in range(self.num_mlp)])
        if self.transformer_flag:
            self.transformer = Bert_Transformer_Layer(self.fusion_config)
        self.feature = nn.Linear(config.hid_dim * token_num, config.out_dim)

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
    

class ModelLoader:
    def __init__(self, sigma, sigma0_u, ep, config: MMDConfig, state_dict: dict, device: str):
        self.sigma = sigma
        self.sigma0_u = sigma0_u
        self.ep = ep

        self.model = mmdModel(config=config).to(device)
        self.model.load_state_dict(state_dict)

    @staticmethod
    def from_pretrained(model_path: str, config: MMDConfig = default_config, device: str = "cuda"):
        """Load a trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=device)
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
            
        sigma = checkpoint.get("sigma", 30.0)
        sigma0_u = checkpoint.get("sigma0_u", 45.0) 
        ep = checkpoint.get("ep", 10.0)
        
        return ModelLoader(sigma, sigma0_u, ep, config, state_dict, device)
    
    def __call__(self, x):
        """Forward pass through the model."""
        return self.model(x)