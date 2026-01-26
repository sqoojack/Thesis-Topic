# utils.py
import torch
import torch.nn.functional as F
from config import Config

def calc_fre_loss(logits, target_ids, config: Config):
    """
    計算 Forward Reasoning Enhancement Loss 
    L = (L_origin + r * L_enhanced) / (r + 1)
    """
    shift_logits = logits
    shift_labels = target_ids
    
    # 計算每個 token 的 loss
    loss_per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction='none'
    )
    loss_per_token = loss_per_token.view(shift_labels.size())
    
    # L_origin (平均 loss)
    l_origin = loss_per_token.mean()
    
    # L_enhanced: 只取前 h 個 tokens 的 loss 
    h = min(config.FRE_H, loss_per_token.size(1))
    l_enhanced = loss_per_token[:, :h].mean()
    
    # 最終加權 Loss
    total_loss = (l_origin + config.FRE_R * l_enhanced) / (config.FRE_R + 1)
    
    return total_loss

def get_embeddings(model, input_ids):
    """獲取 Embeddings，用於梯度計算"""
    return model.get_input_embeddings()(input_ids)