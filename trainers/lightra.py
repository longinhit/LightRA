import copy
import os.path as osp
import numpy as np
import json
from typing import Tuple, Union, List
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Function
from torch.nn.parameter import Parameter

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        if isinstance(text, List):
            parse_custom_hook_func = text[1]
            text = text[0]
        else:
            parse_custom_hook_func = None
        
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if parse_custom_hook_func is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, parse_custom_hook_func])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

class ResidualAttention(nn.Module):
    def __init__(self, prefix_len, embed_dim, dtype):
        super().__init__()
        prefix_shape = (2, prefix_len, embed_dim)
        prefix_pool = torch.zeros(prefix_shape, dtype=dtype)
        torch.nn.init.uniform_(prefix_pool[0], -1, 1)
        self.prefix_pool = Parameter(prefix_pool)

    def forward(self, query, dropout_p, is_causal):
        bsz, num_heads, tgt_len, head_dim = query.shape
        prefix = self.prefix_pool.unsqueeze(0).expand(bsz, -1, -1, -1) # [bs, 2, prefix_len, embed_dim]
        _, _, prefix_len, embed_dim = prefix.shape  # [bs, 2, prefix_len, embed_dim]
        prefix_k = prefix[:, 0, ...].transpose(0, 1).contiguous()  # [prefix_len, bs, embed_dim]
        prefix_v = prefix[:, 1, ...].transpose(0, 1).contiguous()  # [prefix_len, bs, embed_dim]
        
        prefix_k = prefix_k.view(prefix_k.size(0), bsz * num_heads, head_dim).transpose(0, 1)  #for visual: [bs*12, prefix_len, 64]; for text: [n_cls*8, prefix_len, 64]
        prefix_v = prefix_v.view(prefix_v.size(0), bsz * num_heads, head_dim).transpose(0, 1)

        prefix_k = prefix_k.view(bsz, num_heads, prefix_len, head_dim)  #for visual: [bs, 12, prefix_len, 64]; for text: [n_cls, 8, prefix_len, 64]
        prefix_v = prefix_v.view(bsz, num_heads, prefix_len, head_dim)

        attn_output_prefix = F.scaled_dot_product_attention(query, prefix_k, prefix_v, None, dropout_p, is_causal) 
        attn_output_prefix = attn_output_prefix.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim) 
        return attn_output_prefix

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.mean_text_features = None
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        self.n_cls = len(classnames)
        self.LOSS_WEIGHT = cfg.TRAINER.LightRA.LOSS_WEIGHT
        self.COS_WEIGHT = cfg.TRAINER.LightRA.COS_WEIGHT 
        self.visualResidualAttention  = nn.ModuleList([None if i > cfg.TRAINER.LightRA.VISION_DEPTH
                                        else ResidualAttention(cfg.TRAINER.LightRA.N_CTX_TEXT, clip_model.visual.transformer.width, self.dtype)
                                        for i in range(clip_model.visual.transformer.layers)])
        
        self.textResidualAttention = nn.ModuleList([None if i > cfg.TRAINER.LightRA.TEXT_DEPTH
                                        else ResidualAttention(cfg.TRAINER.LightRA.N_CTX_VISION, clip_model.transformer.width, self.dtype)
                                        for i in range(clip_model.transformer.layers)])
        with torch.no_grad():
            self._build_prompt(cfg, classnames)
    
    def _build_prompt(self, cfg, classnames):
        prompt_template = cfg.DATASET.PROMPT_TEMPLATE

        prompt_ctxs = json.load(open(cfg.DATASET.PROMPT_PATH))
        prompt_ctxs = {k.lower().replace("_", " "): v for k, v in prompt_ctxs.items()}
       
        self.tk_prompts = []
        for cname in classnames:
            new_cname =  cname.lower().replace("_", " ")
            suffix = prompt_ctxs[new_cname]
            prompts = [f"{prompt_template.format(new_cname)} {ctx}" for ctx in suffix]
            tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts]).cuda()
            self.tk_prompts.append(tokenized_prompts)

        self.zs_text_features = []
        for cls_id in range(0, len(self.tk_prompts)):
            class_embeddings = self.text_encoder(self.tk_prompts[cls_id])
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            self.zs_text_features.append(class_embeddings)

    def parse_textResidualAttention(self, index):
        return self.textResidualAttention[index]
    
    def parse_visualResidualAttention(self, index):
        return self.visualResidualAttention[index]

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()
        tk_prompts = self.tk_prompts
        
        image_features = self.image_encoder([image.type(self.dtype), self.parse_visualResidualAttention])
        image_features = F.normalize(image_features, dim=-1)
        
        if self.training:
            with torch.no_grad():
                zs_image_features = self.image_encoder(image.type(self.dtype))
                zs_image_features = F.normalize(zs_image_features, dim=-1)

            zs_text_features = []
            sample_prompts = []
            for cls_id in range(0, len(tk_prompts)):
                n_num = tk_prompts[cls_id].shape[0]
                random_sample_indices = torch.randint(0, n_num, (1,), dtype=torch.long).item()
                sample_prompts.append(tk_prompts[cls_id][random_sample_indices,:])
                zs_text_features.append(self.zs_text_features[cls_id][random_sample_indices,:])
            zs_text_features = torch.stack(zs_text_features, 0)
            text_features = self.text_encoder([torch.stack(sample_prompts, 0), self.parse_textResidualAttention])
            text_features = F.normalize(text_features, dim=-1)
            
            logits_t = zs_image_features @ text_features.t()
            logits_i = image_features @ zs_text_features.t()
            logits = image_features @ text_features.t()
            logits_final = logit_scale * (self.COS_WEIGHT  * logits + logits_t + logits_i) / (self.COS_WEIGHT + 1 + 1)
            loss_scl_text = (1.0 - torch.mean(self.cos(text_features, zs_text_features))) 
            loss_scl_image = (1.0 - torch.mean(self.cos(image_features, zs_image_features))) 
            return logits_final, F.cross_entropy(logits_final, label) + (loss_scl_text + loss_scl_image) * self.LOSS_WEIGHT
        else:
            if self.mean_text_features is None:
                mean_text_features = []
                for cls_id in range(0, len(tk_prompts)):
                    class_embeddings = self.text_encoder([tk_prompts[cls_id], self.parse_textResidualAttention])
                    class_embeddings = F.normalize(class_embeddings, dim=-1)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embeddings = F.normalize(class_embeddings, dim=-1)
                    mean_text_features.append(class_embedding)
                self.mean_text_features  = torch.stack(mean_text_features) 
            mean_text_features = self.mean_text_features
            logits = logit_scale * (image_features @ mean_text_features.t())
            return logits


@TRAINER_REGISTRY.register()
class LightRA(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.LightRA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME)

        if cfg.TRAINER.LightRA.PREC == "fp32" or cfg.TRAINER.LightRA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.model.named_parameters():
            if  'visualResidualAttention' in name : 
                param.requires_grad_(True)
            elif 'textResidualAttention' in name : 
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
                
        # Double check
        enabled, param_groups, total_params = set(), [], 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                param_groups.append(param)
                total_params += param.numel() 
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}, total_params: {total_params}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('ResidualAttention', self.model, self.optim, self.sched)
      
        self.scaler = GradScaler() if cfg.TRAINER.LightRA.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.LightRA.PREC
        if prec == "amp":
            with autocast():
                logits, loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        # Average all parameters
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "zs_text_features" in state_dict:
                del state_dict["zs_mean_text_features"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = {}
            # Only save the fine-tuned parameters.
            for key, value in self._models[name].state_dict().items():
                if  'visualResidualAttention' in key  or 'textResidualAttention' in key: 
                    model_dict[key] = value 
            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )