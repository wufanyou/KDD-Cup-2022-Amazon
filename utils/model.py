from typing import Union
from omegaconf import DictConfig
from torch import Tensor
import torch.nn as nn
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel  # type: ignore
__key__ = "model"
__all__ = ["get_model"]


def get_model(cfg: DictConfig):
    if cfg.model.architecture == "CrossEncoder":
        model = CrossEncoder(cfg.model.name, num_labels=cfg.model.num_labels).model

    elif cfg.model.architecture == "AutoModelForSequenceClassification":
        config = AutoConfig.from_pretrained(cfg.model.name)
        num_labels = cfg.model.num_labels
        classifier_trained = True
        if config.architectures is not None:
            classifier_trained = any(
                [
                    arch.endswith("ForSequenceClassification")
                    for arch in config.architectures
                ]
            )
        if num_labels is None and not classifier_trained:
            num_labels = 1
        if num_labels is not None:
            config.num_labels = num_labels
        config.pad_token_id = cfg.model.pad_token_id
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.name, config=config
        )
    else:
        module = eval(cfg.model.architecture)
        model = module(cfg)

    return model  # type: ignore


from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout
from transformers.activations import ACT2FN
from transformers.utils.generic import ModelOutput

# Copied from transformers.models.deberta.modeling_deberta.ContextPooler
class ContextConcat(nn.Module):
    def __init__(
        self,
        hidden_size,
        pooler_dropout,
        initializer_range,
        pooler_hidden_act,
        num_text_types=6,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size * num_text_types, hidden_size)
        self.dense.weight.data.normal_(mean=0, std=initializer_range)
        self.dense.bias.data.zero_()
        self.dropout = StableDropout(pooler_dropout)
        self.activation = ACT2FN[pooler_hidden_act]
        self.output_dim = hidden_size

    def forward(self, hidden_states, y, extra_feats=None):
        x = torch.arange(hidden_states.shape[0]).unsqueeze(-1).repeat([1, y.shape[1]]).reshape(-1)
        context_token = hidden_states[x, y.reshape(-1)]
        context_token = context_token.reshape(hidden_states.shape[0], -1)
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BiEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        if (cfg.export_to_onnx) and ("deberta" in cfg.model.name):
                from .deberta import DebertaV2Model # type: ignore
                config = AutoConfig.from_pretrained(cfg.model.name)
                self.model = DebertaV2Model(config)
        else:
            if cfg.model.use_pretrained:
                if "cocolm" in cfg.model.name:
                    from utils.cocolm import COCOLMModel, COCOLMConfig

                    config = COCOLMConfig.from_pretrained(cfg.model.name)
                    self.model = COCOLMModel.from_pretrained(
                        cfg.model.pretrained_path, config=config
                    )
                elif 'bigbird' in cfg.model.name:
                    config = AutoConfig.from_pretrained(cfg.model.name)
                    config.attention_type = 'original_full'
                    self.model = AutoModel.from_pretrained(cfg.model.name,config=config)
                else:
                    self.model = AutoModel.from_pretrained(cfg.model.pretrained_path)
            else:
                if "cocolm" in cfg.model.name:
                    from utils.cocolm import COCOLMModel, COCOLMConfig
                    config = COCOLMConfig.from_pretrained(cfg.model.name)
                    self.model = COCOLMModel.from_pretrained(cfg.model.name, config=config)
                elif 'bigbird' in cfg.model.name:
                    config = AutoConfig.from_pretrained(cfg.model.name)
                    config.attention_type = 'original_full'
                    self.model = AutoModel.from_pretrained(cfg.model.name,config=config)
                else:
                    self.model = AutoModel.from_pretrained(cfg.model.name)

        args = cfg[__key__]["architecture_args"]
        self.require_token_id = args["require_token_id"]
        self.context_concat = ContextConcat(
            hidden_size=self.model.config.hidden_size,
            pooler_dropout=args.pooler_dropout,
            initializer_range=args.initializer_range,
            pooler_hidden_act=args.pooler_hidden_act,
            num_text_types=args.num_text_types,
        )

        self.classifer = nn.Sequential(
            nn.Linear(
                self.model.config.hidden_size*2,
                args.linear_hidden_size,
            ),
            ACT2FN[args.pooler_hidden_act],
            nn.Linear(args.linear_hidden_size, cfg[__key__].num_labels),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0, std=args.initializer_range)
                m.bias.data.zero_()
        self.classifer.apply(init_weights)

    def forward(
        self,
        query: dict,
        product: Union[dict,Tensor],
        use_embedding = False,
        return_dict = True,
        **kwargs
    ):
        if self.require_token_id:
            query = self.model(
                input_ids=query['input_ids'],
                token_type_ids=query['token_type_ids'],
                attention_mask=query['attention_mask'],
                return_dict=False,
            )[0]

            if not use_embedding:
                product = self.model(
                    input_ids=product['input_ids'], # type: ignore
                    token_type_ids=product['token_type_ids'], # type: ignore
                    attention_mask=product['attention_mask'], # type: ignore
                    return_dict=False,
                )[0]
        else:
            query = self.model(
                input_ids=query['input_ids'],
                attention_mask=query['attention_mask'],
                return_dict=False,
            )[0]
            if not use_embedding:
                product = self.model(
                    input_ids=product['input_ids'], # type: ignore
                    token_type_ids=product['token_type_ids'], # type: ignore
                    attention_mask=product['attention_mask'], # type: ignore
                    return_dict=False,
                )[0]
        
        # query [B, N, 768]
        # product [B, M, 768]
        query = query[:,0]
        product = product[:,0] # type: ignore
        output = self.classifer(torch.cat([query, product], axis=1)) # type: ignore
        output = ModelOutput({"logits": output})
        if not return_dict:
            output = output.to_tuple()
        return output



class CrossEncoderConcat(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        if (cfg.export_to_onnx) and ("deberta" in cfg.model.name):
                from .deberta import DebertaV2Model # type: ignore
                config = AutoConfig.from_pretrained(cfg.model.name)
                self.model = DebertaV2Model(config)
        else:
            if cfg.model.use_pretrained:
                if "cocolm" in cfg.model.name:
                    from utils.cocolm import COCOLMModel, COCOLMConfig

                    config = COCOLMConfig.from_pretrained(cfg.model.name)
                    self.model = COCOLMModel.from_pretrained(
                        cfg.model.pretrained_path, config=config
                    )
                elif 'bigbird' in cfg.model.name:
                    config = AutoConfig.from_pretrained(cfg.model.name)
                    config.attention_type = 'original_full'
                    self.model = AutoModel.from_pretrained(cfg.model.name,config=config)
                else:
                    self.model = AutoModel.from_pretrained(cfg.model.pretrained_path)
            else:
                if "cocolm" in cfg.model.name:
                    from utils.cocolm import COCOLMModel, COCOLMConfig
                    config = COCOLMConfig.from_pretrained(cfg.model.name)
                    self.model = COCOLMModel.from_pretrained(cfg.model.name, config=config)
                elif 'bigbird' in cfg.model.name:
                    config = AutoConfig.from_pretrained(cfg.model.name)
                    config.attention_type = 'original_full'
                    self.model = AutoModel.from_pretrained(cfg.model.name,config=config)
                else:
                    self.model = AutoModel.from_pretrained(cfg.model.name)

        args = cfg[__key__]["architecture_args"]
        self.require_token_id = args["require_token_id"]
        self.context_concat = ContextConcat(
            hidden_size=self.model.config.hidden_size,
            pooler_dropout=args.pooler_dropout,
            initializer_range=args.initializer_range,
            pooler_hidden_act=args.pooler_hidden_act,
            num_text_types=args.num_text_types,
        )

        self.classifer = nn.Linear(
            self.model.config.hidden_size, cfg[__key__].num_labels
        )
        self.classifer.weight.data.normal_(mean=0, std=args.initializer_range)
        self.classifer.bias.data.zero_()

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        speical_token_pos: Tensor,
        return_dict=True,
        **kwargs
    ):
        if self.require_token_id:
            output = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
        else:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
        output = self.context_concat(output, speical_token_pos)
        output = self.classifer(output)
        output = ModelOutput({"logits": output})
        if not return_dict:
            output = output.to_tuple()
        return output


class CrossEncoderConcatExport(CrossEncoderConcat):
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        speical_token_pos: Tensor,
        #relative_pos: Tensor,
    ):
        if self.require_token_id:
            output = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
        else:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
        output = self.context_concat(output, speical_token_pos)
        output = self.classifer(output).argmax(1)
        return output

class CrossEncoderConcatExportLogit(CrossEncoderConcat):
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        speical_token_pos: Tensor,
        #relative_pos: Tensor,
    ):
        if self.require_token_id:
            output = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
        else:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
        output = self.context_concat(output, speical_token_pos)
        output = self.classifer(output)
        return output

class CrossEncoderConcatExtraFeats(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        if (cfg.export_to_onnx) and ("deberta" in cfg.model.name):
                from .deberta import DebertaV2Model # type: ignore
                config = AutoConfig.from_pretrained(cfg.model.name)
                self.model = DebertaV2Model(config)
        else:
            if cfg.model.use_pretrained:
                if "cocolm" in cfg.model.name:
                    from utils.cocolm import COCOLMModel, COCOLMConfig

                    config = COCOLMConfig.from_pretrained(cfg.model.name)
                    self.model = COCOLMModel.from_pretrained(
                        cfg.model.pretrained_path, config=config
                    )
                elif 'bigbird' in cfg.model.name:
                    config = AutoConfig.from_pretrained(cfg.model.name)
                    config.attention_type = 'original_full'
                    self.model = AutoModel.from_pretrained(cfg.model.name,config=config)
                else:
                    self.model = AutoModel.from_pretrained(cfg.model.pretrained_path)
            else:
                if "cocolm" in cfg.model.name:
                    from utils.cocolm import COCOLMModel, COCOLMConfig
                    config = COCOLMConfig.from_pretrained(cfg.model.name)
                    self.model = COCOLMModel.from_pretrained(cfg.model.name, config=config)
                elif 'bigbird' in cfg.model.name:
                    config = AutoConfig.from_pretrained(cfg.model.name)
                    config.attention_type = 'original_full'
                    self.model = AutoModel.from_pretrained(cfg.model.name,config=config)
                else:
                    self.model = AutoModel.from_pretrained(cfg.model.name)

        args = cfg[__key__]["architecture_args"]
        self.require_token_id = args["require_token_id"]
        self.context_concat = ContextConcat(
            hidden_size=self.model.config.hidden_size,
            pooler_dropout=args.pooler_dropout,
            initializer_range=args.initializer_range,
            pooler_hidden_act=args.pooler_hidden_act,
            num_text_types=args.num_text_types,
        )

        self.classifer = nn.Sequential(
            nn.Linear(
                self.model.config.hidden_size + args.extra_feats_num,
                args.linear_hidden_size,
            ),
            ACT2FN[args.pooler_hidden_act],
            nn.Linear(args.linear_hidden_size, cfg[__key__].num_labels),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0, std=args.initializer_range)
                m.bias.data.zero_()

        self.classifer.apply(init_weights)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        speical_token_pos: Tensor,
        extra: Tensor,
        return_dict=True,
        **kwargs
    ):

        if self.require_token_id:
            output = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
        else:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]

        output = self.context_concat(output, speical_token_pos)  # [b,128]
        output = torch.cat([output, extra], 1)
        output = self.classifer(output)
        output = ModelOutput({"logits": output})
        if not return_dict:
            output = output.to_tuple()
        return output
