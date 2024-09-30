import logging
import math
import random
from metrics.tvg.eval_video import find_number
from metrics.tvg.eval_tvg import iou
import json
import os
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import numpy as np
from utils.format_tvg import extract_time
from sklearn.metrics import mean_squared_error
from ivcr.common.registry import registry
from ivcr.models.blip2 import Blip2Base, disabled_train
from transformers import LlamaForCausalLM
# from ivcr.models.Qformer import BertEncoder
from transformers import LlamaTokenizer, BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
import logging
import copy
from ivcr.models.Qformer import BertConfig, BertLMHeadModel
from vllm import LLM
from ivcr.conversation.conversation_video_batch import StoppingCriteriaSub
from ivcr.common.constant import DEFAULT_IMAGE_PATCH_TOKEN,VIDEO_INDEX_FIRST,VIDEO_INDEX_SECOND,VIDEO_INDEX_THIRD,VIDEO_INDEX_FOUR,VIDEO_INDEX_FIVE,\
VIDEO_INDEX_SIX,VIDEO_INDEX_SEVEN,VIDEO_INDEX_EIGHT,VIDEO_INDEX_NINE,VIDEO_INDEX_TEN,DEFAULT_VIDEO_START_TOKEN,DEFAULT_VIDEO_END_TOKEN

# from flamingo_pytorch import PerceiverResampler



@registry.register_model("IVCR")
class IVCR(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_llama_v2": "configs/models/ivcr.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
            self,
            vit_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            frozen_llama_proj=True,
            frozen_video_Qformer=True,

            llama_proj_model='',
            fusion_header_type="seqTransf",
            max_frame_pos=32,
            fusion_head_layers=2,
            num_video_query_token=32,
            lora=False,
            qformer_text_input=False,
            lora_inference_mode=True,
            window_size=0,
            stride=0,
            tokenizer = None,
    ):
        super().__init__()
        
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        ) #self.visual_encoder.num_features 1408
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            print("use text input for Qformer")
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        self.qformer_text_input = qformer_text_input
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        logging.info('Loading LLAMA Tokenizer')
        self.llama_tokenizer = tokenizer
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token
        
        self.vocab = self.llama_tokenizer.get_vocab()
        logging.info('Loading LLAMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                load_in_8bit=True,
                # device_map={'': device_8bit}
            )
        else:
            if max_txt_len > 2048:
                logging.info(f"interpolate llama model's rope from 2048 to {max_txt_len}")
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.bfloat16,
                    max_position_embeddings=max_txt_len,
                    rope_scaling={
                        "type": "linear",
                        "factor": 2.0
                    }
                )
            else:
                # self.llama_model = LLM(llama_model)
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    # torch_dtype=torch.float16,
                    load_in_8bit=True
                )
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.use_special_token = os.getenv("use_special_token")
        if self.use_special_token=="True":
            print("执行到此处了")
            self.llama_tokenizer.add_tokens([DEFAULT_VIDEO_START_TOKEN],special_tokens = True)
            self.llama_tokenizer.add_tokens([DEFAULT_VIDEO_END_TOKEN],special_tokens = True)
            self.llama_tokenizer.add_tokens([VIDEO_INDEX_FIRST],special_tokens = True)
            self.llama_tokenizer.add_tokens([VIDEO_INDEX_SECOND],special_tokens = True)
            self.llama_tokenizer.add_tokens([VIDEO_INDEX_THIRD],special_tokens = True)
            self.llama_tokenizer.add_tokens([VIDEO_INDEX_FOUR],special_tokens = True)
            self.llama_tokenizer.add_tokens([VIDEO_INDEX_FIVE],special_tokens = True)
            self.llama_tokenizer.add_tokens([VIDEO_INDEX_SIX],special_tokens = True)
            self.llama_tokenizer.add_tokens([VIDEO_INDEX_SEVEN],special_tokens = True)
            self.llama_tokenizer.add_tokens([VIDEO_INDEX_EIGHT],special_tokens = True)
            self.llama_tokenizer.add_tokens([VIDEO_INDEX_NINE],special_tokens = True)
            self.llama_tokenizer.add_tokens([VIDEO_INDEX_TEN],special_tokens = True)
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        #激活梯度检查点
        if use_grad_checkpoint:
            logging.info("use gradient checkpointing for LLAMA")
            self.llama_model.gradient_checkpointing_enable()

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLAMA Done')

        self.lora = lora
        if self.lora:
            logging.info('Using LORA')
            from peft import LoraConfig, get_peft_model, TaskType
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=lora_inference_mode,#false
                r=32,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
            )
            self.llama_model = get_peft_model(self.llama_model, config)
            self.llama_model.print_trainable_parameters()

        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        logging.info('Loading llama_proj Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)
        #self.Qformer.config.hidden_size is 768
        self.num_video_query_token = num_video_query_token
        self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(num_query_token=num_video_query_token, \
                                                                              vision_width=self.Qformer.config.hidden_size,
                                                                              num_hidden_layers=2)
        ## 将video_qformer中的cls，wordembeding置为空 
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.window_size = window_size
        self.stride = stride

        if frozen_video_Qformer:
            #  todo frozen  llama_proj
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False

            logging.info('video_Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            logging.info('video_Qformer is not frozen')
        self.cross_fn = nn.CrossEntropyLoss()
    def id2text(self,id):
        for key,value in self.vocab.items():
            if value == id:
                return key
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_videoQformer_visual(self, image, timestamp=None,is_video_clip=False):
        device = image.device
        # input shape b,c,t,h,w
        batch_size, _, time_length, _, _ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) #96 257 1408
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) #  b*t,32,embed_dim
            timestamps_input_ids = timestamp["input_ids"].to(device)
            timestamps_attention_mask = timestamp["attention_mask"].to(device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, timestamps_attention_mask], dim=1)
            query_output = self.Qformer.bert(
                timestamps_input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            # add frame_pos embedding
            q_hidden_state = query_output.last_hidden_state  #n_frms，32，768
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # if self.window_size <= 0:
            if not is_video_clip:
                frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size,
                                                        t=time_length)
                frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
                video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1,
                                                                    -1)  # expand on batch dim

                video_query_output = self.video_Qformer.bert(
                    query_embeds=video_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
                # video_hidden = query_output.last_hidden_state  #1*32*768
                video_hidden = video_query_output.last_hidden_state
                inputs_llama = self.llama_proj(video_hidden)
                atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
            else:
                # use clips
                inputs_llama_list, atts_llama_list = [], []
                for i in range(0, time_length, self.stride):
                    clip_hidden_state = frame_hidden_state[:, i:i + self.window_size, ...]
                    clip_hidden_state = einops.rearrange(clip_hidden_state, 'b t q h -> b (t q) h', b=batch_size)
                    clip_atts = torch.ones(clip_hidden_state.size()[:-1], dtype=torch.long).to(device)
                    video_query_tokens = self.video_query_tokens.expand(clip_hidden_state.shape[0], -1,
                                                                        -1)  # expand on batch dim

                    video_query_output = self.video_Qformer.bert(
                        query_embeds=video_query_tokens,
                        encoder_hidden_states=clip_hidden_state,
                        encoder_attention_mask=clip_atts,
                        return_dict=True,
                    )
                    video_hidden = video_query_output.last_hidden_state  # [bsz, t, dim]

                    inputs_llama = self.llama_proj(video_hidden)
                    atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
                    inputs_llama_list.append(inputs_llama)
                    atts_llama_list.append(atts_llama)

                inputs_llama = torch.cat(inputs_llama_list, dim=1)  # [bsz, t, dim]
                atts_llama = torch.cat(atts_llama_list, dim=1)  # [bsz, t]
        
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            # print(prompt)
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            if self.lora:  # peft
                p_before_embeds = self.llama_model.get_base_model().model.embed_tokens(
                    p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.get_base_model().model.embed_tokens(
                    p_after_tokens.input_ids).expand(batch_size, -1, -1)
            else:
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1,
                                                                                                        -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1,
                                                                                                      -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])

            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img
    
    def forward(self, samples,flag):
        # for samples in samples_list:
        if 'conv_type' in samples.keys() and samples['conv_type'] == 'multi':
            im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
            image = samples["images"]
            input_ids = samples['input_ids']
            category = samples['category']
            clip_num_patch_tokens = None
            img_embeds = None
            img_embeds_list, atts_img_list, num_patch_tokens_list = [], [], []
            if isinstance(image, list):  # nb of frames of some videos is less than ${num_frm}
                assert isinstance(samples["timestamps"], list)
                for img, timestamp in zip(image[0], samples["timestamps"]):
                    img = img.unsqueeze(0)
                    if len(img.size()) == 4:
                        time = 1
                        img = einops.repeat(img, 'b c h w -> b c t h w', t=time)
                    num_patch_tokens = self.num_video_query_token * math.ceil(
                        img.shape[2] / self.stride) if self.stride > 0 else self.num_video_query_token
                    img_embeds, atts_img = self.encode_videoQformer_visual(img, timestamp=timestamp,is_video_clip=False)
                    img_embeds_list.append(img_embeds)
                    atts_img_list.append(atts_img)
                    num_patch_tokens_list.append(num_patch_tokens)
                img_embeds = img_embeds_list
                atts_img = atts_img_list
            else:  # nb of frames of all videos is ${num_frm}
                if len(image.size()) == 4:
                    time = 1
                    image = einops.repeat(image, 'b c h w -> b c t h w', t=time)
                clip_num_patch_tokens = self.num_video_query_token * math.ceil(
                    image.shape[2] / self.stride) if self.stride > 0 else self.num_video_query_token  #96
                img_embeds, atts_img = self.encode_videoQformer_visual(image, timestamp=samples["timestamps"],is_video_clip=True)
            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[temp_input_ids == im_patch_token_id] = 0
            if self.lora:
                temp_input_embedding = self.llama_model.get_base_model().model.embed_tokens(temp_input_ids)
            else:
                temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
                cur_image_features = img_embeds[cur_image_idx]  # [num_video_query_token, dim]
                cur_new_input_embeds = None
                if isinstance(image, list):
                    num_patch_tokens = num_patch_tokens_list[cur_image_idx]
                    masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                    start_index = masked_indices[0]
                    all_token = (cur_input_ids == im_patch_token_id).sum()

                    #为了新的video_id 来设计的视频token插入方式
                    mask_index = 0
                    for index,num_token in enumerate(num_patch_tokens_list):
                        if index == 0:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:start_index], img_embeds[index].squeeze(0)), dim=0)
                        elif index==len(num_patch_tokens_list)-1:
                            cur_new_input_embeds = torch.cat((cur_new_input_embeds,cur_input_embeds[start_mask:end_mask],
                                                                img_embeds[index].squeeze(0),
                                                                cur_input_embeds[end_mask+num_token:]),dim=0)
                            break
                        else:
                            cur_new_input_embeds = torch.cat((cur_new_input_embeds, cur_input_embeds[start_mask:end_mask],
                                                                img_embeds[index].squeeze(0)),dim=0)
                        mask_index += num_token
                        start_mask = masked_indices[mask_index-1] + 1
                        end_mask = masked_indices[mask_index]
                    
                    #旧的插入方式
                    # for index, patch_token in enumerate(num_patch_tokens_list):
                    #     if index == 0:
                    #         cur_new_input_embeds = torch.cat((cur_input_embeds[:start_index], img_embeds[index].squeeze(0)), dim=0)
                    #     elif index==len(num_patch_tokens_list)-1:
                    #         cur_new_input_embeds = torch.cat((cur_new_input_embeds,img_embeds[index].squeeze(0),
                    #                                         cur_input_embeds[start_index+all_token:]),dim=0)
                    #     else:
                    #         cur_new_input_embeds = torch.cat((cur_new_input_embeds,
                    #                                         img_embeds[index].squeeze(0)),dim=0)
                        
                else:
                    if (cur_input_ids == im_patch_token_id).sum() != clip_num_patch_tokens:
                        raise ValueError(
                            "The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                    
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + clip_num_patch_tokens,
                                                    device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")

                    cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features,
                                                    cur_input_embeds[mask_index_start + clip_num_patch_tokens:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)

                cur_image_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            targets = samples['labels']
            attention_mask = samples['attention_mask']
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
                # logger = logging.getLogger('ivcr.train')
                # indice = torch.where(targets[0]==-100)
                # end_index = indice[0].shape
                # end_index = end_index[0]
                # pre_logits = outputs.logits
                # pre_logits = pre_logits.argmax(dim=-1)
                # pre_text = self.llama_tokenizer.batch_decode(pre_logits[:,end_index-1:], skip_special_tokens=False, clean_up_tokenization_spaces=True)
                # output = pre_text[0]
                #     # indice = torch.where(targets[0]==-100)[0].shape[0]
                #     # output = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1)[:, indice-1: ])
                # logger.info(output)
                return {"loss":outputs.loss}
                iou_loss = 0
                index_loss = 0
                """
                if flag:
                    indice = torch.where(targets[0]==-100)
                    end_index = indice[0].shape
                    end_index = end_index[0]
                    pre_logits = outputs.logits
                    pre_logits = pre_logits.argmax(dim=-1)
                    pre_text = self.llama_tokenizer.batch_decode(pre_logits[:,end_index-1:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pre_text = pre_text[0]
                    logger.info(pre_text)
                    first_part = pre_text.split('.')[0]
                    if "temporal video grounding" in first_part:
                        pre_temporal = extract_time(pre_text)
                        gt_temporal = samples['gt_value'][0]
                        if pre_temporal == []:
                            iou_loss = 1.
                        elif len(pre_temporal[0]) == 2:
                                iou_loss = iou(pre_temporal[0],gt_temporal)
                        else:
                            iou_loss = 1.
                    elif "video retrieval" in first_part:
                        iou_loss = 1.
                        second_part = pre_text.split('.')[1]
                        index = find_number(second_part)-1
                        if 0<=index<=9:
                            pre_index = torch.full((1,10),0.01)
                            pre_index[0][index] = 10
                            gt_index = samples['gt_value']
                            gt_index[0] = gt_index[0]-1
                            gt = torch.tensor(gt_index)
                            index_loss = self.cross_fn(pre_index,gt)
                        else:
                            index_loss = 0.
                    loss  = outputs.loss
                    loss = loss + 0.01*((1-iou_loss) + index_loss)
                    return {"loss":loss}
                else:
                    loss = outputs.loss
                    return {"loss": loss}
                """
        else:
            image = samples["image"]

            if len(image.size()) != 5:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w', t=time)
            #no timestamp
            img_embeds, atts_img = self.encode_videoQformer_visual(image)

            if self.prompt_list:
                prompt = random.choice(self.prompt_list)
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

            self.llama_tokenizer.padding_side = "right"
            
            text = [t + self.end_sym for t in samples["text_input"]]

            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(image.device)

            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
                           dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                             dtype=to_regress_tokens.input_ids.dtype,
                             device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            if self.lora:
                bos_embeds = self.llama_model.get_base_model().model.embed_tokens(bos)
                to_regress_embeds = self.llama_model.get_base_model().model.embed_tokens(to_regress_tokens.input_ids)
            else:
                bos_embeds = self.llama_model.model.embed_tokens(bos)
                to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            atts_bos = atts_img[:, :1]

            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg, tokenizer):
        vit_model = cfg.get("vit_model",
                            "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth")
        q_former_model = cfg.get("q_former_model",
                                 "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        lora = cfg.get("lora", False)
        lora_inference_mode = cfg.get("lora_inference_mode", False)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)

        llama_proj_model = cfg.get("llama_proj_model", '')

        fusion_header_type = cfg.get("fusion_header_type", 'seqTransf')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        fusion_head_layers = cfg.get("fusion_head_layers", 2)
        num_video_query_token = cfg.get("num_video_query_token", 32)

        qformer_text_input = cfg.get("qformer_text_input", False)
        window_size = cfg.get("window_size", 0)
        stride = cfg.get("stride", 0)
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            fusion_header_type=fusion_header_type,
            max_frame_pos=max_frame_pos,
            fusion_head_layers=fusion_head_layers,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            num_video_query_token=num_video_query_token,
            llama_proj_model=llama_proj_model,
            lora=lora,
            qformer_text_input=qformer_text_input,
            lora_inference_mode=lora_inference_mode,
            window_size=window_size,
            stride=stride,
            tokenizer= tokenizer,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=True)
            old_frame_pos_embed_size = ckpt['model']['video_frame_position_embedding.weight'].size()
            new_frame_pos_embed_size = model.video_frame_position_embedding.weight.size()
            if old_frame_pos_embed_size != new_frame_pos_embed_size:
                from ivcr.processors.video_processor import interpolate_frame_pos_embed
                print(
                    f'video_frame_position_embedding size is not the same, interpolate from {old_frame_pos_embed_size} to {new_frame_pos_embed_size}')
                ckpt['model']['video_frame_position_embedding.weight'] = interpolate_frame_pos_embed(
                    ckpt['model']['video_frame_position_embedding.weight'], new_n_frm=new_frame_pos_embed_size[0])
            msg = model.load_state_dict(ckpt['model'], strict=False)
            # logger.info(msg)
            # print(f"没有匹配的网络参数{msg}")
        ckpt_path_2 = cfg.get("ckpt_2", "")
        if ckpt_path_2:
            state_dict = {}
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu",weights_only=True)
            for i in ckpt['model'].keys():
                if i.startswith('video_Qformer'):
                    state_dict.update({i:ckpt['model'][i]})
            msg = model.load_state_dict(state_dict, strict=False)
        return model
