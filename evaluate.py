import copy
import os
import torch
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList
from math import ceil
from PIL import Image
import numpy as np
import torch.backends.cudnn as cudnn
from ivcr.common.logger import setup_logger
from ivcr.common.config import Config
from ivcr.common.dist_utils import get_rank
from ivcr.common.registry import registry
from ivcr.conversation.conversation_video_batch import Chat, Conversation, default_conversation, SeparatorStyle, \
    conv_llava_llama_2
import decord

decord.bridge.set_bridge('torch')
import logging
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms
import pdb
import json
from pathlib import Path
import time
import datetime
from tqdm import tqdm
import random

random.seed(42)
from utils.format_tvg import format_tvg_output


def read_txt(path):
    with open(path, "r") as fin:
        data = fin.readline().strip()
    return data


def load_data(args, anno_path, split=None):
    with open(anno_path, 'r') as f:
        # data = json.load(f)["annotations"]
        data = json.load(f)

    if args.debug:
        data = data[:10]
    return data


def save_result(args, output_dir, results, split_name='test', format=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result.json'
    if args.debug:
        file_name = 'debug_' + file_name
    if format:
        file_name = 'fmt_' + file_name
    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(results, f,indent=4)
    return


def format_intent(gcap):
    start_index = gcap.index('.') + 1
    first_part = gcap[:start_index]
    sub_gcap = gcap[start_index:]
    second_index = sub_gcap.index('.') + 1
    second_part = sub_gcap[:second_index]
    return first_part,second_part

def format_video(datas):
    fmt_datas = {}
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        intent,video_id = format_intent(gcap=gcap)

        fmt_datas[i] = {"video_id": video_id,"intent":intent, "query": query, "vid": vid}
    return fmt_datas

def format_tvg(datas):
    fmt_datas = {}
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        qid = int(jterm["id"])
        timestamps = format_tvg_output(gcap)
        intent,second_part = format_intent(gcap=gcap)
        if len(timestamps) == 0:
            cnt += 1
            print(vid, query + "\n", gcap, "\n")
        fmt_datas[qid] = {"timestamp": timestamps,"intent":intent, "query": query, "vid": vid}
    print(f'parse failed number: {cnt}')
    return fmt_datas


def generate(chat, gr_videos, user_messages, num_beams, temperature, top_p, n_frms,task, chat_states=None, img_lists=None):
    N = len(user_messages)
    if chat_states is None:
        chat_states = []
        for i in range(N):
            if args.model_type == 'vicuna':
                chat_state = default_conversation.copy()
            else:
                chat_state = conv_llava_llama_2.copy()
            chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            chat_states.append(chat_state)
    if img_lists is None:
        if task == 'format_video':
            img_lists = [[] for i in range(N)]
            llm_message = chat.upload_top10_video(gr_videos, chat_states, img_lists, n_frms=12)
        else:
            img_lists = [[] for i in range(N)]
            llm_message = chat.upload_video_without_audio(gr_videos, chat_states, img_lists, n_frms=n_frms)

    for user_message, chat_state in zip(user_messages, chat_states):
        chat.ask(user_message, chat_state)

    responses = chat.answer(convs=chat_states,
                            img_lists=img_lists,
                            num_beams=num_beams,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=512,
                            max_length=3000)[0]
    return responses, chat_states, img_lists


def main(args):
    num_beams = 1
    temperature = args.temperature
    top_p = args.top_p
    n_frms = args.num_frames
    eval_start_time = time.time()
    prompt = read_txt(args.prompt_file)

    # load model
    device = torch.device(f"cuda:{args.gpu_id}")
    args.options = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = args.ivcr_model_path
    if args.no_lora:
        model_config.lora = False

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()
    message = '\n' + '\n'.join([f'{k:<25}: {v}' for k, v in vars(args).items()])
    logging.info(message)

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.ivcr_instruct.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')

    # load data
    video_path = args.video_path
    anno_path = args.anno_path
    anno_data = load_data(args, anno_path, split=args.split)
    vids = []
    vnames = []
    captions = []
    qids = []
    if args.sample_num > 0:
        # sample part data to evaluate
        anno_data = random.sample(anno_data, args.sample_num)
    for jterm in anno_data:
        if args.task == "format_video":
        #视频检索
            vname = jterm['video_top10_list']
            vid_path = [os.path.join(video_path, name) for name in vname]
        else:
            #视频片段检索
            vname = jterm['video_path']
            vid_path = os.path.join(video_path, vname)
            qids.append(jterm["id"])
        vids.append(vid_path)
        vnames.append(vname)
        captions.append(jterm['Q'])
        
    results = []
    bz = args.batch_size
    # evaluate using batch
    epoch = ceil(len(vnames) / bz)
    for i in tqdm(range(epoch)):
        sid = i * bz
        eid = min((i + 1) * bz, len(vnames))
        prompts = []
        # load video
        paths = vids[sid:eid]
        # image_ids = qids[sid:eid]
        for pi in range(len(paths)):
            final_prompt = copy.deepcopy(prompt)
            if args.task in ["tvg", "vhd","format_video"]:
                idx = sid + pi
                prompts.append(final_prompt.format(captions[idx].strip('.')))
            else:
                prompts.append(final_prompt)
        outputs, chat_states, img_lists = generate(chat, paths, prompts, num_beams, temperature, top_p, n_frms, args.task)
        for j, (output, chat_state) in enumerate(zip(outputs, chat_states)):
            if args.task == "tvg":
                results.append({
                    "vname": vnames[sid + j],
                    "generated_cap": output,
                    "query": captions[sid + j],
                    "id": qids[sid + j],
                    "prompt": chat_state.get_prompt()
                })
            else:
                results.append({
                    "vname": vnames[sid + j],
                    "query": captions[sid + j],
                    "generated_cap": output,
                    "prompt": chat_state.get_prompt()
                })

            if i < 5:
                print(chat_state.get_prompt())
                print(results[-1]["generated_cap"])
                print('*' * 50)

    # save results
    save_result(args, args.output_dir, results, args.split)

    # format results to calculate metrics
    if args.task == "tvg":
        fmt_results = format_tvg(results)
    elif args.task == "format_video":
        fmt_results = format_video(results)
    else:
        print(f"Not support formatting samples for task {args.task}")
    # save format results
    save_result(args, args.output_dir, fmt_results, args.split, format=True)

    total_time = time.time() - eval_start_time
    # convert seconds to date
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))

    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
        f.write(json.dumps(cfg.to_dict(), indent=4) + "\n")
        f.write(message + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='eval_configs/ivcr.yaml')
    parser.add_argument('--anno_path', type=str, default='./data_processing/IVCR-200k/test_data/xpool-clip/test_tvg.json')
    parser.add_argument('--video_path', type=str, default='/data/hanning/data/ivcr_compress')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--task',default='format_video')  # dvc format_video for dense video captioning; tvg for temporal video grounding; vhd for video highlight detection
    parser.add_argument('--dataset', default='IVCR')
    parser.add_argument('--output_dir', default='./output/test_for_final_ivcr_tvg/IVCR_train_epoch10_2w_accgrad16_vfrm12_changeloss_001--2024_05_28_11_01/xpool_clip_cp7_final_top1')
    parser.add_argument('--split', default='test')
    parser.add_argument('--num_frames', type=int, default=96)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--debug', action='store_true', help='the debug mode will only use 10 data samples')
    parser.add_argument('--prompt_file', default='./ivcr/prompts/video_description.txt')
    parser.add_argument('--ivcr_model_path',
                        default="./ckpt/ivcr/IVCR_train_epoch10_2w_accgrad16_vfrm12_changeloss_001/2024_05_28_11_01/checkpoint_7.pth")
    parser.add_argument('--sample_num', type=int, default=-1, help='fast inference by sampling N instances to evaluate')
    parser.add_argument('--example_output', action='store_true', help='output the example results')
    parser.add_argument('--no_lora', action='store_true')
    args = parser.parse_args()
    main(args)

