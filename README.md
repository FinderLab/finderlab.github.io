<p align="center" width="100%">
<a target="_blank"><img src="llama_logo.png" alt="IVCR" style="width: 40%; min-width: 150px; display: block; margin: auto;"></a>
</p>
<h2 align="center"> IVCR-200K: A Large-Scale Multi-turn Dialogue
Benchmark for Interactive Video Corpus Retrieval</h2>

## Model Architecture
<p align="center" width="100%">
<a target="_blank"><img src="tone.png" alt="Video-LLaMA" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Introduction
**IVCR** enables multi-turn, conversational, realistic interactions
between the user and the retrieval system. The main contributions are summarized as follows: i)-To the best of our knowledge, this is the first
70 work to introduce the “interactive” video corpus retrieval task (IVCR) , which effectively aligns
71 users’ multi-turn behavior in real-world scenarios and significantly enhances user experience. 

ii)-We
72 introduce a dataset and accompanying framework. Notably, the IVCR-200K dataset is a high73
quality, bilingual, multi-turn, conversational, and abstract semantic dataset designed to support video
74 and moment retrieval. The InterLLaVA framework leverages multi-modal large language models
75 (MLLMs) to enable multi-turn dialogue experiences between users and the retrieval system.

## Example Outputs
<p float="left">
    <img src="a.png" style="width: 100%; margin: auto;">
</p>

## Usage
#### Enviroment Preparation 

## How to Run
### Tuning
Using One GPU
```
python train.py --cfg-path /data/longshaohua/IVCR_2/train_configs/stage2_finetune_IVCR.yaml
```
Using Multiple GPUs(use four gpus as example)
```
accelerate launch --num_processes=4 train.py --cfg-path /data/longshaohua/TimeChat/train_configs/stage2_finetune_IVCR.yaml
```

### Evaluating
Temporal Video Grounding
```
python evaluate.py --task tvg
```

Video Retrieval
```
python evaluate.py
```