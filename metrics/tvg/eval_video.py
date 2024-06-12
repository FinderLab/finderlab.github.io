import json
import argparse
import re


def format_intent(gcap):
    try:
        start_index = gcap.index('.') + 1
        first_part = gcap[:start_index]
        sub_gcap = gcap[start_index:]
        second_index = sub_gcap.index('. ') + 1
        second_part = sub_gcap[:second_index]
        pattern = r"video (.+?) matches"
        match = re.search(pattern,second_part)
        if match:
            result = match.group(1)
            result = result.strip()
            return first_part,result
        else:
            return first_part,second_part
    except:
        print(gcap)
    
def find_number(text):
    pattern = r"\d+"
    matches = re.search(pattern, text)
    if matches:
        result = matches.group()
        result = int(result)
        return result
    else:
        # print("No number found")
        return 0

def cal_acc_retrieval(pre_path, gt_path):
    pre_data = read(pre_path)
    gt_data = read(gt_path)
    acc = 0
    no_number = 0
    rank10= 0
    for i,item in enumerate(gt_data):
        for k, v in pre_data.items():
            if v.get('query') == item.get('Q'):
                index = find_number(v.get('video_id'))-1
                if index == -1 or index>9:
                    no_number += 1
                    continue
                else:
                    top10 = v.get('vid')
                    pred_video = top10[index]
                    if pred_video == item.get('video_path') and 'video retrieval' in v.get('intent'):
                        acc+=1
    print(f"找不到index的数量有{no_number}")
    return acc*100/len(gt_data)

def cal_acc_video_name(pred_path, gt_data):
    pred_data = read(pred_path)
    gt_data = read(gt_data)
    acc = 0
    is_in_list = 0
    for i, item in enumerate(gt_data):
        for id, sample in pred_data.items():
            if sample.get('query') == item.get('Q'):
                pred_video = sample.get('video_id')
                gt_video = item.get('video_path')
                if pred_video == gt_video and 'video retrieval' in sample.get('intent'):
                    acc+=1
                else:
                    print(sample.get('intent'))
                if pred_video in sample.get('vid'):
                    is_in_list +=1
            else:
                print('noe')
    return acc*100/len(gt_data)

def cal_acc_intent(pred_path ,gt_data):
    pred_data = read(pred_path)
    acc = 0
    for k,v in pred_data.items():
        intent = v.get('intent')
        if 'video retrieval' in intent:
            acc +=1
    return acc*100/len(pred_data)


def format_video(datas):
    fmt_datas = {}
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        intent,video_id = format_intent(gcap=gcap)
        
        fmt_datas[i] = {"video_id": video_id,"intent":intent, "query": query, "vid": vid}
    # print(f'parse failed number: {cnt}')
    return fmt_datas

def read(path):
    with open(path,'r') as file:
        data = json.load(file)
    return data


def wirte(path,data):
    with open(path,'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    pre_path = './output/test_for_final_ivcr_video_retrieval/IVCR_train_epoch10_2w_accgrad16_vfrm12_changeloss_001--2024_05_28_11_01/xpool_blip2_cp7_final_recall10/fmt_IVCR_test_f96_result.json'
    # gt_path = './data_processing/IVCR-200k/test_data/test_video_dup_data_add_top10_1283_no_zero.json'
    # gt_path = './data_processing/test_data/test_video_no_zero.json'
    gt_path = './data_processing/IVCR-200k/test_data/test_video_dup_data_1283.json'
    acc = cal_acc_retrieval(pre_path=pre_path, gt_path=gt_path)
    print(f"pre_path is {pre_path}")
    print(f"gt_path is {gt_path}")
    print(acc)

    