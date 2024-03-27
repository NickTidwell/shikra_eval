import argparse
import os
import base64
import logging
from io import BytesIO
import os

import json
import torch
import transformers
from PIL import Image
from mmengine import Config
from transformers import BitsAndBytesConfig
import numpy as np
from mllm.dataset.process_function import PlainBoxFormatter
from mllm.dataset.builder import prepare_interactive
from mllm.models.builder.build_shikra import load_pretrained_shikra
from mllm.dataset.utils.transform import expand2square, box_xyxy_expand2square
import re

log_level = logging.DEBUG
transformers.logging.set_verbosity(log_level)
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()


def load_args():
    parser = argparse.ArgumentParser("Shikra HW")
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--load_in_8bit', action='store_true')
    #TODO change these names
    parser.add_argument('--mode',
                            type=int, default=1,
                            help='Select example to run.')
    parser.add_argument('--small_set', action='store_true')

    args = parser.parse_args()
    print(args)
    return args

#########################################
# mllm model init
#########################################


args = load_args()

# def load_shikra_model(args):
model_name_or_path = args.model_path

model_args = Config(dict(
    type='shikra',
    version='v1',

    # checkpoint config
    cache_dir=None,
    model_name_or_path=model_name_or_path,
    vision_tower=r'openai/clip-vit-large-patch14',
    pretrain_mm_mlp_adapter=None,

    # model config
    mm_vision_select_layer=-2,
    model_max_length=2048,

    # finetune config
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,
    freeze_mm_mlp_adapter=False,

    # data process config
    is_multimodal=True,
    sep_image_conv_front=False,
    image_token_len=256,
    mm_use_im_start_end=True,

    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    process_func_args=dict(
        conv=dict(type='ShikraConvProcess'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='ShikraTextProcess'),
        image=dict(type='ShikraImageProcessor'),
    ),

    conv_args=dict(
        conv_template='vicuna_v1.1',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=None),
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
))
training_args = Config(dict(
    bf16=False,
    fp16=True,
    device='cuda',
    fsdp=None,
))

if args.load_in_8bit:
    quantization_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
        )
    )
else:
    quantization_kwargs = dict()

model, preprocessor = load_pretrained_shikra(model_args, training_args, **quantization_kwargs)
if not getattr(model, 'is_quantized', False):
    model.to(dtype=torch.float16, device=torch.device('cuda'))
if not getattr(model.model.vision_tower[0], 'is_quantized', False):
    model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device('cuda'))
print(
    f"LLM device: {model.device}, is_quantized: {getattr(model, 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
print(
    f"vision device: {model.model.vision_tower[0].device}, is_quantized: {getattr(model.model.vision_tower[0], 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")

preprocessor['target'] = {'boxes': PlainBoxFormatter()}
tokenizer = preprocessor['text']

#Converted Function from Web Demo:
def process_request(image_path, user_input):
    boxes_value=[]
    pil_image = Image.open(image_path).convert("RGB")
    ds = prepare_interactive(model_args, preprocessor)
    image = expand2square(pil_image)
    boxes_value = [box_xyxy_expand2square(box, w=pil_image.width, h=pil_image.height) for box in boxes_value]
    ds.set_image(image)
    ds.append_message(role=ds.roles[0], message=user_input, boxes=[], boxes_seq=[])
    model_inputs = ds.to_model_input()
    model_inputs['images'] = model_inputs['images'].to(torch.float16)
    print(f"model_inputs: {model_inputs}")
    gen_kwargs = dict(
        use_cache=True,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
    )
    print(gen_kwargs)
    input_ids = model_inputs['input_ids']
    with torch.inference_mode():
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            output_ids = model.generate(**model_inputs, **gen_kwargs)
    input_token_len = input_ids.shape[-1]
    response = tokenizer.batch_decode(output_ids[:, input_token_len:])[0]
    print(f"response: {response}")
    return response



def group_results(objs_per_image):
    grouped_objs = {
    '1': [],
    '2': [],
    '3': [],
    '4': [],
    '5': [],
    '6-10': [],
    '11-20': [],
    '20+': []
    }
    for num_truth, boxes in objs_per_image.items():
        if num_truth == 1:
            grouped_objs['1'].append(boxes)
        elif num_truth == 2:
            grouped_objs['2'].append(boxes)
        elif num_truth == 3:
            grouped_objs['3'].append(boxes)
        elif num_truth == 4:
            grouped_objs['4'].append(boxes)
        elif num_truth == 5:
            grouped_objs['5'].append(boxes)
        elif 6 <= num_truth <= 10:
            grouped_objs['6-10'].append(boxes)
        elif 11 <= num_truth <= 20:
            grouped_objs['11-20'].append(boxes)
        else:
            grouped_objs['20+'].append(boxes)
    return grouped_objs
def get_obj_complexity():
    objsPerImagePred = dict()
    objsPerImageTruth = dict()


    dataset = json.load(open("./data/lvis_v1_val.json", "r"))
    directory_path = "/datasets/MSCOCO17/val2017"
    # preprocessor, tokenizer, model = load_shikra_model(args)
    itr = 0
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".jpg"):
            if args.small_set:
                if itr == 30:
                    break
            full_path = os.path.join(directory_path, filename)
            input_query = "In the image, I need the bounding box coordinates of every object."
            response = process_request(full_path, input_query)
            print("RESPONSE: ", response)
            target = get_truth_label(dataset, full_path)
            print("TARGET: ", target)
            pred = parse_response(response)
            print("PRED: ", pred)

            objInSample = len(target)

            if objInSample not in objsPerImageTruth:
                objsPerImageTruth[objInSample] = []  
            objsPerImageTruth[objInSample].append(target)
            if objInSample not in objsPerImagePred:
                objsPerImagePred[objInSample] = []
            objsPerImagePred[objInSample].append(pred)
        itr += 1

    gObjPerImgTru = group_results(objsPerImageTruth)
    gObjPerImgPre = group_results(objsPerImagePred)
    print("\nobjsPerImageTruth\n")
    for key, value in objsPerImageTruth.items():
        print(f"{key}: {value}")    
    print("\nObjPerImgTru\n")
    for key, value in gObjPerImgTru.items():
        print(f"{key}: {value}")    
    map_dict = dict()
    for num_truth, _ in gObjPerImgTru.items(): #TODO Gropu obj
        truth_boxes = gObjPerImgTru[num_truth]
        pred_boxes = gObjPerImgPre[num_truth]
        mAP = compute_mAP(truth_boxes, pred_boxes)
        map_dict[num_truth] = mAP
    print("\nOBJECTS COMPLEXITY Output\n")
    for key, value in map_dict.items():
        print(f"{key}: {value}")
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = w1 * h1 + w2 * h2 - intersection_area

    return intersection_area / union_area

def compute_mAP(ground_truth, predictions, thresh=0.5):
    # Assuming ground_truth and predictions are lists of bounding boxes
    # Each bounding box: [x_min, y_min, x_max, y_max]
    if len(ground_truth) == 0:
        print("ERROR: Ground Truth 0")
        return 0.0  # Return 0 if no ground truth boxes
    # Calculate IoU for all pairs
    iou_matrix = np.zeros((len(ground_truth), len(predictions)))
    for i, gt_box in enumerate(ground_truth):
        for j, pred_box in enumerate(predictions):
            iou_matrix[i, j] = calculate_iou(gt_box, pred_box)

    # Compute precision and recall
    sorted_indices = np.argsort(-iou_matrix, axis=1)
    tp, fp = 0, 0
    precision, recall = [], []
    for i in range(len(ground_truth)):
        matched = False
        for j in sorted_indices[i]:
            if iou_matrix[i, j] > thresh:
                matched = True
                break
        if matched:
            tp += 1
        else:
            fp += 1
        precision.append(tp / (tp + fp))
        recall.append(tp / len(ground_truth))

    # Compute mAP
    auc = np.trapz(precision, recall)
    mAP = auc / len(ground_truth)
    return mAP

def parse_response(text):
    out_list = []
    try:
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, text)
        out_list = [tuple(float(x) for x in sublist.split(',')) for box in matches for sublist in box.split(';')]
    except:
        pass

    return out_list

def get_truth_label(dataset, full_path):
    image_id_with_zeros = full_path.split('/')[-1].split('.')[0]
    image_id = int(image_id_with_zeros.lstrip('0'))
    image_annotations = dataset['annotations']
    annotations_for_image = [annotation for annotation in image_annotations if annotation['image_id'] == image_id]
    return annotations_for_image

def get_truth_box(dataset, image_path):
    image_id_with_zeros = image_path.split('/')[-1].split('.')[0]
    image_id = int(image_id_with_zeros.lstrip('0'))
    image_annotations = dataset['annotations']
    bounding_boxes_for_image = [annotation['bbox'] for annotation in image_annotations if annotation['image_id'] == image_id]
    return bounding_boxes_for_image
def get_noval_obj():
    try:
        with open("./data/lvis_v1_val.json", "r") as json_file:
            dataset = json.load(json_file)
    except FileNotFoundError:
        print("Error: LVIS dataset file not found.")
        return

    # Define dataset directory
    image_directory = "/datasets/MSCOCO17/val2017"

    # Calculate category frequencies
    category_frequencies = {}
    for category in dataset['categories']:
        category_id = category['id']
        image_count = category['image_count']
        category_frequencies[category_id] = image_count

    # Calculate quantiles
    frequencies = list(category_frequencies.values())
    quantiles = np.quantile(frequencies, [0.25])  # For example, using the 25th and 75th percentiles

    rare_pred_boxes = []
    common_pred_boxes = []
    rare_truth_boxes = []
    common_truth_boxes = []
    itr = 0
    # Process each image in the directory
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(".png"):
            if args.small_set:
                if itr == 30:
                    break
            full_path = os.path.join(image_directory, filename)
            input_query = "In the image, I need the bounding box coordinates of every object."
            # response = shikra(image_path, "In the image, I need the bounding box coordinates of every object.")

            response = process_request(full_path, input_query)
            target = get_truth_label(dataset, full_path)
            pred = parse_response(response)

            truth_boxes = [truth_label['category_id'] for truth_label in get_truth_box(dataset, full_path)]

            # Sort boxes
            rare_categories = [category_id for category_id in truth_boxes if category_frequencies.get(category_id, 0) <= quantiles[0]]
            common_categories = [category_id for category_id in truth_boxes if category_frequencies.get(category_id, 0) > quantiles[0]]

            if rare_categories:
                rare_pred_boxes.append(pred)
                rare_truth_boxes.append(target)
            if common_categories:
                common_pred_boxes.append(pred)
                common_truth_boxes.append(target)
            itr += 1

    # Calculate mAP for rare and common categories
    mAP_rare = compute_mAP(rare_truth_boxes, rare_pred_boxes)
    mAP_common = compute_mAP(common_truth_boxes, common_pred_boxes)

    scores = {'rare': mAP_rare, 'common': mAP_common}
    print("\nNOVEL OBJECTS\n")
    for key, value in scores.items():
        print(f"{key}: {value}")
    
def test_one_pass():
    input_img_path = "./000000111179.jpg"
    input_query = "Given the following image. Output the bounding box coordinates of each object in the image."
    # input_query = "In the image, I need the bounding box coordinates of every object."
    response = process_request(input_img_path, input_query)
    print(response)

if __name__ == "__main__":
        if args.mode == 1:
            get_obj_complexity()
        if args.mode == 2:
            get_noval_obj()
        if args.mode == 3:
             test_one_pass()
             
             