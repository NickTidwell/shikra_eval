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

from mllm.dataset.process_function import PlainBoxFormatter
from mllm.dataset.builder import prepare_interactive
from mllm.models.builder.build_shikra import load_pretrained_shikra
from mllm.dataset.utils.transform import expand2square, box_xyxy_expand2square

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

    args = parser.parse_args()
    print(args)
    return args

def load_shikra_model():
    #########################################
    # mllm model init
    #########################################
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
    # return preprocessor, tokenizer, model
    return {
        "preprocessor": preprocessor,
        "tokenizer": tokenizer,
        "model": model,
    }
#Converted Function from Web Demo:
def process_request( model, image_path, user_input):
    do_sample = False
    max_length = 512
    top_p = 1.0
    temperature =  1.0
    pil_image = Image.open(BytesIO(base64.b64decode(image_path))).convert("RGB")
    ds = prepare_interactive(model.preprocessor)
    image = expand2square(pil_image)
    boxes_value = [box_xyxy_expand2square(box, w=pil_image.width, h=pil_image.height) for box in boxes_value]
    ds.set_image(image)
    ds.append_message(role=ds.roles[0], message=user_input, boxes=[], boxes_seq=[])
    model_inputs = ds.to_model_input()
    model_inputs['images'] = model_inputs['images'].to(torch.float16)
    print(f"model_inputs: {model_inputs}")
    gen_kwargs = dict(
        use_cache=True,
        do_sample=do_sample,
        pad_token_id=model.tokenizer.pad_token_id,
        bos_token_id=model.tokenizer.bos_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        max_new_tokens=max_length,
    )
    print(gen_kwargs)
    input_ids = model_inputs['input_ids']
    with torch.inference_mode():
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            output_ids = model.model.generate(**model_inputs, **gen_kwargs)
    input_token_len = input_ids.shape[-1]
    response = model.tokenizer.batch_decode(output_ids[:, input_token_len:])[0]
    print(f"response: {response}")
    return response

def get_obj_complexity():
    dataset = json.load(open("./data/lvis_v1_val.json", "r"))
    directory_path = "/datasets/MSCOCO17/val2017"
    model = load_shikra_model()
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".png"):
            full_path = os.path.join(directory_path, filename)
            input_query = "In the image, I need the bounding box coordinates of every object."
            process_request(model, full_path, input_query)


def get_noval_obj():
     pass

def test_one_pass():
    input_img_path = "./000000111179.jpg"
    input_query = "Given the following image. Output the bounding box coordinates of each object in the image."
    # input_query = "In the image, I need the bounding box coordinates of every object."
    model = load_shikra_model()
    response = process_request(model, input_img_path, input_query)
    print(response)

args = load_args()
if __name__ == "__main__":
        if args.mode == 1:
            get_obj_complexity()
        if args.mode == 2:
            get_noval_obj()
        if args.mode == 3:
             test_one_pass()
             
             