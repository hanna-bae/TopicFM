import torch 
import torch.nn as nn
import argparse
import onnx 
from src.models import TopicFM
from src.config.default import get_cfg_defaults
from src.lightning_trainer.trainer import PL_Trainer 
import pytorch_lightning as pl
import argparse
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler
from yacs.config import CfgNode as CN 


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='ARCH', default='TopicFM')
    parser.add_argument('--ckpt', type=str, default='./pretrained/topicfm_fast.ckpt')
    args = parser.parse_args()
    return args 


if __name__ == '__main__':
    args = parse()
    def convert_keys_to_snake_case(cfg_node):
        new_cfg_node = {}
        for key, value in cfg_node.items():
            if isinstance(value, CN):
                # Recursively convert keys in nested CfgNode
                new_value = convert_keys_to_snake_case(value)
            else:
                new_value = value
            new_key = key.lower()
            new_cfg_node[new_key] = new_value
        return new_cfg_node

    # Example usage:
    config = get_cfg_defaults()
    converted_config = convert_keys_to_snake_case(config)

    model = TopicFM(converted_config['model'])
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    dummy_input = (torch.rand(3, 1, 640, 480), torch.rand(3, 1, 640, 480))

    
    torch.onnx.export(model, dummy_input, args.model + '.onnx', verbose=True)