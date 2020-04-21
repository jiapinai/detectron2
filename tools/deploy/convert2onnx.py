#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import os
import numpy as np

import onnx
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import add_export_config, export_caffe2_model

from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
import torch
from torch.onnx import OperatorExportTypes

from export.api import export_onnx_model

def setup_cfg(args):
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = add_export_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a model to Caffe2")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted caffe2 model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))

    cfg = setup_cfg(args)

    # create a torch model
    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    # print(torch_model)

    # get a sample data
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    first_batch = next(iter(data_loader))
    print('first_batch', first_batch[:1] )

    onnx_model = export_onnx_model(cfg, model, first_batch[:1])

    # Preprocessing: get the path to the saved model
    new_model_path = os.path.join('single_relu_new.onnx')
    # Save the ONNX model
    onnx.save(onnx_model, new_model_path)
    print('The model is saved.')

    #data = torch.randn(3, 224, 224)
    # convert
    #saved_path_traced = os.path.dirname(os.path.abspath(__file__)) + '/traced.pt'
    #model_traced = torch.jit.trace(torch_model, [{'image': data}])
    #model_traced.save(saved_path_traced)

    print('ok')
    #print(saved_path_traced)
