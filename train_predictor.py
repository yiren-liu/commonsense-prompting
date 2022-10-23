import os
import logging

import numpy as np

import torch


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (
    AutoTokenizer,
)

from utils.dataloader import ESDDatasetBartCOMET2020
from models.strategy_predictor.LSTM import LSTM_predictor
from config_predictor import Args
from utils.predictor_utils import train, evaluate, generate, set_seed

# Configs
# logger
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    args = Args()

    # Setup CUDA, GPU & distributed training
    if not args.no_cuda:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        device = torch.device("cpu")
        args.device = device
        args.n_gpu = 0

    # Set seed
    set_seed(args)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, cache_dir=args.model_cache_dir
    )
    args.tokenizer = tokenizer

    model = LSTM_predictor(args, len(tokenizer))
    
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
    # if args.do_train and False:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)

        # Load dataset
        with open(args.data_path+"/" + args.train_comet_file, "r", encoding="utf-8") as f:
            comet_trn = f.read().split("\n")
        with open(args.data_path+"/" + args.situation_train_comet_file, "r", encoding="utf-8") as f:
            st_comet_trn = f.read().split("\n")
        with open(args.data_path+"/" + args.train_file_name, "r", encoding="utf-8") as f:
            df_trn = f.read().split("\n")
        with open(args.data_path+"/" + args.situation_train_file_name, "r", encoding="utf-8") as f:
            st_trn = f.read().split("\n")

        with open(args.data_path+"/" + args.eval_comet_file, "r", encoding="utf-8") as f:
            comet_val = f.read().split("\n")
        with open(args.data_path+"/" + args.situation_eval_comet_file, "r", encoding="utf-8") as f:
            st_comet_val = f.read().split("\n")
        with open(args.data_path+"/" + args.eval_file_name, "r", encoding="utf-8") as f:
            df_val = f.read().split("\n")
        with open(args.data_path+"/" + args.situation_eval_file_name, "r", encoding="utf-8") as f:
            st_val = f.read().split("\n")

        with open(args.data_path+"/" + args.test_comet_file, "r", encoding="utf-8") as f:
            comet_test = f.read().split("\n")
        with open(args.data_path+"/" + args.situation_test_comet_file, "r", encoding="utf-8") as f:
            st_comet_test = f.read().split("\n")
        with open(args.data_path+"/" + args.test_file_name, "r", encoding="utf-8") as f:
            df_test = f.read().split("\n")
        with open(args.data_path+"/" + args.situation_test_file_name, "r", encoding="utf-8") as f:
            st_test = f.read().split("\n")

        args.train_dataset = ESDDatasetBartCOMET2020(tokenizer, args, df_trn, comet_trn,
                                        st_comet_trn, st_trn, strategy=args.strategy, evaluate=False, test=False, add_situ=args.context)
        args.eval_dataset = ESDDatasetBartCOMET2020(tokenizer, args, df_val, comet_val,
                                       st_comet_val, st_val, evaluate=True, strategy=args.strategy, test=False, add_situ=args.context)
        args.test_dataset = ESDDatasetBartCOMET2020(tokenizer, args, df_test, comet_test,
                                       st_comet_test, st_test, evaluate=True, strategy=args.strategy, test=True, add_situ=args.context)

        # # Training
        global_step, tr_loss = train(
            args, args.train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

        # evaluation
        # load the model from the checkpoint
        model = torch.load(args.output_dir + "/pytorch_model.bin")
        model.to(args.device)
        test_results = evaluate(args, model, tokenizer,
                                args.test_dataset, "of test set")
        