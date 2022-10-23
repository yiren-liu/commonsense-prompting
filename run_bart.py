import os
import logging

import numpy as np

import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange


from utils.dataloader import ESDDatasetBartCOMET2020, read_data_files
from models.BART import BartATOMIC2020, getBartTokenizerATOMIC2020
from config_bart import Args
from utils.training import train, evaluate, generate, set_seed

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
    tokenizer = getBartTokenizerATOMIC2020(args)
    args.tokenizer = tokenizer

    # model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    model = BartATOMIC2020.from_pretrained(
        args.model_name_or_path, cache_dir=args.model_cache_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
    # if args.do_train and False:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)

        # Load dataset
        # with open(args.data_path+"/" + args.train_comet_file, "r", encoding="utf-8") as f:
        #     comet_trn = f.read().split("\n")
        # with open(args.data_path+"/" + args.situation_train_comet_file, "r", encoding="utf-8") as f:
        #     st_comet_trn = f.read().split("\n")
        # with open(args.data_path+"/" + args.train_file_name, "r", encoding="utf-8") as f:
        #     df_trn = f.read().split("\n")
        # with open(args.data_path+"/" + args.situation_train_file_name, "r", encoding="utf-8") as f:
        #     st_trn = f.read().split("\n")
        df_trn, st_trn, comet_trn, st_comet_trn, comet_by_step_trn = read_data_files(args, split="train")
        df_val, st_val, comet_val, st_comet_val, comet_by_step_eval = read_data_files(args, split="eval")
        df_test, st_test, comet_test, st_comet_test, comet_by_step_test = read_data_files(args, split="test")


        args.train_dataset = ESDDatasetBartCOMET2020(tokenizer, args, df_trn, comet_trn,
                                        st_comet_trn, st_trn, comet_by_step_trn,
                                        strategy=args.strategy, evaluate=False, test=False, add_situ=args.context)
        args.eval_dataset = ESDDatasetBartCOMET2020(tokenizer, args, df_val, comet_val,
                                       st_comet_val, st_val, comet_by_step_eval,
                                       evaluate=True, strategy=args.strategy, test=False, add_situ=args.context)
        args.test_dataset = ESDDatasetBartCOMET2020(tokenizer, args, df_test, comet_test,
                                       st_comet_test, st_test, comet_by_step_test,
                                       evaluate=True, strategy=args.strategy, test=True, add_situ=args.context)

        # # Training
        global_step, tr_loss, tr_lm_loss, tr_strategy_loss, tr_ppl = train(
            args, args.train_dataset, model, tokenizer)
        # logger.info(" global_step = %s, average loss = %s",
        #             global_step, tr_loss)
        logger.info(
            " global_step = %s, average lm loss = %s, average strategy loss = %s, average ppl = %s",
            global_step, tr_lm_loss, tr_strategy_loss, tr_ppl
        )

        # evaluation
        model = BartATOMIC2020.from_pretrained(
            args.output_dir, from_tf=False)
        model.to(args.device)
        test_results = evaluate(args, model, tokenizer,
                                args.test_dataset, "of test set")
        
        # raise NotImplementedError # figure out the perplexity issue


    df_test, st_test, comet_test, st_comet_test, comet_by_step = read_data_files(args, split="test")
    args.test_dataset = ESDDatasetBartCOMET2020(tokenizer, args, df_test, comet_test,
                                    st_comet_test, st_test, comet_by_step=comet_by_step, 
                                    evaluate=True, strategy=args.strategy, test=True, add_situ=args.context)

    model = BartATOMIC2020.from_pretrained(args.load_dir,
        from_tf=False)
    model.resize_token_embeddings(len(tokenizer))

    generate(args, model)

    # raise NotImplementedError
