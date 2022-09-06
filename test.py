from typing import Dict, List, Tuple
import os
import random
import logging
import glob
import shutil
import re


import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    PreTrainedModel,
    PreTrainedTokenizer,
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallConfig
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange


from utils.dataloader import ESDDataset
from models.blenderbot import getBlenderbotTokenizerATOMIC2020

# Configs
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class Args():
    def __init__(self):
        TAG = 'all_loss'
        # TAG = 'emotion'
        # TAG = 'ablation_strategy'
        # TAG = 'ablation_situation'
        # TAG = 'ablation_post'
        # nowtime = '10251756'
        self.output_dir = os.path.join('blender_strategy', TAG)
        self.generation_dir = os.path.join('generated_data', TAG)
        self.model_type = 'mymodel'
     #    self.model_name_or_path = './blender-small'
        self.model_name_or_path = "facebook/blenderbot_small-90M"
        self.config_name = "facebook/blenderbot_small-90M"
        self.tokenizer_name = "facebook/blenderbot_small-90M"
        self.data_path = "./data/dataset"
        self.train_file_name = "trainWithStrategy_short.tsv"
        self.eval_file_name = "devWithStrategy_short.tsv"
        self.test_file_name = "testWithStrategy_short.tsv"
        self.train_comet_file = "trainComet.txt"
        self.eval_comet_file = "devComet.txt"
        self.test_comet_file = "testComet.txt"
        self.situation_train_comet_file = "trainComet_st.txt"
        self.situation_eval_comet_file = "devComet_st.txt"
        self.situation_test_comet_file = "testComet_st.txt"

        self.model_cache_dir = './cached/models/blender-small'
        self.data_cache_dir = './cached/data'
        self.block_size = 512
        self.do_train = True
        self.do_eval = False
        self.generation = False
        self.generate_and_eval = False
        self.evaluate_during_training = True
        self.per_gpu_train_batch_size = 2
        self.per_gpu_eval_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 2e-5  # RAW 2
        self.weight_decay = 0
        self.adam_epsilon = 1e-8  # RAW 8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 8  # raw 10
        self.max_steps = -1
        self.warmup_steps = 120  # raw 120
        self.logging_steps = 30
        self.save_steps = 30
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        #    self.no_cuda = True
        self.overwrite_output_dir = True
        self.overwrite_cache = False
        self.should_continue = False
        self.seed = 42  # raw 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.strategy = False
        self.turn = False
        self.role = False


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=ESDDataset.collate, drop_last=False
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Take care of distributed/parallel training
    model = model.module if hasattr(model, "module") else model
    model.resize_token_embeddings(len(tokenizer))
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if False and (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        #model = BalancedDataParallel(2,model, dim=0).to(args.device)
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[
                args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        ).to(args.device)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if False and args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split(
                "-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) //
                                             args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info(
                "  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch",
                        steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss, tr_lm_loss, logging_lm_loss, tr_emo_loss, \
        logging_emo_loss, tr_strategy_loss, logging_strategy_loss, tr_intensity_loss, logging_intensity_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    best_ppl = 1e8

    model.zero_grad()
    #train_iterator = range(epochs_trained, int(args.num_train_epochs))
    train_iterator = trange(epochs_trained, int(
        args.num_train_epochs), desc="Epoch", disable=True)
    set_seed(args)  # Added here for reproducibility
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    for epoch in train_iterator:
        # if epoch < 3:
        #     for paras in model.model.encoder.parameters():
        #         paras.requires_grad = True
        #     for paras in model.model.decoder.parameters():
        # #         paras.requires_grad = False
        # if epoch < 6:
        #     if epoch % 2 == 0:
        #         for paras in model.model.encoder.parameters():
        #             paras.requires_grad = True
        #         for paras in model.model.decoder.parameters():
        #             paras.requires_grad = False
        #     else:
        #         for paras in model.model.encoder.parameters():
        #             paras.requires_grad = False
        #         for paras in model.model.decoder.parameters():
        #             paras.requires_grad = True
        # else:
        #     for paras in model.model.encoder.parameters():
        #         paras.requires_grad = True
        #     for paras in model.model.decoder.parameters():
        #         paras.requires_grad = True

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            # print("step:",step)
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            input_ids, position_ids, turn_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_turn_ids, \
                decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask, emotion, comet_ids_st, comet_mask_st = batch
            # print(input_ids)
            # for item in input_ids:
            #     print(len(item))
            #     print(tokenizer.decode(item))
            # print(1 / 0)
            decoder_strategy_ids = decoder_strategy_ids[:, 0]
            decoder_strategy_ids = decoder_strategy_ids.to(args.device)

            # print(comet_ids)
            # print(comet_mask)
            # print(1 / 0)
            # print(input_ids, position_ids, turn_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_turn_ids, decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids)
            # print("=="*100)

            model.train()
            if input_ids.shape[1] > 512:
                continue
            emotion = emotion.to(args.device)
            comet_ids = comet_ids.to(args.device)
            comet_mask = comet_mask.to(args.device)

            comet_ids_st = comet_ids_st.to(args.device)
            comet_mask_st = comet_mask_st.to(args.device)

            batch_size, n_attr, len_attr = comet_ids.shape
            comet_ids = comet_ids.view(-1, len_attr)
            with torch.no_grad():
                comet_embs = model.model.encoder(
                    comet_ids, attention_mask=comet_ids.ne(tokenizer.pad_token_id))[0][:, 0, :]
            comet_embs = comet_embs.view(batch_size, n_attr, -1)

            batch_size, n_attr, len_attr = comet_ids_st.shape
            comet_ids_st = comet_ids_st.view(-1, len_attr)
            with torch.no_grad():
                comet_embs_st = model.model.encoder(
                    comet_ids_st, attention_mask=comet_ids_st.ne(tokenizer.pad_token_id))[0][:, 0, :]
            comet_embs_st = comet_embs_st.view(batch_size, n_attr, -1)

            input_ids = input_ids.to(args.device)
            turn_ids = turn_ids.to(args.device)
            role_ids = role_ids.to(args.device)
            decoder_input_ids = decoder_input_ids.to(args.device)
            decoder_turn_ids = decoder_turn_ids.to(args.device)
            decoder_label_ids = decoder_labels.to(args.device)
            decoder_role_ids = decoder_role_ids.to(args.device)
            #decoder_cls_labels = decoder_cls_labels.to(args.device)
            # model.train()
            # we did't use role label and turn number in modeling as they did't carry significant improvement. Codes still remain.
            if not args.role:
                role_ids = None
            if not args.turn:
                turn_ids = None
            if False:
                outputs = model(input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id), decoder_input_ids=decoder_input_ids, decoder_turn_ids=decoder_turn_ids, decoder_role_ids=decoder_role_ids,
                                turn_ids=turn_ids, role_ids=role_ids, labels=decoder_label_ids, decoder_strategy_ids=decoder_strategy_ids,  comet_embs=comet_embs,  comet_mask=comet_mask, emotion=emotion)
                # model outputs are always tuple in transformers (see doc)
                ppl = loss = outputs[0]
            else:
                print("input_ids:",input_ids)
                print("role_ids:",role_ids)
                print("turn_ids:",turn_ids)  

                print("decoder_input_ids:",decoder_input_ids)
                print("decoder_role_ids:",decoder_role_ids)
                print("decoder_turn_ids:",decoder_turn_ids)
                raise NotImplementedError # need to modify model input code


                outputs = model(input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id), decoder_input_ids=decoder_input_ids, decoder_turn_ids=decoder_turn_ids, decoder_role_ids=decoder_role_ids, turn_ids=turn_ids,
                                role_ids=role_ids, labels=decoder_label_ids, decoder_strategy_ids=decoder_strategy_ids, comet_embs=comet_embs, comet_mask=comet_mask, comet_embs_st=comet_embs_st, comet_mask_st=comet_mask_st, emotion=emotion)
                # print(outputs.lm_logits, outputs.emo_logits)
                # print(outputs.loss, outputs.emo_loss, outputs.lm_loss)
                # print(1 / 0)
                loss = outputs.loss
                lm_loss = ppl = outputs.lm_loss
                emo_loss = outputs.emo_loss
                intensity_loss = outputs.intensity_loss
                strategy_loss = outputs.strategy_loss

            # if not args.no_cuda and args.n_gpu >= 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
            #     ppl = ppl.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                backward_loss = outputs.loss
                # backward_loss = outputs.lm_loss
                # if epoch == 0 or epoch == 1:
                #     backward_loss = outputs.strategy_loss
                # else:
                #     backward_loss = outputs.loss
                backward_loss.backward()

            tr_loss += loss.item()
            tr_lm_loss += lm_loss.item()
            tr_emo_loss += emo_loss.item()
            tr_strategy_loss += strategy_loss.item()
            tr_intensity_loss += intensity_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 and global_step > t_total*0.0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(
                            args, model, tokenizer, args.eval_dataset, "{}-{}".format("checkpoint", global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar(
                        "lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("lr: %f, step: %d, loss: %f, lm_loss: %f, emo_loss: %f, strategy_loss: %f, intensity_loss: %f", scheduler.get_last_lr()[0],
                                global_step, (tr_loss - logging_loss) /
                                args.logging_steps, (tr_lm_loss -
                                                     logging_lm_loss) / args.logging_steps,
                                (tr_emo_loss - logging_emo_loss) / args.logging_steps, (tr_strategy_loss -
                                                                                        logging_strategy_loss) / args.logging_steps,
                                (tr_intensity_loss - logging_intensity_loss) / args.logging_steps)

                    logging_loss = tr_loss
                    logging_lm_loss = tr_lm_loss
                    logging_emo_loss = tr_emo_loss
                    logging_strategy_loss = tr_strategy_loss
                    logging_intensity_loss = tr_intensity_loss
                    if results['eval_perplexity'] < best_ppl:
                        best_ppl = results['eval_perplexity']

                        checkpoint_prefix = "checkpoint"

                        output_dir = args.output_dir
                        os.makedirs(output_dir, exist_ok=True)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(
                            output_dir, "training_args.bin"))
                        logger.info(
                            "Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

                        torch.save(optimizer.state_dict(), os.path.join(
                            output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(
                            output_dir, "scheduler.pt"))
                        logger.info(
                            "Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    print("Train finished~")
    return global_step, tr_loss / global_step

# Evaluation of some model
def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix="") -> Dict:
    import numpy as np
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    #eval_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=True)
    os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=ESDDataset.collate, drop_last = False
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    #multi-gpu evaluate
    #if args.n_gpu > 1:
    #    model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    strategy_probs = []
    cls_labels_list = []
    num_samples = []
    emo_hits = []
    # strategy_hits_topk = [[] for _ in range(7)]
    strategy_hits = []

    for batch in tqdm(eval_dataloader, desc="Evaluating",disable=True):
        model.train()
        input_ids, position_ids, turn_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_turn_ids, decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask, emotion, comet_ids_st, comet_mask_st = batch
        if input_ids.shape[1] > 512: continue

        decoder_strategy_ids = decoder_strategy_ids[:, 0]
        decoder_strategy_ids = decoder_strategy_ids.to(args.device)

        emotion = emotion.to(args.device)
        comet_ids = comet_ids.to(args.device)
        comet_mask = comet_mask.to(args.device)
        comet_ids_st = comet_ids_st.to(args.device)
        comet_mask_st = comet_mask_st.to(args.device)

        batch_size, n_attr, len_attr = comet_ids.shape
        comet_ids = comet_ids.view(-1, len_attr)

        with torch.no_grad():
            comet_embs = model.model.encoder(comet_ids, attention_mask = comet_ids.ne(tokenizer.pad_token_id))[0][:,0,:]

        comet_embs = comet_embs.view(batch_size, n_attr, -1)
        batch_size, n_attr, len_attr = comet_ids_st.shape
        comet_ids_st = comet_ids_st.view(-1, len_attr)

        with torch.no_grad():
            comet_embs_st = model.model.encoder(comet_ids_st, attention_mask=comet_ids_st.ne(tokenizer.pad_token_id))[0][:,0, :]
        comet_embs_st = comet_embs_st.view(batch_size, n_attr, -1)

        input_ids = input_ids.to(args.device)
        turn_ids = turn_ids.to(args.device)
        role_ids = role_ids.to(args.device)
        decoder_input_ids = decoder_input_ids.to(args.device)
        decoder_turn_ids = decoder_turn_ids.to(args.device)
        decoder_label_ids = decoder_labels.to(args.device)
        decoder_role_ids = decoder_role_ids.to(args.device)
        decoder_cls_labels = decoder_cls_labels.to(args.device)

        with torch.no_grad():
            if not args.role:
                role_ids = None
            if not args.turn:
                turn_ids = None

            if False:
                outputs = model(input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id),
                                decoder_input_ids=decoder_input_ids,
                                decoder_turn_ids=decoder_turn_ids, decoder_role_ids=decoder_role_ids, turn_ids=turn_ids,
                                role_ids=role_ids, labels=decoder_label_ids, comet_embs=comet_embs,
                                comet_mask=comet_mask,
                                emotion=emotion)
                ppl = loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            else:
                outputs = model(input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id),
                                decoder_input_ids=decoder_input_ids, decoder_turn_ids=decoder_turn_ids,
                                decoder_role_ids=decoder_role_ids, turn_ids=turn_ids, role_ids=role_ids,
                                labels=decoder_label_ids, decoder_strategy_ids=decoder_strategy_ids,
                                comet_embs=comet_embs, comet_mask=comet_mask, comet_embs_st=comet_embs_st,
                                comet_mask_st=comet_mask_st, emotion=emotion)
                loss = outputs.loss

                ppl = outputs.lm_loss
                emo_logits = outputs.emo_logits
                strategy_logits = outputs.strategy_logits

            # print(strategy_logits.argmax(dim=-1))
            for idx, emo_logit in enumerate(emo_logits):
                if emo_logit.argmax() == emotion[idx]:
                    emo_hits.append(1)
                else:
                    emo_hits.append(0)

            # print(decoder_input_ids)
            # strategy_ids = decoder_input_ids[:, 0] - 54944

            for idx, strategy_logit in enumerate(strategy_logits):
                if strategy_logit.argmax() == decoder_strategy_ids[idx]:
                    strategy_hits.append(1)
                else:
                    strategy_hits.append(0)


            # if args.strategy:
            #     cls_labels_list.extend(decoder_cls_labels.cpu().numpy().tolist())
            #     strategy_probs.append(torch.nn.functional.softmax(outputs.lm_logits[0, 0, 54945:54945+8], dim=-1).cpu().numpy().tolist())

            lm_loss = outputs.lm_loss
            num_samples.append((decoder_label_ids.cpu().numpy() != -100).astype(np.int).sum())
            eval_loss += lm_loss.sum().item() * (decoder_label_ids.cpu().numpy() != -100).astype(np.int).sum()

        nb_eval_steps += 1

    eval_loss = eval_loss/ sum(num_samples)
    perplexity = torch.exp(torch.tensor(eval_loss)).item()
    # np_strategy = np.array(strategy_probs)
    # np_cls_labels = np.array(cls_labels_list)
    # result = {"eval_perplexity": perplexity, "eval_emotion_predict_accuracy": sum(emo_hits)/len(emo_hits), "eval_strategy_predict_accuracy": sum(strategy_hits)/len(strategy_hits), "eval_number_of_evaluated_examples": len(emo_hits)}
    result = {"eval_perplexity": perplexity, "eval_emotion_predict_accuracy": sum(emo_hits) / len(emo_hits),"eval_strategy_predict_accuracy": sum(strategy_hits) / len(strategy_hits),
              "eval_number_of_evaluated_examples": len(emo_hits)}
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")


    with open(output_eval_file, "a+") as writer:
        # print("***** Eval results {} *****".format(prefix))
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("***** Eval results {} *****".format(prefix) + "\n")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            # print("  %s = %s" % (key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

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
    tokenizer = getBlenderbotTokenizerATOMIC2020(args)

    model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)

        # Load dataset
        with open(args.data_path+"/" + args.train_comet_file, "r", encoding="utf-8") as f:
            comet_trn = f.read().split("\n")
        with open(args.data_path+"/" + args.situation_train_comet_file, "r", encoding="utf-8") as f:
            st_comet_trn = f.read().split("\n")
        with open(args.data_path+"/" + args.train_file_name, "r", encoding="utf-8") as f:
            df_trn = f.read().split("\n")
        args.train_dataset = ESDDataset(tokenizer, args, df_trn, comet_trn,
                                        st_comet_trn, strategy=args.strategy, evaluate=False, test=False)
        
        global_step, tr_loss = train(args, args.train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.output_dir, from_tf=False)
        model.to(args.device)
        test_results = evaluate(args, model, tokenizer, args.test_dataset, "of test set")

    raise NotImplementedError
