# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import gc
import csv

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer)

from transformers import AdamW, WarmupLinearSchedule

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig,
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def weight_f1_score(val_y, predict, val_id):
    val_dict = {}
    for i in range(len(val_y)):
        if val_id[i] not in val_dict:
            val_dict[val_id[i]] = {'y': val_y[i], 's': 0,
                                   'e': {'tp': 0, 'fp': 0, 'fn': 0}}  # y 不仅是遇到的第一个实体标签值，应该是所有该实体标签的总统计
        else:
            val_dict[val_id[i]]['y'] |= val_y[i]  # 取或运算
        if predict[i] == 1:
            val_dict[val_id[i]]['s'] = 1
            if val_y[i] == 1:
                val_dict[val_id[i]]['e']['tp'] += 1
            else:
                val_dict[val_id[i]]['e']['fp'] += 1
        else:
            if val_y[i] == 1:
                val_dict[val_id[i]]['e']['fn'] += 1
    tps = 0
    fps = 0
    fns = 0
    for k, v in val_dict.items():
        if v['y'] == 1:
            if v['s'] == 1:
                tps += 1
            else:
                fns += 1
        else:
            if v['s'] == 1:
                fps += 1
    if tps + fps != 0:
        ps = tps / (tps + fps)
    else:
        ps = 0
    if tps + fns != 0:
        rs = tps / (tps + fns)
    else:
        rs = 0
    if ps + rs != 0:
        fs1 = 2 * ps * rs / (ps + rs)
    else:
        fs1 = 0
    tpe = 0
    fpe = 0
    fne = 0
    for k, v in val_dict.items():
        tpe += v['e']['tp']
        fpe += v['e']['fp']
        fne += v['e']['fn']
    if tpe + fpe != 0:
        pe = tpe / (tpe + fpe)
    else:
        pe = 0
    if tpe + fne != 0:
        re = tpe / (tpe + fne)
    else:
        re = 0
    if pe + re != 0:
        fe1 = 2 * pe * re / (pe + re)
    else:
        fe1 = 0

    f1 = 0.4 * fs1 + 0.6 * fe1

    simple_f1 = f1_score(val_y, predict, average='macro')
    simple_acc = accuracy_score(val_y, predict)
    simple_pre = precision_score(val_y, predict, average='macro')
    simple_recall = recall_score(val_y, predict, average='macro')
    # simple_roc_auc = roc_auc_score(val_y, prob)
    result = {}
    result["f1"] = f1
    result["fs1"] = fs1
    result["fe1"] = fe1
    result["simple_f1"] = simple_f1
    result["simple_acc"] = simple_acc
    result["simple_pre"] = simple_pre
    result["simple_recall"] = simple_recall
    # result["simple_roc_auc"] = simple_roc_auc
    return result


def train(args, train_examples, eval_examples, model, tokenizer, fold, tb_writer):
    """ Train the model """

    train_dataset = load_examples(args, train_examples, tokenizer, evaluate=False)
    eval_dataset = load_examples(args, eval_examples, tokenizer, evaluate=True)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(args.warmup_proportion * t_total), t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("------------ Fold: %d -------------", fold)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    # for _ in train_iterator:
    for _ in range(int(args.num_train_epochs)):
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # for step, batch in enumerate(epoch_iterator):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 and not args.tpu:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                logger.info("  global_step = {}, loss = {}".format(global_step, loss.item()))

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, eval_examples, eval_dataset, model, tokenizer,
                                           prefix="fold-{}-global_step-{}".format(fold, global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step + fold * t_total)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step + fold * t_total)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps,
                                         global_step + fold * t_total)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'fold-{}-checkpoint-{}'.format(fold, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.tpu:
                args.xla_model.optimizer_step(optimizer, barrier=True)
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step >= args.max_steps:
                # epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            # train_iterator.close()
            break

    del train_dataset, train_dataloader, train_sampler, eval_dataset
    return global_step, tr_loss / global_step


def evaluate(args, eval_examples, eval_dataset, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, type="dev")
        # eval_dataset = load_examples(args, eval_examples, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        # eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                               'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                # eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        # eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds_label = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds_label = np.squeeze(preds)
        # result = compute_metrics(eval_task, preds_label, out_label_ids)
        data_id = []
        for item in eval_examples:
            data_id.append(item.guid)
        result = weight_f1_score(out_label_ids, preds_label, data_id)
        results.update(result)

        # output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
        #     logger.info("***** Eval results {} *****".format(prefix))
        #     for key in sorted(result.keys()):
        #         logger.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n" % (key, str(result[key])))

        logger.info("***** Eval results {} *****".format(prefix))
        logger.info("%s", str(result))

        del eval_sampler, eval_dataloader, data_id, preds_label, out_label_ids, preds
    return results


def test(args, test_examples, model, tokenizer, prefix=""):
    """test"""
    # test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True, type="test")
    test_dataset = load_examples(args, test_examples, tokenizer, evaluate=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running test {} *****".format(prefix))
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    # for batch in tqdm(test_dataloader, desc="Predicting"):
    for batch in test_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    # if args.output_mode == "classification":
    #     preds_label = np.argmax(preds, axis=1)
    # elif args.output_mode == "regression":
    #     preds_label = np.squeeze(preds)

    output_predict_file = os.path.join(args.output_dir, prefix)
    write_test_results(output_predict_file, preds, test_examples)

    del test_dataset, test_dataloader, test_sampler, out_label_ids
    return preds


def write_test_results(output_predict_file, preds, test_examples):
    """根据模型输出的logits输出预测结果文件"""
    from jrst_data_handle import load_test
    test_data = load_test()
    preds_label = np.argmax(preds, axis=1)
    with open(output_predict_file + ".csv", "w", encoding="utf_8_sig") as writer:
        writer.write("id,negative,key_entity\n")
        logging.info("***** Predict results *****")
        cursor = 0
        num_written_lines = 0
        for item in test_data:
            if item[-1] != "":
                entity = [e.strip() for e in item[-1].split(";") if e != ""]
                neg_entity = []
                for (j, term) in enumerate(entity):
                    # 获取概率值最大的类别对应的下标
                    index = preds_label[cursor + j]
                    if index == 1:
                        neg_entity.append(term)
                if len(neg_entity) != 0:
                    output_line = "{},{},{}\n".format(item[0], "1", ";".join(neg_entity))
                else:
                    output_line = "{},{},{}\n".format(item[0], "0", "")
                cursor += len(entity)
            else:  # 句子没有实体
                cursor += 1
                output_line = "{},{},{}\n".format(item[0], "0", "")
            writer.write(output_line)
            num_written_lines += 1
        print("--------test_data:{}, result:{}, cursor:{}, num_written_lines:{}-----------------------".format(
            len(test_data), len(preds_label), cursor, num_written_lines))

    preds_output = [["id", "text", "entity", "prob"]]
    for i, ex in enumerate(test_examples):
        line = [ex.guid.split("-")[1], ex.text_a, ex.text_b, str(preds[i])]
        preds_output.append(line)
    with open(output_predict_file + "_prob.csv", "w", encoding="utf_8_sig", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(preds_output)
    del test_data, preds_output, preds_label


def load_and_cache_examples(args, task, tokenizer, evaluate=False, type='dev'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        str(task),
        type if evaluate else 'train',
        # list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        "0.9",
        args.seed))
    # cached_features_file = os.path.join(args.data_dir, "not_exist_file")
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if evaluate and type == "test":
            examples = processor.get_test_examples(args.data_dir)
        elif evaluate and type == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        # examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def load_examples(args, examples, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[args.task_name]()
    output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=args.max_seq_length,
                                            output_mode=output_mode,
                                            pad_on_left=bool(args.model_type in ['xlnet']),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run predict on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--tpu', action='store_true',
                        help="Whether to run on the TPU defined in the environment variables")
    parser.add_argument('--tpu_ip_address', type=str, default='',
                        help="TPU IP address if none are set in the environment variables")
    parser.add_argument('--tpu_name', type=str, default='',
                        help="TPU name if none are set in the environment variables")
    parser.add_argument('--xrt_tpu_config', type=str, default='',
                        help="XRT TPU config if none are set in the environment variables")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    if args.tpu:
        if args.tpu_ip_address:
            os.environ["TPU_IP_ADDRESS"] = args.tpu_ip_address
        if args.tpu_name:
            os.environ["TPU_NAME"] = args.tpu_name
        if args.xrt_tpu_config:
            os.environ["XRT_TPU_CONFIG"] = args.xrt_tpu_config

        assert "TPU_IP_ADDRESS" in os.environ
        assert "TPU_NAME" in os.environ
        assert "XRT_TPU_CONFIG" in os.environ

        import torch_xla
        import torch_xla.core.xla_model as xm
        args.device = xm.xla_device()
        args.xla_model = xm

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    logger.info("Training/evaluation parameters %s", args)
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    dataset = processor.get_train_examples(args.data_dir)
    data_labels = [d.label for d in dataset]
    test_examples = processor.get_test_examples(args.data_dir)

    # Test
    if args.do_test and args.do_train is False:
        # del model
        # gc.collect()
        # args.do_train = False
        model = model_class.from_pretrained(args.model_name_or_path)
        # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        model.to(args.device)
        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        preds = test(args, test_examples, model, tokenizer, prefix="{}".format(args.model_name_or_path.split('/')[-1]))
        logger.info("-----------preds type:{} shape:{}".format(type(preds), preds.shape))
        logging.info("preds[0:5]: {}".format(preds[0:5]))
        return

    # 五折交叉验证
    oof_test = np.zeros((len(test_examples), 2), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train_index, eval_index) in enumerate(skf.split(dataset, data_labels)):
        logging.info("Fold:{} train_index:{}".format(fold, str(train_index[0:20])))
        logging.info("Fold:{} eval_index:{}".format(fold, str(eval_index[0:20])))
        # 每一折开始要重新加载模型
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        train_examples = [dataset[i] for i in train_index]
        eval_examples = [dataset[i] for i in eval_index]

        # Training
        if args.do_train:
            # train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
            global_step, tr_loss = train(args, train_examples, eval_examples, model, tokenizer, fold, tb_writer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and not args.tpu:
            # Create output directory if needed
            output_dir = os.path.join(args.output_dir, 'fold-{}-checkpoint-{}'.format(fold, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))

            # Load a trained model and vocabulary that you have fine-tuned
            # model = model_class.from_pretrained(args.output_dir)
            # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            # model.to(args.device)

        # Evaluation
        results = {}
        if args.do_eval and args.local_rank in [-1, 0]:
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in
                    sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
                result = evaluate(args, eval_examples, model, tokenizer, prefix=prefix)
                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                results.update(result)

        # Test
        if args.do_test:
            # del model
            # gc.collect()
            # args.do_train = False
            # model = model_class.from_pretrained(args.output_dir)
            # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            # model.to(args.device)
            # if args.fp16:
            #     model.half()
            # model.to(device)
            if args.local_rank != -1:
                try:
                    from apex.parallel import DistributedDataParallel as DDP
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

                model = DDP(model)
            elif args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            preds = test(args, test_examples, model, tokenizer, prefix="fold_{}_test_results".format(fold))
            logger.info("-----------preds type:{} shape:{}".format(type(preds), preds.shape))
            logging.info("preds[0:5]: {}".format(preds[0:5]))
            oof_test += preds
            del preds

        del model, train_examples, eval_examples
        gc.collect()
        # torch.cuda.empty_cache()
    output_predict_file = os.path.join(args.output_dir, "all_fold_test_results")
    oof_test /= 5
    logging.info("oof_test[0:5]: {}".format(oof_test[0:5]))
    write_test_results(output_predict_file, oof_test, test_examples)

    if args.local_rank in [-1, 0]:
        tb_writer.close()


if __name__ == "__main__":
    main()
