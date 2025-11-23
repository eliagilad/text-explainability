from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import itertools
import numpy as np
import torch
import hedge_bert as hedge
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from copy import copy, deepcopy
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (convert_examples_to_features,
                        output_modes, processors)
import time

# Elia
import json

os.environ["CUDA_VISIBLE_DEVICES"]="1"

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def tree_wo_scores(tree):
    new_tree = {}

    for key in tree.keys():
        level_spans = [span[0] for span in tree[key]]
        new_tree[key] = level_spans

    return new_tree

def convert_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_floats(elem) for elem in obj)
    elif isinstance(obj, (np.floating, np.generic)) and isinstance(obj.item(), float):
        obj_con = float(round(obj.item(), 4))
        return obj_con
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)

    return obj

def evaluate(args, model, tokenizer, eval_dataset, fileobject, start_pos=0, end_pos=2000, vis=-1):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    count = start_pos
    start_time = time.time()
    #cur_line = 1

    for batch in itertools.islice(eval_dataloader, start_pos, end_pos):
        # if count + 1 < start_line:
        #     cur_line += 1
        #     continue

        model.eval()
        #batch = tuple(t.to(args.device) for t in batch)
        device = next(model.parameters()).device
        batch = tuple(t.to(device) for t in batch)
        count += 1

        if count % 10 == 0:
            print(f"batch {count}")

        fileobject.write(str(count))
        fileobject.write('\n')
        ori_text_idx = list(batch[0].cpu().numpy()[0])
        if 0 in ori_text_idx:
            ori_text_idx = [idx for idx in ori_text_idx if idx != 0]
        pad_start = len(ori_text_idx)

        time0 = time.time()
        print('timestamp1: {}'.format(time0 - start_time))

        with torch.no_grad():
            inputs = {'input_ids':      torch.unsqueeze(batch[0][0,:pad_start], 0),
                      'attention_mask': torch.unsqueeze(batch[1][0,:pad_start], 0),
                      'token_type_ids': torch.unsqueeze(batch[2][0,:pad_start], 0) if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        time1 = time.time()
        print('prediciton time: {}'.format(time1 - time0))

        print(count,len(inputs['input_ids'][0]) - 2)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        for btxt in ori_text_idx:
            if (tokenizer.ids_to_tokens[btxt] != '[CLS]' and tokenizer.ids_to_tokens[btxt] != '[SEP]'):
                fileobject.write(tokenizer.ids_to_tokens[btxt])
                fileobject.write(' ')
        fileobject.write(' >> ')
        if batch[3].cpu().numpy()[0] == 0:
            fileobject.write('0')
            fileobject.write(' ||| ')
        else:
            fileobject.write('1')
            fileobject.write(' ||| ')

        shap = hedge.HEDGE(model, inputs, args, thre=100)

        time2 = time.time()
        print('timestamp3: {}'.format(time2 - time1))

        # Elia: change window/number of neighbors
        #shap.compute_shapley_hier_tree(model, inputs, 2)
        shap.compute_shapley_hier_tree(model, inputs, 1)

        time3 = time.time()
        print('Build tree time: {}'.format(time3 - time2))

        # word_list, _ = shap.get_importance_phrase()
        word_list, score_list = shap.get_importance_phrase()
        time4 = time.time()
        print('Importance score time: {}'.format(time4 - time3))

        # Elia's output
        # Original input tokens (excluding CLS/SEP)
        text_ids = inputs['input_ids'][0].detach().cpu().numpy()
        tokens = [tokenizer.ids_to_tokens[idx] for idx in text_ids if
                  idx not in [tokenizer.cls_token_id, tokenizer.sep_token_id]]

        # Construct a readable explanation object
        explanation = {
            "text_tokens": tokens,
            "true_label": int(inputs['labels'].cpu().numpy()[0]),
            "predicted_label": int(np.argmax(model(**inputs)[1].detach().cpu().numpy())),
            "spans": {},
            #"tree": convert_floats(shap.hier_tree)
            "tree": tree_wo_scores(shap.hier_tree)
        }

        for feaidx, score in zip(word_list, score_list):
            # Convert index span to actual tokens
            span_tokens = [tokens[i] for i in feaidx if i < len(tokens)]
            explanation["spans"][str(feaidx)] = float(round(score, 4))

        #print(type(explanation["tree"]["0"][0][1]))

        # Save to a JSON file (named by sentence index or timestamp)
        with open(f"hedge_output_{count}_eraser.json", "w") as f_out:
            try:
                json.dump(convert_floats(explanation), f_out, indent=2)
            except TypeError as e:
                print("Serialization failed on:", type(e), e)

        time5 = time.time()
        print('Save json time: {}'.format(time5 - time4))

        # Original output
        for feaidx in word_list:
            if len(feaidx) == 1:
                if tokenizer.ids_to_tokens[ori_text_idx[feaidx[0]]] != '[CLS]' and tokenizer.ids_to_tokens[ori_text_idx[feaidx[0]]] != '[SEP]':
                    fileobject.write(str(feaidx[0]))
                    fileobject.write(' ')
            else:
                fea_end = -1
                for fea in feaidx[-1::-1]:
                    if tokenizer.ids_to_tokens[ori_text_idx[fea]] != '[CLS]' and tokenizer.ids_to_tokens[ori_text_idx[fea]] != '[SEP]':
                        fea_end = fea
                        break
                if fea_end > -1 and fea_end>feaidx[0]:
                    fileobject.write(str(feaidx[0]))
                    fileobject.write('-')
                    fileobject.write(str(fea_end))
                    fileobject.write(' ')

        fileobject.write(' >> ')
        if np.argmax(logits.detach().cpu().numpy(), axis=1) == 0:
            fileobject.write('0')
        else:
            fileobject.write('1')
        fileobject.write('\n')
        if vis > -1:
            shap.visualize_tree(inputs, tokenizer, fontsize=10, tag=start_pos)

        if count == 1:
            time0 = time.time()
            print('First batch time: {}'.format(time0 - start_time))
    end_time = time.time()
    print('Elasped time: {}'.format(end_time-start_time))

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    eval_acc = (preds == out_label_ids).mean()

    return eval_loss, eval_acc

def load_and_cache_examples(args, task, tokenizer, type, max_lines=100000):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        ## Elia: add weights_only=False
        features = torch.load(cached_features_file, weights_only=False)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir, max_lines=max_lines)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_dir", default='./dataset/IMDB', type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--model_type", default='bert', type=str,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str,
                    help="Path to pre-trained model or shortcut name selected in the list: ")
parser.add_argument("--task_name", default='SST-2', type=str,
                    help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
parser.add_argument("--output_dir", default='./output/IMDB', type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length", default=250, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--do_lower_case", default=True,
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=1.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=5.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=111,
                    help="random seed for initialization")
parser.add_argument('--gpu', default=-1, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--start_pos', default=0, type=int, help='start position in the dataset')
parser.add_argument('--end_pos', default=2000, type=int, help='end position in the dataset')
parser.add_argument('--visualize', type=int, default=-1, help='index of the sentence to visualize, set to -1 to generate interpretations for all the sentences')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# if args.gpu > -1:
#     args.device = "cuda"
# else:
#     args.device = "cpu"
args.n_gpu = 1

# Set seed
set_seed(args)

# Prepare GLUE task
args.task_name = args.task_name.lower()
processor = processors[args.task_name]()
args.output_mode = output_modes[args.task_name]
label_list = processor.get_labels()
num_labels = len(label_list)
args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

# Load a trained model and vocabulary that you have fine-tuned
model = model_class.from_pretrained(args.output_dir)
tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
#model.to(args.device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if __name__ == '__main__':
    print("Elia's version")

    if args.visualize > -1:
        start_pos = args.visualize
        end_pos = start_pos + 1
    else:
        start_pos = args.start_pos
        end_pos = args.end_pos

    ## Elia
    max_lines_to_process = 200
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, type='test', max_lines=max_lines_to_process)
    file_name = 'hedge_bert_imdb_' + str(start_pos) + '-'+str(end_pos)+'.txt'
    #start_pos = 2
    with open(file_name, 'w') as f:
        test_loss, test_acc = evaluate(args, model, tokenizer, test_dataset, f, start_pos, end_pos, args.visualize)
    print('\ntest_loss {:.6f} | test_acc {:.6f}'.format(test_loss, test_acc))
