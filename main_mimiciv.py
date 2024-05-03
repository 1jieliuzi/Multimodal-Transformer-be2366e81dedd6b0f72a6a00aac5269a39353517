import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import warnings
import time
import logging
logger = logging.getLogger(__name__)
from model import *
from train import *
from checkpoint import *
from util import *
from accelerate import Accelerator
from interp import *




class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main():
    # args = {
    #     "task": "ihm-48-cxr-notes-ecg",
    #     "modeltype": "TS_CXR_Text_ECG",
    #     "file_path": "/cis/home/charr165/Documents/multimodal/preprocessing",
    #     "output_dir": "Checkpoints",
    #     "tensorboard_dir": None,
    #     "seed": 42,
    #     "mode": "train",
    #     "eval_score": ['auc', 'auprc', 'f1'],
    #     "num_labels": 2,
    #     "max_length": 128,
    #     "pad_to_max_length": False,  # default for action='store_true' is False
    #     "model_path": None,
    #     "train_batch_size": 8,
    #     "eval_batch_size": 32,
    #     "num_update_bert_epochs": 10,
    #     "num_train_epochs": 6,
    #     "txt_learning_rate": 5e-5,
    #     "ts_learning_rate": 0.0004,
    #     "gradient_accumulation_steps": 1,
    #     "weight_decay": 0.01,
    #     "lr_scheduler_type": "linear",
    #     "pt_mask_ratio": 0.15,
    #     "mean_mask_length": 3,
    #     "chunk": False,
    #     "chunk_type": 'sent_doc_pos',
    #     "warmup_proportion": 0.10,
    #     "kernel_size": 1,
    #     "num_heads": 8,
    #     "layers": 3,
    #     "cross_layers": 3,
    #     "embed_dim": 128,
    #     "irregular_learn_emb_ts": True,
    #     "irregular_learn_emb_text": True,
    #     "reg_ts": True,
    #     "tt_max": 48,
    #     "embed_time": 64,
    #     "ts_to_txt": False,
    #     "txt_to_ts": False,
    #     "dropout": 0.10,
    #     "model_name": 'BioBert',
    #     "num_of_notes": 5,
    #     "notes_order": None,
    #     "ratio_notes_order": None,
    #     "bertcount": 3,
    #     "first_n_item": 3,
    #     "fine_tune": False,
    #     "TS_mixup": True,
    #     "mixup_level": 'batch',
    #     "fp16": False,
    #     "debug": False,
    #     "generate_data": False,
    #     "FTLSTM": False,
    #     "Interp": False,
    #     "cpu": False,
    #     "datagereate_seed": 42,
    #     "TS_model": 'Atten',
    #     "cross_method": 'self_cross',
    #     # "debug": True,
    #     "gating_function": 'laplace',
    #     "num_of_experts": 12,
    #     "hidden_size": 512,
    #     "top_k": 4,
    #     "irregular_learn_emb_cxr": True,
    #     "use_pt_text_embeddings": True,
    #     "num_modalities": 4
    # }

    # args = Struct(**args)

    args = parse_args()

    print(args)

    if args.fp16:
        args.mixed_precision="fp16"
    else:
        args.mixed_precision="no"
    accelerator = Accelerator(mixed_precision=args.mixed_precision,cpu=args.cpu)

    device = accelerator.device
    print(device)
    os.makedirs(args.output_dir, exist_ok = True)
    if args.tensorboard_dir!=None:
        writer = SummaryWriter(args.tensorboard_dir)
    else:
        writer=None

    warnings.filterwarnings('ignore')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    make_save_dir(args)

    if args.seed==0:
        copy_file(args.ck_file_path+'model/', src=os.getcwd())
    if args.mode=='train':
        if 'Text' in args.modeltype:
            BioBert, BioBertConfig, tokenizer = loadBert(args,device)
        else:
            BioBert, tokenizer = None, None

        from data_mimiciv import data_perpare

        train_dataset, train_sampler, train_dataloader = data_perpare(args, 'train', tokenizer)
        val_dataset, val_sampler, val_dataloader = data_perpare(args, 'val', tokenizer)
        _, _, test_data_loader = data_perpare(args,'test',tokenizer)

    if args.modeltype == 'Text':
        # pure text
        model= TextModel(args=args,device=device,orig_d_txt=768,Biobert=BioBert)
    elif args.modeltype == 'TS':
        # pure time series
        model= TSMixed(args=args,device=device,orig_d_ts=30,orig_reg_d_ts=60, ts_seq_num=args.tt_max)
    else:
        # multimodal fusion
        model= MULTCrossModel(args=args,device=device,orig_d_ts=30, orig_reg_d_ts=60, orig_d_txt=768,ts_seq_num=args.tt_max,text_seq_num=args.num_of_notes,Biobert=BioBert )
    print(device)
    
    # TODO: CXR learning rate
    if args.modeltype=='TS':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    elif args.modeltype=='TS_CXR':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    elif 'Text' in args.modeltype:
        optimizer= torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n]},
                {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.txt_learning_rate}
            ], lr=args.ts_learning_rate)
    else:
        raise ValueError("Unknown modeltype in optimizer.")

    model, optimizer, train_dataloader,val_dataloader,test_data_loader = \
    accelerator.prepare(model, optimizer, train_dataloader,val_dataloader,test_data_loader)

    trainer_irg(model=model,args=args,accelerator=accelerator,train_dataloader=train_dataloader,\
        dev_dataloader=val_dataloader, test_data_loader=test_data_loader, device=device,\
        optimizer=optimizer,writer=writer)
    eval_test(args,model,test_data_loader, device)
    print(f"New maximum memory allocated on GPU: {torch.cuda.max_memory_allocated(device)} bytes")
    print(f'Results saved in:\n{args.ck_file_path}')


if __name__ == "__main__":

    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
