import torch
import argparse
import numpy as np
from pathlib import Path
from trainer import Trainer
from omegaconf import OmegaConf
import torch.backends.cudnn as cudnn
import random
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
from torch.utils.tensorboard import SummaryWriter
def get_args_parser():
    parser = argparse.ArgumentParser('InstDiff training and evaluation script', add_help=False)
    parser.add_argument("--DATA_ROOT", type=str, default="", help="path to DATA")
    parser.add_argument("--OUTPUT_ROOT", type=str, default="OUTPUT", help="path to OUTPUT")
    parser.add_argument("--name", type=str, default="checkpoint-daytimeclear-withmask-8gpu",
                        help="checkpoints and related files will be stored in OUTPUT_ROOT/name")
    parser.add_argument("--seed", type=int, default=123, help="used in sampler")
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument("--yaml_file", type=str, default="configs/train_sd15.yaml",
                        help="paths to base configs.")
    parser.add_argument("--base_learning_rate", type=float, default=5e-5, help="")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="")
    parser.add_argument("--scheduler_type", type=str, default='constant', help="cosine or constant")
    parser.add_argument("--batch_size", type=int, default=8, help="")
    parser.add_argument("--workers", type=int, default=8, help="")
    parser.add_argument("--official_ckpt_name", type=str, default="pretrained/v1-5-pruned-emaonly.ckpt",
                        help="SD ckpt name and it is expected in DATA_ROOT, thus DATA_ROOT/official_ckpt_name must exist")
    parser.add_argument("--ckpt", type=lambda x: x if type(x) == str and x.lower() != "none" else None, default="pretrained/instancediffusion_sd15.pth",
                        help=("If given, then it will start training from this ckpt"
                              "It has higher priority than official_ckpt_name, but lower than the ckpt found in autoresuming (see trainer.py) ")
                        )
    parser.add_argument('--enable_ema', default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--ema_rate", type=float, default=0.9999, help="")
    parser.add_argument("--total_iters", type=int, default=500000, help="")
    parser.add_argument("--save_every_iters", type=int, default=10000, help="")
    parser.add_argument("--total_epochs", type=int, default=40, help="")
    parser.add_argument("--disable_inference_in_training", type=lambda x: x.lower() == "true", default=False,
                        help="Do not do inference, thus it is faster to run first a few iters. It may be useful for debugging ")
    parser.add_argument("--wandb_name", type=str, default="instdiff", help="name for wandb run")
    parser.add_argument('--fp32', type=lambda x: x.lower() == "true", default=False, help="use fp32")
    parser.add_argument("--train_file", type=str, default="dataset/daytimeclear_train_withmask.txt", help="list of JSON files for model training")
    parser.add_argument("--count_dup", type=lambda x: x.lower() == "true", default=True, help="count number of duplicated classes")
    parser.add_argument("--re_init_opt", type=lambda x: x.lower() == "true", default=True, help="reinitialize optimizer and scheduler")
    parser.add_argument("--random_blip", type=float, default=0.5, help="randomly use blip embeddings")
    parser.add_argument("--use_masked_att", type=lambda x: x.lower() == "true", default=False, help="use masked attention given the bounding box or not")
    parser.add_argument("--add_inst_cap_2_global", type=lambda x: x.lower() == "true", default=False, help="add instance captions to the global captions or not")
    parser.add_argument("--use_instance_sampler", type=lambda x: x.lower() == "true", default=False, help="using multi-instance sampler during training or not")
    parser.add_argument("--mis_ratio", type=float, default=0, help="the percentage of timesteps using multi-instance-sampler")
    parser.add_argument("--use_crop_paste", type=lambda x: x.lower() == "true", default=False, help="using use_crop_paste for multi-instance sampler or not")
    parser.add_argument("--use_instance_loss", type=lambda x: x.lower() == "true", default=False, help="using instance loss")
    parser.add_argument("--instance_loss_weight", type=float, default=0.0, help="weights for instance loss")
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return parser

def main(args):
    # fix seed
    # 每个进程根据自己的local_rank设置应该使用的GPU
    # torch.cuda.set_device(args.local_rank)
    # device = torch.device('cuda', args.local_rank)

    # 初始化分布式环境，主要用来帮助进程间通信
    torch.distributed.init_process_group(backend='nccl')

    config = OmegaConf.load(args.yaml_file)
    config.update(vars(args))
    config.total_batch_size = config.batch_size
    config.local_rank = args.local_rank
    torch.cuda.set_device(args.local_rank)

    # Ensure distributed key is present and set to False
    config.distributed = True

    # create output dir
    Path(args.OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    # 只 master 进程做 logging，否则输出会很乱
    if args.local_rank == 0:
        tb_writer = SummaryWriter(comment='ddp-training')


    # create trainer
    trainer = Trainer(config)

    # start training
    trainer.start_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('InstDiff training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)