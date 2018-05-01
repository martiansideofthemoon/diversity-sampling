"""Parse all the default arguments."""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data', help='data directory containing input.txt')
parser.add_argument('--char', action='store_true', default=False, help='Train a character level RNN')
parser.add_argument('--config_file', type=str, default="config/default.yml", help='Model configuration file')
parser.add_argument('--vocab', type=str, default="vocab", help='Use SRILM processed vocabulary')
parser.add_argument('--dataset', type=str, default="ptb", help='Dataset for LM experiments')
parser.add_argument('--lm', type=str, default="LM", help='Use SRILM processed vocabulary')
parser.add_argument('--train_dir', type=str, default='save', help='directory to store checkpointed models')
parser.add_argument('--best_dir', type=str, default='save_best', help='directory to store checkpointed models')
parser.add_argument('--loss_mode', type=str, default="l1", choices=["l1", "l2", "mixed", "alternate"], help='Can be l1, mixed, l2 or adaptive')
parser.add_argument('--mixed_constant', type=float, default=0.5, help='Can be l1, mixed, l2 or adaptive')
parser.add_argument('--mode', type=str, default="train", choices=["test", "valid", "train", "generate", "beam"], help='train / test')
parser.add_argument('--gen_config', type=str, default="{\"prior\": \"a\", \"length\": 100}", help='config for generate mode')
parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "gpu"], help='gpu / cpu')
parser.add_argument('--job_id', type=str, default="job_0", help='ID of the current job')

parser.add_argument('--diversity_type', type=str, default="hamming")
parser.add_argument('--diversity_lambda', type=float, default=5)
parser.add_argument('--diversity_beam_size', type=int, default=10)
parser.add_argument('--diversity_prior', type=str, default="why")

SUMMARY = "This model uses the new augmented cost function for training"
