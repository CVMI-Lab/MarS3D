sync_bn = False
enable_amp = False
empty_cache = False
workers = 16
batch_size = 16
batch_size_val = 4
batch_size_test = 1
seed = None  # train process will init a random seed and record
log_freq = 1
eval_freq = 1
save_freq = 1
save_start_rate = 0.5
save_path = "exp"
weight = None   # path to initial weight (default: none)
resume = None  # path to latest checkpoint (default: none)
evaluate = True
metric = "mIoU"

distributed = True
train_gpu = [0, 1]
test_gpu = [0]
dist_url = "tcp://localhost:8888"
dist_backend = "nccl"
multiprocessing_distributed = True
find_unused_parameters = False
param_dicts = None
world_size = 1
rank = 0

max_batch_points = 1e10
mix_prob = 0