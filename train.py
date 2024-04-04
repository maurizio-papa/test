import torch 

from utility.loader import EpicKitchenLoader
from build_model import TAL_model
from utility.trainer import train_one_epoch
from detector.TriDet.libs.core import load_config
from detector.TriDet.libs.utils import  (save_checkpoint, make_optimizer, make_scheduler, fix_random_seed, ModelEma)


cfg_optimizer =  {
                        # solver
                        "type": "AdamW",  # SGD or AdamW
                        # solver params
                        "momentum": 0.9,
                        "weight_decay": 0.0,
                        "learning_rate": 1e-3,
                        # excluding the warmup epochs
                        "epochs": 30,
                        # lr scheduler: cosine / multistep
                        "warmup": True,
                        "warmup_epochs": 5,
                        "schedule_type": "cosine",
                        # Minimum learning rate
                        "eta_min": 1e-8,
                        # in #epochs excluding warmup
                        "schedule_steps": [],
                        "schedule_gamma": 0.1,
                 }


def train(epochs = 8):

    # build dataset and dataloader
    train_dataset = EpicKitchenLoader(feature_folder = '/ext/tensor_dir', 
                                      json_file = '/ext/annotations/epic_kitchens_100_noun.json',
                                      split = 'training',
                                      num_frames = 16,
                                      feat_stride = 8,
                                      default_fps = 30,
                                      num_classes = 300)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= 1,
        num_workers= 0,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    print(train_dataset.dict_db)

    # build model 
    model = TAL_model(chunk_size = 4, sampling_ratio = 0.3)

    # optimizer
    optimizer = make_optimizer(model, cfg_optimizer)
     
    # schedule
    num_iters_per_epoch = len(train_loader) 
    print(num_iters_per_epoch)
    scheduler = make_scheduler(optimizer, cfg_optimizer, num_iters_per_epoch)    

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, epoch, scheduler, train_loader)


if __name__ == "__main__":
    train(8)