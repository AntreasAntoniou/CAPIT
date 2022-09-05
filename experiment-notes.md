
```bash
python3 \
run.py \
hydra.verbose=False \
resume=False \
batch_size=2 \
datamodule.dataloader_config.num_workers=2 \
trainer=ddp \
trainer.gpus=2 \
datamodule=debug-tali \
datamodule.dataset_config.modality_config.image=True \
datamodule.dataset_config.modality_config.text=True \
datamodule.dataset_config.modality_config.audio=True \
datamodule.dataset_config.modality_config.video=True \
datamodule.dataset_config.rescan_paths=False \
datamodule.dataloader_config.prefetch_factor=1 \
datamodule.dataset_config.dataset_size_identifier=deci \
wandb_project_name=gcp-dev \
model=base_modus_prime_vi-transformer16 \
datamodule.dataloader_config.train_num_samples=2000 \
trainer.val_check_interval=0.25
```

```bash
python3 \
run.py \
hydra.verbose=False \
resume=False \
batch_size=12 \
datamodule.dataloader_config.num_workers=5 \
trainer=ddp \
trainer.gpus=16 \
datamodule=tali \
datamodule.dataset_config.modality_config.image=True \
datamodule.dataset_config.modality_config.text=True \
datamodule.dataset_config.modality_config.audio=True \
datamodule.dataset_config.modality_config.video=True \
datamodule.dataset_config.rescan_paths=False \
datamodule.dataloader_config.prefetch_factor=1 \
datamodule.dataset_config.dataset_size_identifier=base \
datamodule.dataset_config.num_video_frames_per_datapoint=8 \
wandb_project_name=godzilla-gcp-experiments \
model=base_modus_prime_vi-transformer16 \
trainer.val_check_interval=0.03
```

```bash
gcloud \
beta \
compute \
instances \
create \
instance-1 \
--project=tali-multi-modal \
--zone=us-central1-f \
--machine-type=a2-megagpu-16g \
--network-interface=network-tier=PREMIUM,subnet=default \
--no-restart-on-failure \
--maintenance-policy=TERMINATE \
--preemptible \
--service-account=tali-multi-modal@tali-multi-modal.iam.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/cloud-platform \
--accelerator=count=16,type=nvidia-tesla-a100 \
--create-disk=auto-delete=yes,boot=yes,device-name=instance-1,image=projects/tali-multi-modal/global/images/tali-ubuntu-cuda110-pytorch-personalized-v-1-1,mode=rw,size=250,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-balanced \
--create-disk=device-name=disk-4,mode=ro,name=disk-4,size=12000,source-snapshot=projects/tali-multi-modal/global/snapshots/tali-base-train,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-balanced \
--create-disk=device-name=disk-5,mode=ro,name=disk-5,size=1000,source-snapshot=projects/tali-multi-modal/global/snapshots/tali-val-test,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-balanced \
--no-shielded-secure-boot \
--shielded-vtpm \
--shielded-integrity-monitoring \
--reservation-affinity=any \
--provisioning-model=SPOT
```

```bash
gcloud \
beta \
compute \
instances \
create \
instance-1 \
--project=tali-multi-modal \
--zone=us-central1-f \
--machine-type=a2-megagpu-16g \
--network-interface=network-tier=PREMIUM,subnet=default \
--no-restart-on-failure \
--maintenance-policy=TERMINATE \
--preemptible \
--service-account=tali-multi-modal@tali-multi-modal.iam.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/cloud-platform \
--accelerator=count=16,type=nvidia-tesla-a100 \
--create-disk=auto-delete=yes,boot=yes,device-name=instance-1,image=projects/tali-multi-modal/global/images/tali-ubuntu-cuda110-pytorch-personalized-v-1-1,mode=rw,size=250,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-balanced \
--no-shielded-secure-boot \
--shielded-vtpm \
--shielded-integrity-monitoring \
--reservation-affinity=any \
--provisioning-model=SPOT
```

```bash
gcloud \
compute \
disks \
create \
disk-tali-train-1 \
--project=tali-multi-modal \
--type=pd-balanced \
--size=12000GB \
--zone=us-central1-f \
--source-snapshot=tali-base-train
```

```bash
gcloud \
compute \
disks \
create \
disk-tali-eval-1 \
--project=tali-multi-modal \
--type=pd-balanced \
--size=1000GB \
--zone=us-central1-f \
--source-snapshot=tali-val-test
```