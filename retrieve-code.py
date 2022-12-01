import wandb

run = wandb.init()
artifact = run.use_artifact(
    "machinelearningbrewery/CAPInstagramImageTextFiltered/project-source:v13",
    type="code",
)
artifact_dir = artifact.download()
