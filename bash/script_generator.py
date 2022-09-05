# python run.py hydra.verbose=True trainer=default \
# resume=True batch_size=48 trainer.gpus=1 model=base_modus_prime_resnet50
#
# #######################################################################################
#
# datamodule.dataset_config.modality_config.image=True
# datamodule.dataset_config.modality_config.text=False
# datamodule.dataset_config.modality_config.video=False
# datamodule.dataset_config.modality_config.audio=True
import os


def main():
    # batch_size and gpus should be set by model
    experiment_script_dir = "experiment_runner_scripts/"
    dataset_names = ["base", "milli"]  # , "milli/tali", hecta/tali"]
    system_names = [
        "centi_modus_prime_resnet50",
        "deci_modus_prime_resnet50",
        "base_modus_prime_resnet50",
        "centi_modus_prime_vi-transformer16",
        "deci_modus_prime_vi-transformer16",
        "base_modus_prime_vi-transformer16",
    ]
    exp_dict = {}
    num_gpu_config = {1: 12, 2: 24, 4: 48, 8: 96, 16: 96}

    for num_gpus, num_workers in num_gpu_config.items():
        for use_image_modality in [True]:
            for use_audio_modality in [False, True]:
                for use_video_modality in [False, True]:
                    for use_text_modality in [False, True]:
                        for model_name in system_names:
                            for dataset_name in dataset_names:
                                if any(
                                    [
                                        use_text_modality,
                                        use_audio_modality,
                                        use_video_modality,
                                    ]
                                ):

                                    template_command = (
                                        f"fuser -k /dev/nvidia*; \\\n"
                                        f"python $CODE_DIR/run.py \\\n"
                                        f"hydra.verbose=True \\\n"
                                        f"trainer=default \\\n"
                                        f"resume=True \\\n"
                                        f"batch_size={num_gpus * 2} \\\n"
                                        f"trainer.gpus={num_gpus} \\\n"
                                        f"trainer.auto_scale_batch_size=True \\\n"
                                        f"datamodule.dataset_config.rescan_paths=True \\\n"
                                        f"datamodule.prefetch_factor=3 \\\n"
                                        f"datamodule.num_workers={num_workers} \\\n"
                                        f"model={model_name} \\\n"
                                        f"datamodule.dataset_config.dataset_size_identifier={dataset_name} \\\n"
                                        f"datamodule.dataset_config.modality_config.image={use_image_modality} \\\n"
                                        f"datamodule.dataset_config.modality_config.text={use_text_modality} \\\n"
                                        f"datamodule.dataset_config.modality_config.audio={use_audio_modality} \\\n"
                                        f"datamodule.dataset_config.modality_config.video={use_video_modality} \n\n"
                                    )
                                    exp_dict[
                                        f"{dataset_name}_{model_name}_"
                                        f"image={use_image_modality}_audio={use_audio_modality}_"
                                        f"video={use_video_modality}_text={use_text_modality}_"
                                        f"auto_scale_batch_size=True_{num_gpus}g"
                                    ] = template_command

    if not os.path.exists(experiment_script_dir):
        os.makedirs(experiment_script_dir)

    for name, script_contents in exp_dict.items():
        with open("setup_scripts/experiment_script_template.sh", "r") as f:
            template_contents = list(f.readlines())

        with open(os.path.join(experiment_script_dir, f"{name}.sh"), "w") as f:
            content_list = list(template_contents)
            content_list.append(script_contents)
            f.writelines(content_list)


if __name__ == "__main__":
    main()
