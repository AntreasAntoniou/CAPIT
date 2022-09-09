from hydra.core.config_store import ConfigStore


def add_hydra_configs(config_store: ConfigStore):
    config_store.store(
        group="hydra",
        name="custom_logging_path",
        node={
            "job_logging": {
                "version": 1,
                "formatters": {
                    "simple": {
                        "level": "INFO",
                        "format": "%(message)s",
                        "datefmt": "[%X]",
                    }
                },
                "handlers": {
                    "rich": {
                        "class": "rich.logging.RichHandler",
                        "formatter": "simple",
                    }
                },
                "root": {"handlers": ["rich"], "level": "INFO"},
                "disable_existing_loggers": False,
            },
            "hydra_logging": {
                "version": 1,
                "formatters": {
                    "simple": {
                        "level": "INFO",
                        "format": "%(message)s",
                        "datefmt": "[%X]",
                    }
                },
                "handlers": {
                    "rich": {
                        "class": "rich.logging.RichHandler",
                        "formatter": "simple",
                    }
                },
                "root": {"handlers": ["rich"], "level": "INFO"},
                "disable_existing_loggers": False,
            },
            "run": {
                "dir": "${current_experiment_dir}/hydra-run/${now:%Y-%m-%d_%H-%M-%S}"
            },
            "sweep": {
                "dir": "${current_experiment_dir}/hydra-multirun/${now:%Y-%m-%d_%H-%M-%S}",
                "subdir": "${hydra.job.num}",
            },
        },
    )

    return config_store
