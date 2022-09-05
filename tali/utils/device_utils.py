import nvidia_smi


def get_current_gpu_memory_stats():
    nvidia_smi.nvmlInit()

    device_count = nvidia_smi.nvmlDeviceGetCount()
    info_string = "\n"
    for i in range(device_count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        info_string += (
            f"Device {i}: {nvidia_smi.nvmlDeviceGetName(handle)},\n"
            f"Memory : ({100 * info.free / info.total}% free): \n"
            f"{info.total}(total), "
            f"{info.free} (free), {info.used} (used)\n"
        )

    nvidia_smi.nvmlShutdown()
    return info_string
