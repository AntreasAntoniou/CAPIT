from tali.utils.arg_parsing import DictWithDotNotation


def convert_dict_kwargs_to_dictwithdotnotation(method):
    def convert(*args, **kwargs):
        kwargs_cache = kwargs.copy()
        for key, value in kwargs_cache.items():
            if isinstance(value, dict):
                kwargs[key] = DictWithDotNotation(value)
        return method(*args, **kwargs)

    return convert
