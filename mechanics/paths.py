def get_by_path(data: dict, path: str):
    """Gets a value from a nested dictionary using a dot-separated path."""
    keys = path.split('.')
    val = data
    for k in keys:
        if not isinstance(val, dict):
            return None
        val = val.get(k)
        if val is None:
            return None
    return val

def set_by_path(data: dict, path: str, value) -> None:
    """Sets a value in a nested dictionary using a dot-separated path."""
    keys = path.split('.')
    for k in keys[:-1]:
        if k not in data or not isinstance(data[k], dict):
            data[k] = {}
        data = data[k]
    data[keys[-1]] = value
