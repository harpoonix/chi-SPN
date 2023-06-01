
_plot_settings = {
    "dpi": 300,
    "ext": "pdf"
}


def get_plot_config(setting_name):
    path = _plot_settings.get(setting_name, None)
    if path is None:
        raise ValueError(f"unknown plot config: {setting_name}")

    return path
