from os.path import join
from configparser import ConfigParser


class ML:
    configuration = None

    def __init__(self):
        self.source_file_path = None
        self.target_file_path = None

    @classmethod
    def set_configuration(cls, path_to_config_file: str):
        cls.configuration = Configuration(path_to_config_file)


class Configuration:
    def __init__(self, path_to_config_file: str):
        self._config = ConfigParser()
        self._config.read(path_to_config_file)

    def get_config_setting(self, config_section: str, config_name: str, model_type: str = None):
        try:
            if model_type is not None:
                return eval(self._config[config_section][config_name]).get(model_type)
            else:
                return self._config[config_section][config_name]
        except KeyError:
            return f'Either section {config_section!r} or var_name {config_name!r} not found. Please check config file!'

    def get_path(self, config_section: str, config_name: str, model_type: str = None) -> str:
        return join(self.get_config_setting(config_section=config_section, config_name='file_path'),
                    self.get_config_setting(config_section=config_section, config_name=config_name,
                                            model_type=model_type))
