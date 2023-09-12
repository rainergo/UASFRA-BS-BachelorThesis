import pandas as pd
import joblib as jl

from modelling.ml import ML


class DataAccessDescriptor:
    def __get__(self, obj, objtype=None):
        file_path = obj.source_file_path.lower()
        try:
            if file_path.endswith('.zip') or file_path.endswith('.csv'):
                return pd.read_csv(obj.source_file_path)
            elif file_path.endswith('.joblib'):
                return jl.load(obj.source_file_path)
            else:
                raise ValueError('FILE NOT RETRIEVED: File extension unknown. Extension must be ".csv" or ".joblib" !')
        except OSError:
            print(f'File could not be read from file path "{obj.source_file_path}" !')

    def __set__(self, obj, value):
        file_path = obj.target_file_path.lower()
        try:
            if file_path.endswith('.zip') or file_path.endswith('.csv'):
                value.to_csv(obj.target_file_path)
            elif file_path.endswith('.joblib'):
                jl.dump(value, obj.target_file_path)
            else:
                raise ValueError('FILE NOT SAVED: File extension unknown. Extension must be ".csv" or ".joblib" !')
            print(f'The file was successfully saved as/to: {obj.target_file_path} !')
        except OSError:
            print(f'File could not be written to file path "{obj.target_file_path}" !')



class InsideAccess(ML):
    _stored_dataframe = DataAccessDescriptor()
    _stored_fitted_estimator = DataAccessDescriptor()
    _stored_prediction_input_object = DataAccessDescriptor()
    _stored_explanation_input_object = DataAccessDescriptor()

    def __init__(self):
        if super().configuration is not None:
            super().__init__()
            self.dataframe = None
            self.fitted_estimator = None
            self.data_for_prediction = None
            self.prediction_input_object = None
            self.explanation_input_object = None
        else:
            raise PermissionError('configuration and access to configuration file not set yet !')

    def load_dataframe(self, config_section: str, config_name: str, model_type: str = None):
        self.source_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self.dataframe = self._stored_dataframe

    def save_dataframe(self, config_section: str, config_name: str, model_type: str = None):
        self.target_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self._stored_dataframe = self.dataframe

    def load_fitted_estimator(self, config_section: str, config_name: str, model_type: str = None):
        self.source_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self.fitted_estimator = self._stored_fitted_estimator

    def save_fitted_estimator(self, config_section: str, config_name: str, model_type: str = None):
        self.target_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self._stored_fitted_estimator = self.fitted_estimator

    def load_data_for_prediction(self, config_section: str, config_name: str, model_type: str = None):
        self.source_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self.dataframe = self._stored_dataframe

    def save_data_for_prediction(self, config_section: str, config_name: str, model_type: str = None):
        self.target_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self._stored_dataframe = self.dataframe

    def load_prediction_input_object(self, config_section: str, config_name: str, model_type: str = None):
        self.source_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self.prediction_input_object = self._stored_prediction_input_object

    def save_prediction_input_object(self, config_section: str, config_name: str, model_type: str = None):
        self.target_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self._stored_prediction_input_object = self.prediction_input_object

    def load_explanation_input_object(self, config_section: str, config_name: str, model_type: str = None):
        self.source_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self.explanation_input_object = self._stored_explanation_input_object

    def save_explanation_input_object(self, config_section: str, config_name: str, model_type: str = None):
        self.target_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self._stored_explanation_input_object = self.explanation_input_object


class OutsideAccess(ML):
    _stored_prediction_dataframe = DataAccessDescriptor()
    _stored_result_dataframe = DataAccessDescriptor()
    _stored_quantile_portfolio = DataAccessDescriptor()
    _stored_transfer_model_input_object = DataAccessDescriptor()

    # These are the aggregated results (ONE concatenated pd.DataFrame for each) from the individual instances:
    result_dataframe = None
    quantile_portfolio = None

    def __init__(self):
        if super().configuration is not None:
            super().__init__()
            self.prediction_dataframe = None
            self.transfer_model_input_object = None
        else:
            raise PermissionError('configuration and access to configuration file not set yet !')

    # Must be at the class level as attribute is class attribute
    def load_result_dataframe(self, config_section: str, config_name: str, model_type: str = None):
        self.source_file_path = self.configuration.get_path(config_section, config_name, model_type)
        OutsideAccess.result_dataframe = self._stored_result_dataframe

    # Must be at the class level as attribute is class attribute
    def save_result_dataframe(self, config_section: str, config_name: str, model_type: str = None):
        self.target_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self._stored_result_dataframe = OutsideAccess.result_dataframe

    # Must be at the class level as attribute is class attribute
    def load_quantile_portfolio(self, config_section: str, config_name: str, model_type: str = None):
        self.source_file_path = self.configuration.get_path(config_section, config_name, model_type)
        OutsideAccess.quantile_portfolio = self._stored_quantile_portfolio

    # Must be at the class level as attribute is class attribute
    def save_quantile_portfolio(self, config_section: str, config_name: str, model_type: str = None):
        self.target_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self._stored_quantile_portfolio = OutsideAccess.quantile_portfolio

    def load_prediction_dataframe(self, config_section: str, config_name: str, model_type: str = None):
        self.source_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self.prediction_dataframe = self._stored_prediction_dataframe

    def save_prediction_dataframe(self, config_section: str, config_name: str, model_type: str = None):
        self.target_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self._stored_prediction_dataframe = self.prediction_dataframe

    def load_transfer_model_input_object(self, config_section: str, config_name: str, model_type: str = None):
        self.source_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self.transfer_model_input_object = self._stored_transfer_model_input_object

    def save_transfer_model_input_object(self, config_section: str, config_name: str, model_type: str = None):
        self.target_file_path = self.configuration.get_path(config_section, config_name, model_type)
        self._stored_transfer_model_input_object = self.transfer_model_input_object
