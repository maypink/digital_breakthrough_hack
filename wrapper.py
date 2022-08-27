import datetime
from datetime import timedelta
from os.path import exists

import numpy as np
import pandas as pd
import os

from fedot.core.composer.metrics import MAE
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams
from typing import List, Union, Tuple, Dict

ROOT_PATH_DATA = os.path.join(os.getcwd(), 'data')


class FedotWrapper:
    def __init__(self):
        self.train_ts = pd.read_excel(os.path.join(ROOT_PATH_DATA, 'Train.xlsx'))
        self.path_to_models_params = os.path.join(os.getcwd(), 'pipelines')

    def get_ts_name_with_most_correlation(self, time_series: np.ndarray) -> Tuple[str, pd.Series]:
        """ Get name of ts which has the biggest correlation with specified ts"""
        horizont = self._get_horizons_to_predict(df=pd.DataFrame(time_series))
        cur_ts_len = len(time_series) - horizont[0]
        if self.train_ts.shape[0] > cur_ts_len:
            ts = self.train_ts.head(cur_ts_len)
        else:
            ts = self.train_ts
            time_series = time_series[:self.train_ts.shape[0]]
        ts['cur_ts'] = time_series[:cur_ts_len].astype(np.float)

        # calculate correlation
        corr = ts.corr()
        cor_coefs = list(corr.iloc[-1].values)[:-1]
        ts_index_with_max_corr = cor_coefs.index(max(cor_coefs))
        ts_name_with_max_corr = ts.columns[ts_index_with_max_corr]
        ts_column_with_max_corr = ts.iloc[:, ts_index_with_max_corr]
        return ts_name_with_max_corr, ts_column_with_max_corr

    @staticmethod
    def _get_horizons_to_predict(df: Union[pd.DataFrame, pd.Series]) -> Dict[int, int]:
        """ How far ahead to predict """
        forecast_count = {}
        for column in df.columns:
            value_column = df[column].value_counts()
            if "Forecast" not in value_column:
                value_column = 0
            else:
                value_column = value_column['Forecast']
            forecast_count[column] = value_column
        return forecast_count

    def predict(self, root_data_path: str):
        for file in os.listdir(root_data_path):
            file_path = os.path.join(root_data_path, file)
            if 'Test' not in file_path:
                continue
            df = pd.read_excel(file_path, sheet_name='Monthly')
            result_df = pd.DataFrame()
            for column in df.columns:
                if 'Unnamed' in column:
                    continue

                test_number = self._get_test_number(file_name=file)

                ts_name_with_max_corr, ts_with_max_corr = \
                    self.get_ts_name_with_most_correlation(time_series=df[column].values)

                pipeline = self._get_pipeline(ts_name_with_max_corr=ts_name_with_max_corr)

                horizon = self._get_horizons_to_predict(pd.DataFrame(df[column].values))[0]

                # relative time series
                if horizon == 0:
                    result_df[column] = df[column].values
                    continue

                task = Task(TaskTypesEnum.ts_forecasting,
                            TsForecastingParams(forecast_length=horizon))

                time_series = df[column].values

                train_ts = time_series[:-horizon]

                train_data = InputData(idx=np.arange(len(train_ts)),
                                       features=train_ts,
                                       target=train_ts,
                                       task=task,
                                       data_type=DataTypesEnum.ts)

                test_data = InputData(idx=np.arange(len(train_ts)),
                                      features=train_ts,
                                      target=None,
                                      task=task,
                                      data_type=DataTypesEnum.ts)

                df.fillna(0)

                pipeline.fit(input_data=train_data)
                pipeline.fine_tune_all_nodes(
                    loss_function=MAE.metric,
                    input_data=train_data,    # TODO: check if there will be overfitting
                    timeout=1)

                forecast = np.ravel(pipeline.predict(test_data).predict)

                result = self._complete_column_with_preds(df[column], forecast)
                result_df[column] = result
            self._save_result(test_number=test_number, result_df=result_df)

    def _complete_column_with_preds(self, start_data: pd.Series, forecast: np.ndarray):
        """ Fill start data with predictions """
        horizon = self._get_horizons_to_predict(pd.DataFrame(start_data.values))[0]
        start_data.values[-horizon:] = forecast
        return start_data

    @staticmethod
    def _get_test_number(file_name: str) -> int:
        """ Get the number of current test """
        # number = file_name.split('.xlsx')[0].split('Test_input_')[1]
        number = file_name.split('.xlsx')[0].split('Test_example')[1]    # for local testing
        return int(number)

    def _get_pipeline(self, ts_name_with_max_corr: str) -> Pipeline:
        """ Get pipeline with the biggest correlation """
        pipeline = None
        for file in os.listdir(self.path_to_models_params):
            ts_name_with_max_corr = ts_name_with_max_corr.replace('/', '')[:-1]
            if ts_name_with_max_corr in file:
                model_dir = os.path.join(self.path_to_models_params, file)
                for model_file in os.listdir(model_dir):
                    if model_file.endswith('.json'):
                        pipeline = Pipeline().from_serialized(source=os.path.join(model_dir, model_file))
        return pipeline

    @staticmethod
    def _save_result(test_number: int, result_df: pd.DataFrame):
        result_file_name = f'Test_output_{test_number}.xlsx'

        result_path = os.path.join(ROOT_PATH_DATA, 'results')
        if not exists(result_path):
            os.makedirs(result_path)

        path_to_file = os.path.join(result_path, result_file_name)

        print(f'Results were saved to: {path_to_file}')
        result_df.to_excel(path_to_file)


if __name__ == '__main__':
    wrap = FedotWrapper()
    root_data_path = os.path.join(ROOT_PATH_DATA)
    wrap.predict(root_data_path=root_data_path)
