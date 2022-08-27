import numpy as np
import pandas as pd
import os

from fedot.core.composer.metrics import MAE
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams


ROOT_PATH_DATA = os.path.join(os.getcwd(), 'data')


class FedotWrapper:
    def __init__(self):
        self.train_ts = pd.read_excel(os.path.join(ROOT_PATH_DATA, 'data', 'Train.xlsx'))
        self.path_to_models_params = os.path.join(ROOT_PATH_DATA, 'models_params')

    def get_ts_name_with_most_correlation(self, time_series: np.ndarray):
        """ Get name of ts which has the biggest correlation with specified ts"""
        horizont = self._get_horizonts_to_predict(df=pd.DataFrame(time_series))
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
        return ts_name_with_max_corr

    @staticmethod
    def _get_horizonts_to_predict(df: pd.DataFrame):
        """ How far ahead to predict """
        forecast_count = {}
        for column in df.columns:
            value_column = df[column].value_counts()
            if "Forecast" not in value_column:
                value_column = 0
            else:
                value_column = value_column['Forecast']
            print(value_column)
            forecast_count[column] = value_column
        return forecast_count

    def predict(self, root_data_path: str):
        for file in os.listdir(root_data_path):
            file_path = os.path.join(root_data_path, file)
            if 'Test' not in file_path:
                continue
            df = pd.read_excel(file_path, sheet_name='Monthly')
            for i in df.columns:
                if 'Unnamed' in i:
                    continue

                ts_name_with_max_corr = self.get_ts_name_with_most_correlation(time_series=df[i].values)

                pipeline = self._get_pipeline(ts_name_with_max_corr=ts_name_with_max_corr)

                task = Task(TaskTypesEnum.ts_forecasting,
                            TsForecastingParams(forecast_length=12))
                time_series = df[i].values

                test_data = InputData(idx=np.arange(len(time_series)),
                                      features=time_series,
                                      target=time_series,
                                      task=task,
                                      data_type=DataTypesEnum.ts)

                train_input = InputData(idx=np.arange(len(time_series)),
                                        features=time_series,
                                        target=time_series,
                                        task=task,
                                        data_type=DataTypesEnum.ts)

                train_data, test_data = train_test_data_setup(train_input)

                # run AutoML model design in the same way
                pipeline = PipelineBuilder().add_node('lagged').add_node('ridge').to_pipeline()
                df.fillna(0)
                pipeline.fine_tune_all_nodes(
                    loss_function=MAE.metric,
                    input_data=train_data,
                    timeout=1)

                forecast = np.ravel(pipeline.predict(test_data).predict)
                target = np.ravel(test_data.target)

    def _get_pipeline(self, ts_name_with_max_corr: str) -> Pipeline:
        """ Get pipeline with biggest correlation """
        pipeline = None
        for file in self.path_to_models_params:
            if ts_name_with_max_corr in file:
                pipeline = Pipeline().from_serialized(source=os.path.join(self.path_to_models_params, file))
        return pipeline


if __name__ == '__main__':
    wrap = FedotWrapper()
    root_data_path = os.path.join(ROOT_PATH_DATA, 'data')
    wrap.predict(root_data_path=root_data_path)
