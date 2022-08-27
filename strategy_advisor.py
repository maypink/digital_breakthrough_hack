from typing import Union

import pandas as pd
# month_data = pd.read_excel(sheet_name='Monthly')
# month_data = pd.read_excel(sheet_name='Quarterly')
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


class StrategyAdvisor:
    def is_exogenous_df(self, df: pd.DataFrame):
        return self.return_exogenous_df(df)

    def return_exogenous_df(self, df: pd.DataFrame) -> pd.DataFrame:
        f_c = self._get_horizonts_to_predict(df)
        df_res = pd.DataFrame()
        for k, v in f_c.items():
            if v == 0:
                pd.concat([df_res, pd.DataFrame(columns=[k])], axis=0)
        return df_res

    @staticmethod
    def _get_horizonts_to_predict(df: Union[pd.DataFrame, pd.Series]):
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






