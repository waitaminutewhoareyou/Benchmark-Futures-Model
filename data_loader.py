import os

import numpy as np
import pandas as pd

data_dir = './data/Asset with features/'
pd.set_option('display.max_columns', 120)
pd.set_option('display.max_rows', 120)


class Dataloader:
    def __init__(self):
        self.data_dir = './data/Asset/'

    def create_xarray(self):

        time_index, features, columns = [], [], []
        data = []

        for csv_file in os.listdir(self.data_dir):
            if not csv_file.endswith("csv"):
                continue
            # get the number between two parenthesis
            asset_ix = int(csv_file[csv_file.find("(") + 1:csv_file.find(")")])

            # For now, then consider extended futures pool
            if asset_ix > 136:
                continue

            columns.append(asset_ix)

            cur_df = pd.read_csv(self.data_dir + csv_file,
                                 parse_dates={'Date': ['Year', 'Month', 'Day']},
                                 infer_datetime_format=True,
                                 keep_date_col=True,
                                 na_values=[-999, '-999'])

            cur_df.set_index('Date', inplace=True)

            # cur_df = pd.read_csv(data_dir + csv_file, index_col=0)

            if not len(time_index):
                "Name	Cur	Year	Month	Day	%change	Vol	Px	Fx	FxBase"
                time_index = cur_df.index
                features = cur_df.columns

                iterables = [time_index, features]
                index = pd.MultiIndex.from_product(iterables, names=["Date", "Features"])

            data.append(cur_df.stack(dropna=False).values)

        data = list(map(list, zip(*data)))  # transpose

        matrix = pd.DataFrame(data, index=index, columns=columns)
        return matrix


if __name__ == "__main__":
    loader = Dataloader()
    xarray = loader.create_xarray()
    xarray.to_pickle("./data/consolidated_table_wo_features")
