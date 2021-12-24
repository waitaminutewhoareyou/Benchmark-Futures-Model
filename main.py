import pandas as pd
from weight_generation import adjusted_volume_weighted
import matplotlib.pyplot as plt
from BackTest import backtest

data_dir = './data/consolidated_table_wo_features'
start_date = '2013-1-1'
end_date = '2025-1-1'
lag = 1
OUTPUT_DIR = './BackTest/Daily Return.xlsx'


class PipeLine:
    def __init__(self):
        # load data
        self.df = pd.read_pickle(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.lag = lag

        self.ret = self.df.xs("%change", level='Features')
        mask = (self.ret.index > start_date) & (self.ret.index < end_date)
        self.ret = self.ret[mask]

    def weight_generation(self):
        weights = adjusted_volume_weighted(self.df)
        return weights

    def return_generation(self):
        weights = self.weight_generation()
        ret_table = self.ret.multiply(weights, axis=0)

        return ret_table.sum(axis=1)

    def main(self):
        ret_series = self.return_generation()
        ret_series.to_excel(OUTPUT_DIR)
        return


model = PipeLine()
model.main()
