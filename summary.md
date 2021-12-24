This report documents methodologies and findings during benchmark model research

# Architecture

The pipeline that yields benchmark models concist of three components


1.   Dataloader
     - This component screens,cleans, and output the data into a three dimensional array.
     - Such 3D data is difficult to visualize via csv file, so it is saved as .pkl  
     - ```sh
                          Asset 1   Asset 2   Asset 3
        Day 1 Feature 1    xx        xx        xx
              Feature 2    xx        xx        xx
              Feature 3    xx        xx        xx

        Day 2 Feature 1    xx        xx        xx
              Feature 2    xx        xx        xx
              Feature 3    xx        xx        xx
        ```
     - Read auxiliary information, such as leverage, sector indicator etc.  
2.   Singal generation
     - Read the .pkl, and generate daily weight as instructed by the benchmark algorithm, multiply the weight on the (transfomred) return, and then output the return series

3.   Backtest
     - Take the return series and output relevant statsitcs such as Sharpe ratio, maximum drawdown.








```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
import os

drive.mount('/content/drive')
root_dir = "/content/drive/My Drive/Quant Research/Benchmark Models/" 
%cd "/content/drive/My Drive/Quant Research/Benchmark Models/" 
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    /content/drive/My Drive/Quant Research/Benchmark Models
    


```python


class BackTest:
    def __init__(self, data, start_time=None, end_time=None):

        data = data.squeeze()
        data = data[start_time:end_time]
        self.data = data.dropna(
        )
        self.time_freq = pd.infer_freq(data.index)

        if self.time_freq == 'B' or self.time_freq == 'D':
            self.T = 250
            self.monthly_return = data.resample('M').agg(lambda x: (x + 1).prod() - 1)
            self.time_freq = 'D'  # reset business day frequency to daily frequency

        elif self.time_freq == 'M':
            self.T = 12
            self.monthly_return = data

        else:
            print("Can't detect time frequency. Using daily frequency by default. ")
            self.T = 250
            self.monthly_return = data.resample('M').agg(lambda x: (x + 1).prod() - 1)
            self.time_freq = 'D'  # reset business day frequency to daily frequency
             #raise Exception("Can't detect time frequency")

        self.AUM = (1 + self.monthly_return).cumprod()
        self.dd = 1 - self.AUM / np.maximum.accumulate(self.AUM)

    def total_return(self):
        return np.prod(1 + self.data.values) - 1

    def annualized_return(self):
        return (1 + self.total_return()) ** (self.T / len(self.data)) - 1

    def annualized_volatility(self):
        return np.std(self.data.values) * np.sqrt(self.T)

    def information_ratio(self):
        ret = self.annualized_return()
        vol = self.annualized_volatility()
        return ret / vol

    def compute_drawdown_duration_peaks(self, dd: pd.Series):
        iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
        iloc = pd.Series(iloc, index=dd.index[iloc])
        df = iloc.to_frame('iloc').assign(prev=iloc.shift())
        df = df[df['iloc'] > df['prev'] + 1].astype(int)


        # If no drawdown since no trade, avoid below for pandas sake and return nan series
        if not len(df):
            return (dd.replace(0, np.nan),) * 2

        df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
        df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
        df = df.reindex(dd.index)
        return df['duration'], df['peak_dd']

    def under_water_time(self):
        df = self.data.to_frame()
        df['Return'] = df.values
        df['cummax'] = df['Return'].cummax()

        df['underwater'] = pd.to_timedelta((df['Return'] < df['cummax']).astype(int), unit=self.time_freq)
        total_time_under_water = df['underwater'].sum()

        time_under_water, max_time_under_water = 0, 0

        # loop through an Boolean series
        for val in (df['Return'] < df['cummax']).astype(int).values:
            time_under_water = time_under_water + 1 if val else 0
            max_time_under_water = max(time_under_water, max_time_under_water)

        # for display purpose
        max_time_under_water = pd.to_timedelta(max_time_under_water, unit=self.time_freq)

        return max_time_under_water, total_time_under_water

    def max_drawdown_one_month(self):
        monthly_negative_return = self.monthly_return[self.monthly_return < 0]
        return monthly_negative_return.min()

    def compute_stat(self, title=None):
        equity = self.AUM.values
        index = self.data.index
        dd = self.dd
        dd = 1 - equity / np.maximum.accumulate(self.AUM)
        # dd_dur, dd_peaks = self.compute_drawdown_duration_peaks(pd.Series(dd, index=index))
        dd_dur, dd_peaks = self.compute_drawdown_duration_peaks(dd)

        s = pd.Series(dtype=object)
        s.loc['Start'] = index[0]
        s.loc['End'] = index[-1]
        s.loc['Duration'] = s.End - s.Start

        s.loc['Return (Ann.) [%]'] = self.annualized_return() * 100
        s.loc['Volatility (Ann.) [%]'] = self.annualized_volatility() * 100
        s.loc['Information Ratio'] = self.information_ratio()

        s.loc['Final AUM [unitless]'] = equity[-1]
        s.loc['AUM Peak [$]'] = equity.max()
        s.loc['Final Return [%]'] = (equity[-1] - equity[0]) / equity[0] * 100
        max_dd = -np.nan_to_num(dd.max())
        s.loc['Max. Drawdown [%]'] = max_dd * 100
        s.loc['Max. Drawdown Duration'] = dd_dur.max()
        s.loc['Avg. Drawdown Duration'] = dd_dur.mean()

        max_time_under_water, total_time_under_water = self.under_water_time()
        s.loc['Max. Underwater Duration'] = max_time_under_water
        s.loc['Total Underwater Duration'] = total_time_under_water
        self.AUM.plot(title=title)
        plt.grid(True)
        plt.show()
        return s
```

# Dataloading

The dataset consists of 114 futures asset, indexed by HCM id from 1 - 135.

Auxiliary files include asset specs.csv which provides sector indicator information.

Date range: *End of 1970 to end of 2021*

Though the avaialble data traces back to last century, we remove data before 2015 and only backtest during 2015 - 2020 for relevance and saving computational time.



Features used include
- **Name** - Most active contract
- **Cur** - Currency
- **Year Month Day** - Timestamp
- **%change** - Unadjusted daily return
- **Vol** - Volume
- **Px** - Settlement price
- **FxBase** - 0 if USD based, otherwise 1
- **sector indicator** - Fx, Commodity, Equity and Bond


## Dataset

We loop through 114 csv files. Append Fx-ajusted return as a new column, and stack all csv files into a 3d array.

This part is already done by data_loader.py and the consoliadted table is saved as a pickle, we simply load it.

and set the appropriate time range.


```python
data_dir = root_dir + 'Data/'
df = pd.read_pickle(data_dir + 'consolidated_table')

start_date = '2016-12-10'
end_date = '2022-01-01'
data = df.loc[start_date:]
data
```





  <div id="df-5dc6bdbf-9d44-4f44-9c47-316ca286a86d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>...</th>
      <th>90</th>
      <th>91</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
      <th>101</th>
      <th>102</th>
      <th>105</th>
      <th>106</th>
      <th>107</th>
      <th>108</th>
      <th>109</th>
      <th>110</th>
      <th>111</th>
      <th>112</th>
      <th>113</th>
      <th>114</th>
      <th>115</th>
      <th>116</th>
      <th>117</th>
      <th>118</th>
      <th>119</th>
      <th>120</th>
      <th>121</th>
      <th>122</th>
      <th>123</th>
      <th>124</th>
      <th>125</th>
      <th>126</th>
      <th>127</th>
      <th>128</th>
      <th>129</th>
      <th>130</th>
      <th>131</th>
      <th>134</th>
      <th>135</th>
    </tr>
    <tr>
      <th>Date</th>
      <th>Features</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2016-12-10</th>
      <th>Name</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Cur</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Year</th>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>...</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>Month</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Day</th>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>...</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2021-12-17</th>
      <th>Vol</th>
      <td>112025</td>
      <td>93829</td>
      <td>104577</td>
      <td>272619</td>
      <td>405405</td>
      <td>46180</td>
      <td>240458</td>
      <td>85265</td>
      <td>21578</td>
      <td>90893</td>
      <td>173134</td>
      <td>1.11623e+06</td>
      <td>276424</td>
      <td>705646</td>
      <td>266118</td>
      <td>130248</td>
      <td>12595</td>
      <td>15934</td>
      <td>10481</td>
      <td>36074</td>
      <td>114498</td>
      <td>12205</td>
      <td>172053</td>
      <td>335435</td>
      <td>4582</td>
      <td>153103</td>
      <td>37621</td>
      <td>6</td>
      <td>14793</td>
      <td>18245</td>
      <td>31515</td>
      <td>38776</td>
      <td>98174</td>
      <td>31176</td>
      <td>37418</td>
      <td>30045</td>
      <td>35883</td>
      <td>166</td>
      <td>58670</td>
      <td>702</td>
      <td>...</td>
      <td>22288</td>
      <td>4559</td>
      <td>81450</td>
      <td>270782</td>
      <td>319371</td>
      <td>17482</td>
      <td>794468</td>
      <td>32082</td>
      <td>8267</td>
      <td>149</td>
      <td>229</td>
      <td>18378</td>
      <td>NaN</td>
      <td>3634</td>
      <td>NaN</td>
      <td>143970</td>
      <td>189</td>
      <td>3045</td>
      <td>36</td>
      <td>253</td>
      <td>752</td>
      <td>1094</td>
      <td>433</td>
      <td>1093</td>
      <td>382</td>
      <td>747</td>
      <td>1460</td>
      <td>472</td>
      <td>879</td>
      <td>1031</td>
      <td>364</td>
      <td>4818</td>
      <td>36</td>
      <td>NaN</td>
      <td>42</td>
      <td>9455</td>
      <td>1.82859e+06</td>
      <td>85846</td>
      <td>3997</td>
      <td>4381</td>
    </tr>
    <tr>
      <th>Px</th>
      <td>98.36</td>
      <td>98.745</td>
      <td>143.61</td>
      <td>134.21</td>
      <td>174.43</td>
      <td>215.7</td>
      <td>112.21</td>
      <td>150.2</td>
      <td>152.11</td>
      <td>109.32</td>
      <td>126.88</td>
      <td>131.172</td>
      <td>109.16</td>
      <td>121.234</td>
      <td>162.562</td>
      <td>199.594</td>
      <td>1002.7</td>
      <td>2497</td>
      <td>234.75</td>
      <td>429.5</td>
      <td>593.25</td>
      <td>107.3</td>
      <td>73.52</td>
      <td>70.72</td>
      <td>160.25</td>
      <td>1804.9</td>
      <td>221.62</td>
      <td>68600</td>
      <td>80.8</td>
      <td>136.425</td>
      <td>3.534</td>
      <td>22.533</td>
      <td>1288.5</td>
      <td>376.5</td>
      <td>53.97</td>
      <td>19.11</td>
      <td>775</td>
      <td>19717</td>
      <td>641.5</td>
      <td>3420.75</td>
      <td>...</td>
      <td>278.5</td>
      <td>729</td>
      <td>23.5739</td>
      <td>35252</td>
      <td>2167.6</td>
      <td>26482</td>
      <td>15788</td>
      <td>2256.1</td>
      <td>4626</td>
      <td>228.7</td>
      <td>1089.1</td>
      <td>2720.5</td>
      <td>34440</td>
      <td>114.7</td>
      <td>NaN</td>
      <td>108561</td>
      <td>1847.5</td>
      <td>9539</td>
      <td>615.8</td>
      <td>920.7</td>
      <td>558.1</td>
      <td>472.25</td>
      <td>1026.7</td>
      <td>1687.2</td>
      <td>750.5</td>
      <td>699.9</td>
      <td>1387.5</td>
      <td>1969.7</td>
      <td>190.7</td>
      <td>130.75</td>
      <td>131.38</td>
      <td>1.3235</td>
      <td>11.058</td>
      <td>0.15618</td>
      <td>10.977</td>
      <td>17.56</td>
      <td>76.105</td>
      <td>17019</td>
      <td>702.5</td>
      <td>46250</td>
    </tr>
    <tr>
      <th>Fx</th>
      <td>0.7125</td>
      <td>0.7125</td>
      <td>1.2889</td>
      <td>1.124</td>
      <td>1.124</td>
      <td>1.124</td>
      <td>1.124</td>
      <td>1.124</td>
      <td>113.63</td>
      <td>1181</td>
      <td>1.3245</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.2889</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>113.63</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1.124</td>
      <td>1.124</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.124</td>
      <td>1</td>
      <td>1</td>
      <td>4.2205</td>
      <td>113.63</td>
      <td>1</td>
      <td>1</td>
      <td>1.124</td>
      <td>1.124</td>
      <td>NaN</td>
      <td>5.6859</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.124</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>76.085</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>FxBase</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ret</th>
      <td>-0.00119124</td>
      <td>0.000138256</td>
      <td>0.00188319</td>
      <td>0.00128967</td>
      <td>0.00218252</td>
      <td>0.00436505</td>
      <td>0.000496028</td>
      <td>0.00496028</td>
      <td>-0.000100035</td>
      <td>0.000400999</td>
      <td>-0.00129239</td>
      <td>0.0006</td>
      <td>-0.0004</td>
      <td>-0.0003</td>
      <td>0.0027</td>
      <td>0.0082</td>
      <td>0.0170479</td>
      <td>-0.0196</td>
      <td>-0.0089</td>
      <td>-0.0022</td>
      <td>0.0034</td>
      <td>-0.0217</td>
      <td>-0.0187</td>
      <td>-0.0198</td>
      <td>-0.0143</td>
      <td>0.0037</td>
      <td>-0.0206</td>
      <td>-0.00290102</td>
      <td>0.0056</td>
      <td>-0.0046</td>
      <td>-0.0232</td>
      <td>0.0021</td>
      <td>0.008</td>
      <td>0.0212</td>
      <td>-0.0139</td>
      <td>-0.0149</td>
      <td>0.0058</td>
      <td>0.0013</td>
      <td>-0.0176</td>
      <td>-0.0118</td>
      <td>...</td>
      <td>0.00902771</td>
      <td>0.00892851</td>
      <td>0.0331</td>
      <td>-0.0151</td>
      <td>0.0087</td>
      <td>-0.00753963</td>
      <td>-0.005</td>
      <td>-0.0143</td>
      <td>0.00149531</td>
      <td>-0.000900317</td>
      <td>-0.0268</td>
      <td>-0.002</td>
      <td>-0.00519122</td>
      <td>-0.00257935</td>
      <td>NaN</td>
      <td>-0.0116029</td>
      <td>-0.0162</td>
      <td>-0.0093</td>
      <td>-0.0076</td>
      <td>-0.0134</td>
      <td>-0.0222</td>
      <td>-0.0236</td>
      <td>-0.018</td>
      <td>-0.0078</td>
      <td>-0.0177</td>
      <td>-0.0162</td>
      <td>-0.0069</td>
      <td>-0.0073</td>
      <td>0.00416664</td>
      <td>0.0038</td>
      <td>0.0028</td>
      <td>-0.003</td>
      <td>-0.0037</td>
      <td>-0.0008</td>
      <td>-0.0086</td>
      <td>-0.0014</td>
      <td>-0.00110013</td>
      <td>-0.0178</td>
      <td>0.001</td>
      <td>-0.0367</td>
    </tr>
  </tbody>
</table>
<p>20174 rows × 113 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5dc6bdbf-9d44-4f44-9c47-316ca286a86d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-5dc6bdbf-9d44-4f44-9c47-316ca286a86d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5dc6bdbf-9d44-4f44-9c47-316ca286a86d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Asset Specs

Load the sector indicator and maximum available leverage


```python
asset_specs = pd.read_csv(data_dir + "asset_spec.csv", index_col=['Asset number']).sort_index()
asset_specs
```





  <div id="df-c19ed74e-d809-44a6-ad7c-86bd7c2d7f3a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Asset name</th>
      <th>Illiquid</th>
      <th>Max lev</th>
      <th>Adjustment</th>
      <th>New</th>
      <th>Base</th>
      <th>Eur Overwrite</th>
      <th>Equity indicator</th>
      <th>Bond indicator</th>
      <th>Comm indicator</th>
      <th>FX indicator</th>
      <th>EM indicator</th>
      <th>NA indicator</th>
      <th>EU indicator</th>
      <th>Broker code</th>
    </tr>
    <tr>
      <th>Asset number</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Australia 10-year</td>
      <td>0</td>
      <td>20</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Australia 3-year</td>
      <td>0</td>
      <td>20</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Canada 10-year</td>
      <td>0</td>
      <td>20</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany BOBL</td>
      <td>0</td>
      <td>20</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany BUND</td>
      <td>0</td>
      <td>20</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>131</th>
      <td>India CNX Nifty SG</td>
      <td>0</td>
      <td>10</td>
      <td>1.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>134</th>
      <td>MSCI Taiwan</td>
      <td>1</td>
      <td>10</td>
      <td>1.2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>135</th>
      <td>BTC</td>
      <td>1</td>
      <td>1</td>
      <td>0.3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Palm Oil China</td>
      <td>1</td>
      <td>5</td>
      <td>0.3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>137</th>
      <td>FTSE Taiwan SGX</td>
      <td>0</td>
      <td>10</td>
      <td>1.2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>115 rows × 15 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c19ed74e-d809-44a6-ad7c-86bd7c2d7f3a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c19ed74e-d809-44a6-ad7c-86bd7c2d7f3a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c19ed74e-d809-44a6-ad7c-86bd7c2d7f3a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
liquid_asset = [asset for asset in data.columns if asset_specs.loc[asset, 'Illiquid']==0]
data = data[liquid_asset]
```

# Methodology

Most of the the benchmark models share the following similarities


*   Estimate volatility based on past one year standard deviation.
   * Such estimation is agnostic to the permuation of daily standard devation,
   perhapes we could apply a weight decay to the time sereis of standard deviation.
  


*   Amplify the raw Fx-adjusted return by a leverage, which will set the asset's volatily to 30% (capped by max leverage shown in asset pecs).

It is convinent to define a parent class *RiskParity* that encodes such similarities, and each benchmark model would inherit the parent class and definies its own signal generation process.


```python
class RiskParity:
  def __init__(self, raw_data, asset_specs, vol_target=0.3, N=252):
    self.ret = raw_data.xs("ret", level='Features')
    self.leverage = asset_specs['Max lev']
    self.N = N
    self.vol_target = vol_target 
    self.asset_specs = asset_specs.loc[self.ret.columns, :]
  
  def rolling_std(self):
    return self.ret.rolling(self.N, min_periods=2).std().shift(1) * np.sqrt(self.N)
  
  def compute_leverage(self):


    def _cap_leverage(column):
      return np.minimum(column, self.leverage.loc[column.name])
    
    rolling_table = self.rolling_std()
    leverage_table = self.vol_target/rolling_table
    leverage_table = leverage_table.apply(_cap_leverage) 
    return leverage_table
  
  def transform_ret(self):
    leverage_table = self.compute_leverage()
    return self.ret.multiply(leverage_table, axis=0)

  def visualize_return(self, ret):
    ax, fig = plt.subplots()
    ret = ret.dropna()
    ax = (1+ret).cumprod().plot()
    plt.title('Simple risk parity')
    plt.grid()
    plt.show()  
```

## Simple Risk Parity


```python
class SimpleRiskParity(RiskParity):
    def __init__(self, raw_data, asset_specs, vol_target=0.3, N=252):
      RiskParity.__init__(self, raw_data, asset_specs, vol_target=0.3, N=252)
      self.transformed_ret = self.transform_ret()

    def compute_return(self):
        return self.transformed_ret.mean(axis=1)


srp = SimpleRiskParity(data, asset_specs)
ret = srp.compute_return()
SimpleRiskParity = BackTest(ret).compute_stat('SimpleRiskParity')
SimpleRiskParity
```


    
![png](Benchmark_Model_Summary_files/Benchmark_Model_Summary_16_0.png)
    





    Start                        2016-12-14 00:00:00
    End                          2021-12-17 00:00:00
    Duration                      1829 days 00:00:00
    Return (Ann.) [%]                        8.33365
    Volatility (Ann.) [%]                    10.9477
    Information Ratio                       0.761227
    Final AUM [unitless]                     1.51867
    AUM Peak [$]                              1.5401
    Final Return [%]                         54.7493
    Max. Drawdown [%]                       -20.8305
    Max. Drawdown Duration         607 days 00:00:00
    Avg. Drawdown Duration         191 days 00:00:00
    Max. Underwater Duration       770 days 00:00:00
    Total Underwater Duration     1297 days 00:00:00
    dtype: object



## Asset Class Parity

### Four Sectors


```python
class AssetRiskParity(RiskParity):
    def __init__(self, raw_data, asset_specs, vol_target=0.3, N=252, 
                 sector_names =['Equity indicator', 'Bond indicator', 'Comm indicator', 'FX indicator'] ):
      RiskParity.__init__(self, raw_data, asset_specs, vol_target=0.3, N=252)
      self.transformed_ret = self.transform_ret().dropna()
      self.sector_classifier = self.asset_specs[sector_names]
    

    def compute_return(self):
      risk_parity_return = self.transformed_ret
      weights = self.sector_classifier / self.sector_classifier.sum(axis=0)
      num_sectors = self.sector_classifier.shape[1]
      weights = 1 / num_sectors * weights.sum(axis=1) 
      assert abs(weights.sum() - 1) < 1e-4, 'Sector weights overflow'

      return pd.Series(data =risk_parity_return.values @ weights.values, index=risk_parity_return.index )

acrp = AssetRiskParity(data, asset_specs)
ret = acrp.compute_return()
AssetRiskParity4Secotrs = BackTest(ret).compute_stat('AssetRiskParity4Secotrs')
AssetRiskParity4Secotrs
```

    Can't detect time frequency. Using daily frequency by default. 
    


    
![png](Benchmark_Model_Summary_files/Benchmark_Model_Summary_19_1.png)
    





    Start                        2016-12-15 00:00:00
    End                          2021-12-17 00:00:00
    Duration                      1828 days 00:00:00
    Return (Ann.) [%]                         6.0413
    Volatility (Ann.) [%]                    11.1269
    Information Ratio                       0.542945
    Final AUM [unitless]                      1.2698
    AUM Peak [$]                              1.3611
    Final Return [%]                         28.6876
    Max. Drawdown [%]                       -19.2479
    Max. Drawdown Duration         517 days 00:00:00
    Avg. Drawdown Duration         175 days 21:20:00
    Max. Underwater Duration       614 days 00:00:00
    Total Underwater Duration     1009 days 00:00:00
    dtype: object



### Three Sectors


```python
asset_specs_four_sectors = pd.read_csv('Archived/asset_spec_archived.csv').sort_values('Asset number').set_index('Asset number')
acrp = AssetRiskParity(data, asset_specs_four_sectors,sector_names=['Equity indicator','Bond indicator', 'Others indicator'])
ret = acrp.compute_return()
AssetRiskParity3Secotrs = BackTest(ret).compute_stat('AssetRiskParity3Secotrs')
AssetRiskParity3Secotrs
```

    Can't detect time frequency. Using daily frequency by default. 
    


    
![png](Benchmark_Model_Summary_files/Benchmark_Model_Summary_21_1.png)
    





    Start                        2016-12-15 00:00:00
    End                          2021-12-17 00:00:00
    Duration                      1828 days 00:00:00
    Return (Ann.) [%]                        8.33825
    Volatility (Ann.) [%]                    10.6941
    Information Ratio                       0.779703
    Final AUM [unitless]                     1.38558
    AUM Peak [$]                             1.46606
    Final Return [%]                         39.4642
    Max. Drawdown [%]                       -17.1715
    Max. Drawdown Duration         547 days 00:00:00
    Avg. Drawdown Duration         255 days 09:36:00
    Max. Underwater Duration       599 days 00:00:00
    Total Underwater Duration     1008 days 00:00:00
    dtype: object



## Volume-Weighted Risk Parity


```python
class VolumeWeightedRiskParity(RiskParity):
    def __init__(self, raw_data, asset_specs, vol_target=0.3, N=252):
      RiskParity.__init__(self, raw_data, asset_specs, vol_target=0.3, N=252)
      self.transformed_ret = self.transform_ret()
      self.volume_df = raw_data.xs("Vol", level='Features').shift(1).dropna(axis=0, how='all')


    
    def compute_return(self):
      weights_df = self.volume_df.div(self.volume_df.sum(axis=1), axis=0)
      return self.transformed_ret.multiply(weights_df, axis=0).sum(axis=1)

vwrp = VolumeWeightedRiskParity(data, asset_specs)
ret = vwrp.compute_return()
VolumeWeightedRiskParityResult = BackTest(ret).compute_stat('VolumeWeightedRiskParity')
VolumeWeightedRiskParityResult
```


    
![png](Benchmark_Model_Summary_files/Benchmark_Model_Summary_23_0.png)
    





    Start                        2016-12-10 00:00:00
    End                          2021-12-17 00:00:00
    Duration                      1833 days 00:00:00
    Return (Ann.) [%]                        8.11454
    Volatility (Ann.) [%]                    9.82617
    Information Ratio                        0.82581
    Final AUM [unitless]                     1.77245
    AUM Peak [$]                             1.85151
    Final Return [%]                         81.8369
    Max. Drawdown [%]                       -9.92218
    Max. Drawdown Duration         212 days 00:00:00
    Avg. Drawdown Duration         116 days 14:00:00
    Max. Underwater Duration       824 days 00:00:00
    Total Underwater Duration     1818 days 00:00:00
    dtype: object



## Adjusted Volume-Weighted Risk Parity


```python
class AdjustedVolumeWeightedRiskParity(RiskParity):
    def __init__(self, raw_data, asset_specs, vol_target=0.3, N=252):
      RiskParity.__init__(self, raw_data, asset_specs, vol_target=0.3, N=252)
      self.transformed_ret = self.transform_ret()
      self.volume_df = raw_data.xs("Vol", level='Features').shift(1).dropna(axis=0, how='all')
      self.matrix = raw_data 

    
    def compute_return(self):
      ret_df = self.transformed_ret 

      rolling_std_df = ret_df.shift(1).rolling(self.N, min_periods=2).apply(np.nanstd) * np.sqrt(
        self.N)
      
      rolling_lev = self.vol_target / rolling_std_df

      volume_df = self.matrix.xs("Vol", level='Features').shift(1)

      adjusted_volume = volume_df.divide(rolling_lev, axis=0)

      
      scaling_factor_df = (1 / adjusted_volume.sum(axis=1)).replace([np.inf, -np.inf], np.nan)

      weights_df = self.volume_df.multiply(scaling_factor_df, axis=0)

      return self.transformed_ret.multiply(weights_df, axis=0).sum(axis=1)

avwrp = AdjustedVolumeWeightedRiskParity(data, asset_specs)
ret = avwrp.compute_return()
AdjustedVolumeWeightedRiskParityRes = BackTest(ret).compute_stat("AdjustedVolumeWeightedRiskParityRes")
AdjustedVolumeWeightedRiskParityRes
```


    
![png](Benchmark_Model_Summary_files/Benchmark_Model_Summary_25_0.png)
    





    Start                        2016-12-10 00:00:00
    End                          2021-12-17 00:00:00
    Duration                      1833 days 00:00:00
    Return (Ann.) [%]                        7.64337
    Volatility (Ann.) [%]                    9.93503
    Information Ratio                       0.769336
    Final AUM [unitless]                     1.71656
    AUM Peak [$]                             1.80396
    Final Return [%]                         70.8814
    Max. Drawdown [%]                       -14.9724
    Max. Drawdown Duration         212 days 00:00:00
    Avg. Drawdown Duration         119 days 02:00:00
    Max. Underwater Duration       824 days 00:00:00
    Total Underwater Duration     1816 days 00:00:00
    dtype: object



## Momentum Risk Parity


```python
class MomentumRiskParity(RiskParity):
  def __init__(self, raw_data, asset_specs, vol_target=0.3, N=252, 
                 sector_names =['Equity indicator', 'Bond indicator', 'Comm indicator', 'FX indicator'] ):
      RiskParity.__init__(self, raw_data, asset_specs, vol_target=0.3, N=252)
      self.transformed_ret = self.transform_ret()
      

  def cumulative_return(self,x):
    return (1+x).prod() - 1

  def momentumQ(self, x):
    return 1 if self.cumulative_return(x) > 0 else 0 

  def momentum_rolling(self): 
    binary_table = self.transformed_ret.shift(1).rolling(self.N, min_periods=2).apply(self.momentumQ)
    return  binary_table.fillna(0) # rolling include current row
  
  def filtered_return_table(self):
    filter_table = self.momentum_rolling()
    filtered_return_table = pd.DataFrame(self.transformed_ret.values * filter_table.values, 
                                         columns=self.transformed_ret.columns, 
                                         index=self.transformed_ret.index)
    return filtered_return_table

  def compute_return(self):
    ret = self.filtered_return_table()
    return ret.mean(axis=1)

mrp = MomentumRiskParity(data, asset_specs)
ret = mrp.compute_return()
MomentumRiskParityRes = BackTest(ret).compute_stat('MomentumRiskParityRes')
MomentumRiskParityRes
```


    
![png](Benchmark_Model_Summary_files/Benchmark_Model_Summary_27_0.png)
    


## Momentum Risk Parity by Asset Class


```python
class MomentumRiskParityByAseet(MomentumRiskParity):
  def __init__(self, raw_data, asset_specs, vol_target=0.3, N=252, 
                 sector_names =['Equity indicator', 'Bond indicator', 'Comm indicator', 'FX indicator'] ):
      MomentumRiskParity.__init__(self, raw_data, asset_specs, vol_target, N)
      self.transformed_ret = self.transform_ret()
      self.sector_classifier = self.asset_specs[sector_names]

  def compute_return(self):
    # matrix multiplication of return matrix and sector indicator,
    filtered_return_table = self.filtered_return_table()
    ret = pd.DataFrame(filtered_return_table.fillna(0).values @ self.sector_classifier.values,
                       index=filtered_return_table.index,
                       columns = self.sector_classifier.columns)

    # average within asset class
    mask = self.momentum_rolling()
    count_mat = pd.DataFrame(mask.values @ self.sector_classifier.values,
                       index = mask.index,
                       columns = self.sector_classifier.columns).replace(0, np.nan) # replace 0 by nan to avoid division by zero

    ret = ret.div(count_mat)

    return ret.mean(axis=1,skipna=True)

mrp_by_asset = MomentumRiskParityByAseet(data, asset_specs)
ret = mrp_by_asset.compute_return()
MomentumRiskParityByAseetRes = BackTest(ret).compute_stat('MomentumRiskParityByAseet')
MomentumRiskParityByAseetRes
```


    
![png](Benchmark_Model_Summary_files/Benchmark_Model_Summary_29_0.png)
    





    Start                        2016-12-16 00:00:00
    End                          2021-12-17 00:00:00
    Duration                      1827 days 00:00:00
    Return (Ann.) [%]                        1.01855
    Volatility (Ann.) [%]                    9.85649
    Information Ratio                       0.103338
    Final AUM [unitless]                     1.07691
    AUM Peak [$]                             1.23004
    Final Return [%]                         6.48277
    Max. Drawdown [%]                       -31.8254
    Max. Drawdown Duration        1430 days 00:00:00
    Avg. Drawdown Duration         402 days 18:00:00
    Max. Underwater Duration       634 days 00:00:00
    Total Underwater Duration     1819 days 00:00:00
    dtype: object



## Momentum Risk Parity with Softmax Activation

At each time period, this benchmark filters asset by its momentum, then using sharpe ratio of the current lookback period, apply a softmax function to the sharpe ratio vector, yield a weight vector.


```python
from scipy.special import softmax
class MomentumRiskParityWithActivation(RiskParity):
  def __init__(self, raw_data, 
               asset_specs, 
               vol_target=0.3, 
               N=252, 
                sector_names =['Equity indicator', 'Bond indicator', 'Comm indicator', 'FX indicator'] ):
      
      RiskParity.__init__(self, raw_data, asset_specs, vol_target=0.3, N=252)
      self.transformed_ret = self.transform_ret()
  
  @staticmethod
  def filtered_return(ret):
    ret.dropna(inplace=True)
    net_ret = ret.mean() / ret.std() * np.sqrt(252) if (1+ret).prod()>1 else 0
    return  net_ret
  
  def compute_return(self):
    weights = self.transformed_ret.shift(1).rolling(self.N, min_periods=1).apply(MomentumRiskParityWithActivation.filtered_return).fillna(0)
    weights = weights.apply(softmax, axis=1)

    ret =self.transformed_ret.multiply(weights).sum(axis=1)

    return ret

MMT_activation =  MomentumRiskParityWithActivation(data, asset_specs)
ret = MMT_activation.compute_return()
MomentumRiskParityWithActivationRes = BackTest(ret).compute_stat('MomentumRiskParityWithActivation')
MomentumRiskParityWithActivationRes
```


    
![png](Benchmark_Model_Summary_files/Benchmark_Model_Summary_32_0.png)
    





    Start                                2016-12-10 00:00:00
    End                                  2021-12-17 00:00:00
    Duration                              1833 days 00:00:00
    Return (Ann.) [%]                                  8.184
    Volatility (Ann.) [%]                             11.708
    Information Ratio                               0.699011
    Final AUM [unitless]                             1.78082
    AUM Peak [$]                                     1.86804
    Final Return [%]                                 81.0108
    Max. Drawdown [%]                               -23.0035
    Max. Drawdown Duration                 577 days 00:00:00
    Avg. Drawdown Duration       221 days 10:17:08.571428572
    Max. Underwater Duration              1421 days 00:00:00
    Total Underwater Duration             1821 days 00:00:00
    dtype: object



## Volume-Weighted Momentum Risk Parity


```python
class VolumeWeightedMomentumRiskParity(RiskParity):
  def __init__(self, raw_data, asset_specs, vol_target=0.3, N=252, 
                 sector_names =['Equity indicator', 'Bond indicator', 'Comm indicator', 'FX indicator'] ):
      RiskParity.__init__(self, raw_data, asset_specs, vol_target=0.3, N=252)
      self.transformed_ret = self.transform_ret()
      self.filter_table = self.momentum_rolling()
      self.volume_df = raw_data.xs("Vol", level='Features').shift(1)

  def cumulative_return(self,x):
    return (1+x).prod() - 1

  def momentumQ(self, x):
    return 1 if self.cumulative_return(x) > 0 else 0 

  def momentum_rolling(self): 
    binary_table = self.transformed_ret.shift(1).rolling(self.N, min_periods=2).apply(self.momentumQ)
    return  binary_table.fillna(0) # rolling include current row


  def filtered_return_table(self):

    filtered_return_table = pd.DataFrame(self.transformed_ret * self.filter_table.values, 
                                         columns=self.transformed_ret.columns, 
                                         index=self.transformed_ret.index)
    return filtered_return_table


  def filtered_volume_table(self):

    filtered_volume_table = pd.DataFrame(self.volume_df.values * self.filter_table.values, 
                                         columns=self.volume_df.columns, 
                                         index=self.volume_df.index).dropna(axis=0, how='all')
    return filtered_volume_table
  def compute_return(self):
    ret = self.filtered_return_table()
    vol = self.filtered_volume_table()
    total_vol = vol.sum(axis=1)
    weights = vol.div(total_vol[total_vol != 0], axis=0)
    return ret.multiply(weights).sum(axis=1)

vmrp = VolumeWeightedMomentumRiskParity(data, asset_specs)
ret = vmrp.compute_return()
VolumeWeightedMomentumRiskParityRes = BackTest(ret).compute_stat('VolumeWeightedMomentumRiskParity')
VolumeWeightedMomentumRiskParityRes
```


    
![png](Benchmark_Model_Summary_files/Benchmark_Model_Summary_34_0.png)
    





    Start                        2016-12-10 00:00:00
    End                          2021-12-17 00:00:00
    Duration                      1833 days 00:00:00
    Return (Ann.) [%]                        10.9026
    Volatility (Ann.) [%]                    12.0831
    Information Ratio                       0.902306
    Final AUM [unitless]                     2.13645
    AUM Peak [$]                             2.16257
    Final Return [%]                          111.51
    Max. Drawdown [%]                       -10.8455
    Max. Drawdown Duration         335 days 00:00:00
    Avg. Drawdown Duration         111 days 06:00:00
    Max. Underwater Duration       637 days 00:00:00
    Total Underwater Duration     1817 days 00:00:00
    dtype: object



# Result


```python
res = pd.concat([SimpleRiskParity.rename('SimpleRP') ,
                 AssetRiskParity4Secotrs.rename('AssetRP4Secotrs'),
                 AssetRiskParity3Secotrs.rename('AssetRP3Secotrs'),
                 VolumeWeightedRiskParityResult.rename('VolWeightedRP'),
                 AdjustedVolumeWeightedRiskParityRes.rename('AdjVolWeightedRP'),
                 MomentumRiskParityRes.rename(' MomentumRP'),
                 MomentumRiskParityByAseetRes.rename('MomentumRPByAsset'),
                 MomentumRiskParityWithActivationRes.rename('MomentumRPWithActivation'),
                 VolumeWeightedMomentumRiskParityRes.rename(' VolWeightedMomentumRP')], axis=1)

```


```python
def formatter(x):
  if isinstance(x, float):
    return str(round(x, 2))
  elif isinstance(x, pd._libs.tslibs.timedeltas.Timedelta):
    return x.days
  elif isinstance(x, pd._libs.tslibs.timestamps.Timestamp):
    return x.strftime('%Y-%m-%d')

def highlight_max(s):
    if s.name in ['Start', 'End','Duration' ]:
      return ['' for _ in range(len(s))]

    is_max = s == s.max()
    color_scheme = []
    for cell in range(len(s)):
      if s.iloc[cell] == s.max():
        color_scheme.append('background: coral')
      elif s.iloc[cell] == s.min():
        color_scheme.append('background: lightgreen')
      else:
        color_scheme.append('')
    return color_scheme

d = dict(selector="th",
    props=[('text-align', 'center')])

res.applymap(formatter).style.apply(highlight_max,axis=1).set_properties(**{'width':'5em', 'text-align':'center'})\
        .set_table_styles([d])
```




<style  type="text/css" >
    #T_3c40402e_6305_11ec_8d0a_0242ac1c0002 th {
          text-align: center;
    }#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col0,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col1,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col8{
            width:  5em;
            text-align:  center;
        }#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col5,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col7,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col7{
            background:  coral;
            width:  5em;
            text-align:  center;
        }#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col6,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col3,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col4,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col8,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col2,#T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col2{
            background:  lightgreen;
            width:  5em;
            text-align:  center;
        }</style><table id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002" class="dataframe"><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >SimpleRP</th>        <th class="col_heading level0 col1" >AssetRP4Secotrs</th>        <th class="col_heading level0 col2" >AssetRP3Secotrs</th>        <th class="col_heading level0 col3" >VolWeightedRP</th>        <th class="col_heading level0 col4" >AdjVolWeightedRP</th>        <th class="col_heading level0 col5" > MomentumRP</th>        <th class="col_heading level0 col6" >MomentumRPByAsset</th>        <th class="col_heading level0 col7" >MomentumRPWithActivation</th>        <th class="col_heading level0 col8" > VolWeightedMomentumRP</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row0" class="row_heading level0 row0" >Start</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col0" class="data row0 col0" >2016-12-14</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col1" class="data row0 col1" >2016-12-15</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col2" class="data row0 col2" >2016-12-15</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col3" class="data row0 col3" >2016-12-10</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col4" class="data row0 col4" >2016-12-10</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col5" class="data row0 col5" >2016-12-14</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col6" class="data row0 col6" >2016-12-16</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col7" class="data row0 col7" >2016-12-10</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row0_col8" class="data row0 col8" >2016-12-10</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row1" class="row_heading level0 row1" >End</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col0" class="data row1 col0" >2021-12-17</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col1" class="data row1 col1" >2021-12-17</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col2" class="data row1 col2" >2021-12-17</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col3" class="data row1 col3" >2021-12-17</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col4" class="data row1 col4" >2021-12-17</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col5" class="data row1 col5" >2021-12-17</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col6" class="data row1 col6" >2021-12-17</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col7" class="data row1 col7" >2021-12-17</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row1_col8" class="data row1 col8" >2021-12-17</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row2" class="row_heading level0 row2" >Duration</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col0" class="data row2 col0" >1829</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col1" class="data row2 col1" >1828</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col2" class="data row2 col2" >1828</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col3" class="data row2 col3" >1833</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col4" class="data row2 col4" >1833</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col5" class="data row2 col5" >1829</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col6" class="data row2 col6" >1827</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col7" class="data row2 col7" >1833</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row2_col8" class="data row2 col8" >1833</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row3" class="row_heading level0 row3" >Return (Ann.) [%]</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col0" class="data row3 col0" >8.33</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col1" class="data row3 col1" >6.04</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col2" class="data row3 col2" >8.34</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col3" class="data row3 col3" >8.11</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col4" class="data row3 col4" >7.64</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col5" class="data row3 col5" >6.89</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col6" class="data row3 col6" >1.02</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col7" class="data row3 col7" >8.18</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row3_col8" class="data row3 col8" >10.9</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row4" class="row_heading level0 row4" >Volatility (Ann.) [%]</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col0" class="data row4 col0" >10.95</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col1" class="data row4 col1" >11.13</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col2" class="data row4 col2" >10.69</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col3" class="data row4 col3" >9.83</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col4" class="data row4 col4" >9.94</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col5" class="data row4 col5" >6.62</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col6" class="data row4 col6" >9.86</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col7" class="data row4 col7" >11.71</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row4_col8" class="data row4 col8" >12.08</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row5" class="row_heading level0 row5" >Information Ratio</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col0" class="data row5 col0" >0.76</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col1" class="data row5 col1" >0.54</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col2" class="data row5 col2" >0.78</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col3" class="data row5 col3" >0.83</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col4" class="data row5 col4" >0.77</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col5" class="data row5 col5" >1.04</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col6" class="data row5 col6" >0.1</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col7" class="data row5 col7" >0.7</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row5_col8" class="data row5 col8" >0.9</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row6" class="row_heading level0 row6" >Final AUM [unitless]</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col0" class="data row6 col0" >1.52</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col1" class="data row6 col1" >1.27</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col2" class="data row6 col2" >1.39</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col3" class="data row6 col3" >1.77</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col4" class="data row6 col4" >1.72</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col5" class="data row6 col5" >1.42</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col6" class="data row6 col6" >1.08</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col7" class="data row6 col7" >1.78</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row6_col8" class="data row6 col8" >2.14</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row7" class="row_heading level0 row7" >AUM Peak [$]</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col0" class="data row7 col0" >1.54</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col1" class="data row7 col1" >1.36</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col2" class="data row7 col2" >1.47</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col3" class="data row7 col3" >1.85</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col4" class="data row7 col4" >1.8</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col5" class="data row7 col5" >1.45</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col6" class="data row7 col6" >1.23</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col7" class="data row7 col7" >1.87</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row7_col8" class="data row7 col8" >2.16</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row8" class="row_heading level0 row8" >Final Return [%]</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col0" class="data row8 col0" >54.75</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col1" class="data row8 col1" >28.69</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col2" class="data row8 col2" >39.46</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col3" class="data row8 col3" >81.84</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col4" class="data row8 col4" >70.88</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col5" class="data row8 col5" >41.23</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col6" class="data row8 col6" >6.48</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col7" class="data row8 col7" >81.01</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row8_col8" class="data row8 col8" >111.51</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row9" class="row_heading level0 row9" >Max. Drawdown [%]</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col0" class="data row9 col0" >-20.83</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col1" class="data row9 col1" >-19.25</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col2" class="data row9 col2" >-17.17</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col3" class="data row9 col3" >-9.92</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col4" class="data row9 col4" >-14.97</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col5" class="data row9 col5" >-7.33</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col6" class="data row9 col6" >-31.83</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col7" class="data row9 col7" >-23.0</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row9_col8" class="data row9 col8" >-10.85</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row10" class="row_heading level0 row10" >Max. Drawdown Duration</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col0" class="data row10 col0" >607</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col1" class="data row10 col1" >517</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col2" class="data row10 col2" >547</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col3" class="data row10 col3" >212</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col4" class="data row10 col4" >212</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col5" class="data row10 col5" >577</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col6" class="data row10 col6" >1430</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col7" class="data row10 col7" >577</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row10_col8" class="data row10 col8" >335</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row11" class="row_heading level0 row11" >Avg. Drawdown Duration</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col0" class="data row11 col0" >191</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col1" class="data row11 col1" >175</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col2" class="data row11 col2" >255</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col3" class="data row11 col3" >116</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col4" class="data row11 col4" >119</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col5" class="data row11 col5" >167</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col6" class="data row11 col6" >402</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col7" class="data row11 col7" >221</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row11_col8" class="data row11 col8" >111</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row12" class="row_heading level0 row12" >Max. Underwater Duration</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col0" class="data row12 col0" >770</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col1" class="data row12 col1" >614</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col2" class="data row12 col2" >599</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col3" class="data row12 col3" >824</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col4" class="data row12 col4" >824</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col5" class="data row12 col5" >1000</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col6" class="data row12 col6" >634</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col7" class="data row12 col7" >1421</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row12_col8" class="data row12 col8" >637</td>
            </tr>
            <tr>
                        <th id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002level0_row13" class="row_heading level0 row13" >Total Underwater Duration</th>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col0" class="data row13 col0" >1297</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col1" class="data row13 col1" >1009</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col2" class="data row13 col2" >1008</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col3" class="data row13 col3" >1818</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col4" class="data row13 col4" >1816</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col5" class="data row13 col5" >1297</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col6" class="data row13 col6" >1819</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col7" class="data row13 col7" >1821</td>
                        <td id="T_3c40402e_6305_11ec_8d0a_0242ac1c0002row13_col8" class="data row13 col8" >1817</td>
            </tr>
    </tbody></table>



The maximum value in each metric is highlighted by coral.
The minimum value in each metric is hilighted by lightgreen.
