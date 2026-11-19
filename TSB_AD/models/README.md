### Extra Installation Direction

If you want to use [Chronos](https://github.com/amazon-science/chronos-forecasting), please install the following
```bash
git clone https://github.com/autogluon/autogluon
cd autogluon && pip install -e timeseries/[TimeSeriesDataFrame,TimeSeriesPredictor]
```

If you want to use [MOMENT](https://github.com/moment-timeseries-foundation-model/moment), please install the following
```bash
pip install momentfm   # only support Python 3.11 for now
```

If you want to use [TimesFM](https://github.com/google-research/timesfm), please install the following
```bash
pip install timesfm[torch]
```

If you want to use [TSPulse](https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tspulse), please install the following commands
```bash
git clone "https://github.com/ibm-granite/granite-tsfm.git" 
cd granite-tsfm
pip install ".[notebooks]"
```
For exact result reproducibility we recommend the use of Python version - 3.12.9, and the IBM granite TSFM release - v0.3.2. 
An example usage is available at [Link](https://github.com/TheDatumOrg/TSB-AD/tree/main/tutorials/TSPulse.py).

If you want to use [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama), please install the following
```bash
gluonts[torch]<=0.14.4
```
and download the checkpoint from [Link](https://github.com/time-series-foundation-models/lag-llama) and add the path to [Lag_Llama.py](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/models/Lag_Llama.py).


If you want to use [xLSTMAD](https://github.com/Nyderx/xlstmad), please install the following
```bash
lightning>=2.5.6
xlstm>=2.0.5
```