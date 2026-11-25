# Task 2 Report: Five Step Bike Sharing Demand Forecasting with Encoder Decoder Attention

**Course:** Deep Learning

**Dataset:** Capital Bikeshare hourly records, 2011–2012

**Target (`pred_column`):** `cnt`, total rentals per hour

**Implementation:** PyTorch; code and reproducible commands are in the repository root.

## 1. Problem and research objective

The objective is to predict bike rental demand at five consecutive future hours from the preceding 24 hourly observations. We use the convention that the final observed hour is $t$; the output is $\hat{y}_{t+1},\ldots,\hat{y}_{t+5}$. Equivalently, if the first unobserved hour is called the current step, the output covers that step through four later steps. No target value at or after the first forecast hour enters the model at inference.

The input is a multivariate sequence of shape $24 \times 19$. It contains past rental counts, calendar variables, weather variables, and cyclic encodings of hour, month, and weekday. The output is a five-element vector of predicted `cnt` values. The task uses an LSTM encoder, attention over the encoded history, and an autoregressive LSTM decoder.

## 2. Dataset and preprocessing

The [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) provides 17,379 hourly records from 2011–2012. Its `cnt` field is the sum of `casual` and `registered`. We exclude those two component fields from the inputs when forecasting `cnt`, while retaining historical `cnt` itself. We also drop the row identifier and date string after deriving timestamps and cyclical features. The model has 19 input features.

Rows are sorted by date and hour, then split chronologically: first 70% for training, next 15% for validation, and final 15% for testing. The validation and test windows may use the immediately preceding 24 observations as historical context, but their target labels begin within their own split. Both feature and target standardization are fitted on training data only.

The hourly file contains 75 gaps between adjacent rows. A sliding window is retained only if every adjacent timestamp throughout its 24 input hours and five target hours differs by exactly one hour. This yields **10,404 train**, **2,603 validation**, and **2,469 test** windows. Thus each retained target vector represents five consecutive hours. Overlapping windows remain separate forecast origins, so pooled predictions are not independent observations; this does not invalidate the reported point metrics but matters for confidence intervals and significance claims.

An exploratory check found 165 absent hourly timestamps across those 75 gaps, no blank cells or duplicate timestamps, and no disagreement with `cnt = casual + registered`. The median count is 142 and the 95th percentile is 563.1. In training data, mean demand is highest at 17:00 (390.9 rentals) and lowest at 04:00 (5.2). This hourly pattern motivates a 24-hour history and cyclic hour features. The chronological split means are 159.9 for training, 281.0 for validation, and 235.7 for test. These differences motivate chronological evaluation and caution against assuming the test score applies unchanged to other periods. The reproducible [EDA summary](EDA.md) records each preprocessing decision and its supporting observation.

## 3. Model architecture and training

The encoder is a two-layer LSTM with 64 hidden units per layer. It maps the 24 input vectors to 24 encoder states and passes its final hidden and cell states to the decoder. At each of five decoding steps, Bahdanau additive attention scores every encoder state against the preceding decoder hidden state. Softmax-normalized weights create a context vector. The decoder receives that context and the preceding target value; a readout network combines the decoder output, context, and preceding target to predict the next count. The first decoder input is the last observed count. At validation and test time, later decoder inputs are the model's own predictions.

The network has **138,497 trainable parameters**. Training uses MSE loss over all five outputs, AdamW with learning rate 0.001 and weight decay 0.00001, batch size 128, dropout 0.2, and gradient clipping at norm 1. Teacher forcing begins at 0.5 and declines by epoch. The learning rate is reduced when validation MSE stalls; early stopping selects the checkpoint with the lowest autoregressive validation MSE. The saved checkpoint was evaluated again on the test set to verify the reported scores.

## 4. Evaluation method

All test predictions are generated without teacher forcing, inverse transformed to rental counts, and clipped at zero. For (N) test windows and horizon (H=5), let (Y,\hat{Y}\in\mathbb{R}^{N\times H}). Each metric is calculated for every horizon across all windows; a macro result averages the five horizon-specific values; and an overall result flattens all (N\times H) entries before calculation.

For an observed vector (y_1,\ldots,y_n), predicted vector (\hat y_1,\ldots,\hat y_n), and observed mean (\bar y):

\[
\mathrm{MAE}=\frac{1}{n}\sum_i |y_i-\hat y_i|,\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i (y_i-\hat y_i)^2}.
\]

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2},\qquad
\mathrm{NSE}=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.
\]

With these standard definitions, **R² and NSE are numerically identical on the same vector**. They are both shown because the assignment requires both. The optional `R2_Pearson` in the output files is squared Pearson correlation and answers a different question: it can remain high despite bias. MAE and RMSE are expressed in rentals per hour. RMSE gives greater weight to large errors.

## 5. Test results

The following values come from the rerun after filtering incomplete time windows. The test set contains 2,469 forecast origins and 12,345 horizon-specific predictions.

| Forecast horizon | R² | NSE | MAE | RMSE |
|---|---:|---:|---:|---:|
| t+1 | 0.9408 | 0.9408 | 34.1007 | 52.6394 |
| t+2 | 0.9152 | 0.9152 | 40.7705 | 62.8955 |
| t+3 | 0.9052 | 0.9052 | 43.3751 | 66.3678 |
| t+4 | 0.8947 | 0.8947 | 45.9726 | 69.9566 |
| t+5 | 0.8781 | 0.8781 | 49.4454 | 75.2998 |
| Macro average | 0.9068 | 0.9068 | 42.7329 | 65.4318 |
| Overall pooled | **0.9069** | **0.9069** | **42.7329** | **65.8717** |

Accuracy declines as the forecast horizon increases: MAE rises from 34.10 at the first hour to 49.45 at the fifth. This is consistent with error accumulation in autoregressive decoding. The overall result indicates good fit on this chronological test period, but it does not isolate the effect of attention because no no-attention ablation was run. The prediction plot, loss curve, and mean attention map are available in `outputs/figures/`; the machine-readable values are in `outputs/metrics/test_metrics.json`.

For context, on the same complete test windows, repeating the last observed count gives MAE 166.89 and RMSE 234.95; using the same hour of the previous day gives MAE 79.99 and RMSE 132.60. The saved model's MAE 42.73 and RMSE 65.87 are lower than both. These comparisons support the value of the full forecasting pipeline, but do not identify which component produces the gain.

## 6. Reproducibility and limitations

Run `pip install -r requirements.txt`, `python eda.py`, then `python main.py` from the repository root. The default command uses `data/hour.csv`, `pred_column=cnt`, a 24-step input, and a five-step output. A full rerun with seed 42 stopped at epoch 13 and restored the epoch 5 checkpoint (validation MSE 0.1756). Its checkpoint SHA-256 and every test metric matched the saved outputs exactly. New runs save model configuration, feature order, scaler parameters, and data/checkpoint hashes in JSON beside the weights. The project also contains a separate earlier daily experiment under `outputs/day`; its metrics are not part of the hourly results above.

This experiment covers one dataset, one chronological split, and one principal architecture. Weather observations in the input are historical; the model does not use future weather forecasts. The test period is later than the training period, so year-to-year distribution changes may affect transfer to other periods. An attention ablation and repeated training with several seeds would be needed for a stronger architecture comparison.

## References

1. Fanaee-T, H., and Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*, 2, 113–127. [Dataset record](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
2. Bahdanau, D., Cho, K., and Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473).
3. Nash, J. E., and Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I: A discussion of principles. *Journal of Hydrology*, 10(3), 282–290. [DOI](https://doi.org/10.1016/0022-1694(70)90255-6).
4. scikit-learn developers. [R² score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html), [mean absolute error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html), and [root mean squared error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html) documentation.
