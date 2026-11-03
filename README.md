# 開発メモ

## data
- jquantsからapiでデータ取るか、bulkで取るか
  - adjがapiしかないので、一旦apiで良いとする
- synthetic -> 後回し
  - gaussianのみ用意
- plot_candlestickは欲しい

- 金利とかコモディティとかのデータと合わせたい

## features
- とりあえずmacdとそのplot

## factors
- 特徴量として後でやる

## calibration
- 一旦無視

## portfolio, backtest
- 後回しで良い

## alpha
- lightgbm_screeningだけやれば良い

# なので、
- 1銘柄でmacdをだすようにする
  - APIでok
  - plotできる
- 全銘柄でmacdのゴールデンクロスを抽出できるようにする
- lightgbm_screeningできるようにする
- データ、feature増やす
  - 金利、コモディティ, ...
  - RSI、CAPM, ブルベア, ...
- portfolio, backtest
