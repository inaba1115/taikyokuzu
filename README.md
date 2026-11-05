# 開発メモ

## data
- jquantsからapiでデータ取るか、bulkで取るか
  - adjがapiしかないので、一旦apiで良いとする
- synthetic -> 後回し
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

# やること
- 1銘柄でmacdを出すようにする -> now
  - APIでok
  - plotもやる

- 全銘柄でmacdのゴールデンクロスを抽出できるようにする
  - macdでのスクリーニング

- lightgbm_screeningできるようにする

- データ、feature増やす -> ここに時間かけるでok
  - 金利、コモディティ, ...
  - RSI、CAPM, ブルベア, ...
  - dfデータの作り方を工夫する必要あると思われる

- portfolio, backtest
