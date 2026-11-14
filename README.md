# project
- データ取得、可視化
  - data: jquantsや金利データなどを取得する
  - plotting: 各種可視化
- 機械学習
  - features
  - targets
  - ; 学習/予測、モデル選択、Walk-Forward Validationはライブラリ使う
- バックテスト
  - backtest
- システム化
  - trading: 実市場への接続、発注、ポジション管理
- 理論
  - portfolio: ポートフォリオ理論
  - factors: ファクターモデル
  - pricing: プライシング理論
  - risk: リスク管理
  - ; featuresに入れれるのはそっちで良い

# design
- アーキ
  - MACDゴールデンクロス
  - forward_returns(1)
  - リスク管理(ポートフォリオ)
  - トレーリングストップ
- memo
  - forward_returns(5)とかでも良い。バックテストでパラメタ決めて良い
  - 始値からの値上がり率で考えた方が良いかも
