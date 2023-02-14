"""
diabetes datasetを使って糖尿病データ(diabetes dataset)を分析する
age: 年齢
bmi: ボディマス指数
bp: 平均血圧
s1 tc: T細胞（白血球の一種）
"""

load_diabetes()
X, y = datasets.load_diabetes(return_X_y=True)
print(X.shape)
print(X[0])

#numpyのnewaxis関数を使って新しい配列を作成
X = X[:, np.newaxis, 2]

#scikit - データを学習用・テスト用に分類
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)

#モデルを線形回帰で学習
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

#テストデータに対する予測値を算出
y_pred = model.predict(X_test)

#データをプロットして表示
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()


"""
pandasを使ってかぼちゃデータを分析
"""

import pandas as pd
pumpkins = pd.read_csv('../data/US-pumpkins.csv')
pumpkins.head()

#データフレームに欠損データがあるかをチェック
pumpkins.isnull().sum()

#データフレームを扱いやすくするために、drop() 関数を使っていくつかの列を削除
#5つのデータを完全に含むデータ行のみを抽出する
new_columns = ['Package', 'Month', 'Low Price', 'High Price', 'Date']
pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)

#かぼちゃの平均価格を求める
price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2
#日付列の月のみを抽出したデータフレームを作成する
month = pd.DatetimeIndex(pumpkins['Date']).month

#変換したデータをPandasの新しいデータフレームにコピーする。
new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})

#かぼちゃ1個単位で売られているもの、1ポンド単位で売られているもの等、単位が統一されていないので
#「bushel」という文字列を持つカボチャだけを選択してフィルタリングデータフレームに加える
pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]

#1ブッシェルあたりの価格を表示するためには、計算して価格を標準化する
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
