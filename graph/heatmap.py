# ヒートマップに表示する列数
k = 10

cols = numcorr.nlargest(k, 'target')['target'].index
cm = np.corrcoef(a[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()