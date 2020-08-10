from sklearn.linear_model import ElasticNetCV

alphas = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.0003, 0.0005, 0.0006, 0.00055]
l1ratio = [0.1, 0.3, 0.5, 0.9, 0.95, 0.99, 1]

elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)

elasticmod = elastic_cv.fit(x_train, y_train.ravel())
ela_pred=elasticmod.predict(x_test)
print('RMSE : test = '+ str(math.sqrt(sklm.mean_squared_error(y_test, ela_pred))))
print(elastic_cv.alpha_)