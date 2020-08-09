from sklearn.linear_model import Lasso

parameters= {'alpha':[10, 1, 0.1, 0.01, 0.001, 0.0001, 0.0003, 0.0005, 0.0007, 0.00075]}


lasso=Lasso()
lasso_reg=ms.GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)
lasso_reg.fit(x_train,y_train)

print(f'ベストパラメータ: {lasso_reg.best_params_}')


lasso_mod=Lasso(alpha=0.0007)
lasso_mod.fit(x_train,y_train)
y_lasso_train=lasso_mod.predict(x_train)
y_lasso_test=lasso_mod.predict(x_test)

print(f'RMSE : train = {str(math.sqrt(sklm.mean_squared_error(y_train, y_lasso_train)))}')
print(f'RMSE : test = {score(y_lasso_test)}')