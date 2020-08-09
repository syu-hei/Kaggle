import sklearn.model_selection as GridSearchCV
from sklearn.linear_model import Ridge

ridge=Ridge()
parameters= {'alpha':[x for x in range(1,101)]}

ridge_reg=ms.GridSearchCV(ridge, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)
ridge_reg.fit(x_train,y_train)
print(f"ベストパラメータ : {ridge_reg.best_params_}")
print(f"ベストスコア : {math.sqrt(-ridge_reg.best_score_)}")
ridge_pred=math.sqrt(-ridge_reg.best_score_)


ridge_mod=Ridge(alpha=17)
ridge_mod.fit(x_train,y_train)
y_pred_train=ridge_mod.predict(x_train)
y_pred_test=ridge_mod.predict(x_test)

print(f'RMSE : train = {str(math.sqrt(sklm.mean_squared_error(y_train, y_pred_train)))}')
print(f'RMSE : test = {score(y_pred_test)}')