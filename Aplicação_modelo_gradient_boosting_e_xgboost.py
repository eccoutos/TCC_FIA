# ## Modelo: GradientBoostingClassifier com Score: accuracy
modelo = GradientBoostingClassifier()

# Tunning: 
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [18, 36, 54, 72]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 1, 'max_depth': 2, 'max_features': 18, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1, 'n_estimators': 100}

# Aplicando o Modelo
model = GradientBoostingClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
print('métrica: acurácia')
print('score_treino: ' + str(round(accuracy_score(Y_train, predict_train),6) ))
print('score_teste: ' + str(round(accuracy_score(Y_test, predict_test),6) ))
print('melhores hiperparametros: ' + str(mod.best_params_))


# ## Modelo: GradientBoostingClassifier com Score: roc_auc
modelo = GradientBoostingClassifier()

# Tunning: 
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [18, 36, 54, 72]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 1, 'max_depth': 2, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 16}

# Aplicando o Modelo
model = GradientBoostingClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
print('métrica: roc_auc')
print('score_treino: ' + str(round(roc_auc_score(Y_train, predict_train),6) ))
print('score_teste: ' + str(round(roc_auc_score(Y_test, predict_test),6) ))
print('melhores hiperparametros: ' + str(mod.best_params_))


# ## Modelo: XGBClassifier com Score: accuracy
modelo = XGBClassifier()

# Tunning: 
parametros = {'nthread': [4], 'objective': ['binary:logistic'], 'learning_rate': [0.05, 0.1, 0.15], 'max_depth': [4, 8, 9, 10], 'min_child_weight': [12, 20, 30], 'silent': [1], 'subsample': [0.5, 0.6, 0.7], 'colsample_bytree': [0.6, 0.7, 0.8], 'n_estimators': [5, 10, 20, 100]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5}

# Aplicando o Modelo
model = XGBClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
print('métrica: acurácia')
print('score_treino: ' + str(round(accuracy_score(Y_train, predict_train),6) ))
print('score_teste: ' + str(round(accuracy_score(Y_test, predict_test),6) ))
print('melhores hiperparametros: ' + str(mod.best_params_))


# ## Modelo: XGBClassifier com Score: roc_auc
modelo = XGBClassifier()

# Tunning: 
parametros = {'nthread': [4], 'objective': ['binary:logistic'], 'learning_rate': [0.05, 0.1, 0.15], 'max_depth': [4, 8, 9, 10], 'min_child_weight': [12, 20, 30], 'silent': [1], 'subsample': [0.5, 0.6, 0.7], 'colsample_bytree': [0.6, 0.7, 0.8], 'n_estimators': [5, 10, 20, 100]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 10, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5}

# Aplicando o Modelo
model = XGBClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
print('métrica: roc_auc')
print('score_treino: ' + str(round(roc_auc_score(Y_train, predict_train),6) ))
print('score_teste: ' + str(round(roc_auc_score(Y_test, predict_test),6) ))
print('melhores hiperparametros: ' + str(mod.best_params_))
