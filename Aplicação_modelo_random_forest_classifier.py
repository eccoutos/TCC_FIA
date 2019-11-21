# ## Modelo: RandomForestClassifier com Score: Acurácia
modelo = RandomForestClassifier()

# Tunning: 
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [18, 36, 54, 72]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': True, 'max_depth': 8, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}

# Aplicando o Modelo
model = RandomForestClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
print('métrica: acurácia')
print('score_treino: ' + str(round(accuracy_score(Y_train, predict_train),6) ))
print('score_teste: ' + str(round(accuracy_score(Y_test, predict_test),6) ))
print('melhores hiperparametros: ' + str(mod.best_params_))



# ## Modelo: RandomForestClassifier com Score: roc_auc
modelo = RandomForestClassifier()

# Tunning: 
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [18, 36, 54, 72]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': False, 'max_depth': 8, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 10}

# Aplicando o Modelo
model = RandomForestClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
print('métrica: roc_auc')
print('score_treino: ' + str(round(roc_auc_score(Y_train, predict_train),6) ))
print('score_teste: ' + str(round(roc_auc_score(Y_test, predict_test),6) ))
print('melhores hiperparametros: ' + str(mod.best_params_))
