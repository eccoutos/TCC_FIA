# ## Modelo: DecisionTreeClassifier com Score: Acurácia
modelo = DecisionTreeClassifier()

# Tunning: 
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [18, 36, 54, 72]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Aplicando o Modelo
model = DecisionTreeClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
print('métrica: acurácia')
print('score_treino: ' + str(round(accuracy_score(Y_train, predict_train),6) ))
print('score_teste: ' + str(round(accuracy_score(Y_test, predict_test),6) ))
print('melhores hiperparametros: ' + str(mod.best_params_))



# ## Modelo: DecisionTreeClassifier com Score: roc_auc
modelo = DecisionTreeClassifier()

# Tunning: 
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [18, 36, 54, 72]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Aplicando o Modelo
model = DecisionTreeClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
print('métrica: acurácia')
print('score_treino: ' + str(round(roc_auc_score(Y_train, predict_train),6) ))
print('score_teste: ' + str(round(roc_auc_score(Y_test, predict_test),6) ))
print('melhores hiperparametros: ' + str(mod.best_params_))


# Visualizando a árvore de decisão:
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                class_names=True,
                feature_names = X_train.columns.tolist())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png()) 
