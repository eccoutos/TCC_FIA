# Dividindo a base entre variaveis explicativas e variavel resposta
X = df2.loc[:, df2.columns !='result']
Y = df2.result

# Separando a Base entre Teste e Treino
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=7)
