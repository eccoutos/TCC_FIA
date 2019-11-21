# Atribuindo parametros de entrada
target = 'result'

# Importando os dados de entrada
df = pd.read_csv('C:/Users/elen/Desktop/FIA/TCC/Rim/chronic_kidney_disease_full.csv', sep = ',', decimal = '.')

# Adicionando coluna de Target
df2 = pd.concat([df['result']])

# Criando Dataframe auxiliar
df_temp = pd.DataFrame()

# Adicionando colunas continuas com nulos
df_temp = df.age.fillna( df.age.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.bgr.fillna( df.bgr.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.bu.fillna( df.bu.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.sc.fillna( df.sc.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.sod.fillna( df.sod.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.pot.fillna( df.pot.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.hemo.fillna( df.hemo.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.pcv.fillna( df.pcv.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.wbcc.fillna( df.wbcc.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.rbcc.fillna( df.rbcc.mean() )
df2 = pd.concat([df_temp, df2], axis=1)

# Adicionando e preparando colunas com dummies com tratamento de NULL
df_temp = df.bp.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='bp' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.sg.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='sg' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.al.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='al' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.su.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='su' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.rbc.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='rbc' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.pc.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='pc' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.pcc.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='pcc' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.ba.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='ba' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.htn.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='htn' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.dm.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='dm' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.cad.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='cad' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.appet.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='appet' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.pe.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='pe' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.ane.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='ane' )
df2 = pd.concat([df_temp, df2], axis=1)
 
