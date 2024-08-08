import pandas as pd
import matplotlib.pyplot as plt

# 1. Leitura dos Dados
# Supondo que os dados estejam em um arquivo CSV chamado 'dados_saude.csv'
df = pd.read_csv('dados_saude.csv')

# Visualizando as primeiras linhas do DataFrame
print(df.head())

# 2. Limpeza dos Dados
# Remover duplicatas
df = df.drop_duplicates()

# Verificar valores ausentes
print(df.isnull().sum())

# Preencher valores ausentes ou removê-los, dependendo do caso
df['idade'] = df['idade'].fillna(df['idade'].median())
df.dropna(subset=['pressao_arterial', 'glicose'], inplace=True)

# 3. Análise Exploratória de Dados (EDA)
# Estatísticas Descritivas
print(df.describe())

# Distribuição de Idade
plt.hist(df['idade'], bins=10, color='blue', alpha=0.7)
plt.title('Distribuição de Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

# Analisando a Correlação entre Variáveis
correlation_matrix = df.corr()
print(correlation_matrix)

# Plotando a Correlação entre Pressão Arterial e Glicose
plt.scatter(df['pressao_arterial'], df['glicose'], color='red')
plt.title('Correlação entre Pressão Arterial e Nível de Glicose')
plt.xlabel('Pressão Arterial')
plt.ylabel('Nível de Glicose')
plt.show()

# 4. Identificação de Pacientes de Alto Risco
# Definindo critérios de alto risco (ex: Pressão Arterial > 140 e Glicose > 126)
alto_risco = df[(df['pressao_arterial'] > 140) & (df['glicose'] > 126)]
print(f"Número de pacientes de alto risco: {alto_risco.shape[0]}")

# 5. Relatórios Automatizados
# Gerar um relatório resumido dos pacientes de alto risco
alto_risco_resumo = alto_risco.groupby(['idade', 'genero']).agg({
    'pressao_arterial': 'mean',
    'glicose': 'mean',
    'peso': 'mean'
}).reset_index()

print("Resumo de Pacientes de Alto Risco")
print(alto_risco_resumo)

# Salvando o relatório em um arquivo CSV
alto_risco_resumo.to_csv('relatorio_alto_risco.csv', index=False)

# 6. Resultados Adicionais
# Análise por Faixa Etária
faixa_etaria = pd.cut(df['idade'], bins=[0, 30, 50, 70, 100], labels=['0-30', '31-50', '51-70', '71-100'])
df['faixa_etaria'] = faixa_etaria

# Média de Pressão Arterial por Faixa Etária
media_pressao_faixa = df.groupby('faixa_etaria')['pressao_arterial'].mean()
print("Média de Pressão Arterial por Faixa Etária")
print(media_pressao_faixa)

# Média de Nível de Glicose por Faixa Etária
media_glicose_faixa = df.groupby('faixa_etaria')['glicose'].mean()
print("Média de Nível de Glicose por Faixa Etária")
print(media_glicose_faixa)

# Plotando as médias de pressão arterial e glicose por faixa etária
media_pressao_faixa.plot(kind='bar', color='green', alpha=0.6)
plt.title('Média de Pressão Arterial por Faixa Etária')
plt.xlabel('Faixa Etária')
plt.ylabel('Média de Pressão Arterial')
plt.show()

media_glicose_faixa.plot(kind='bar', color='orange', alpha=0.6)
plt.title('Média de Nível de Glicose por Faixa Etária')
plt.xlabel('Faixa Etária')
plt.ylabel('Média de Nível de Glicose')
plt.show()

# 7. Conclusão
print("Análise concluída. Relatório de pacientes de alto risco salvo como 'relatorio_alto_risco.csv'.")
