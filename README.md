# Projetos

# 1) Análise Exploratória de Dados sobre a Poluição Atmosférica no Estado de São Paulo e Aplicação de Modelos de Regressão para Estimar a Concentração de NO2 em São José dos Campos (SP)

<p align="center">
    <img width="400" src="https://github.com/Samirnunes/exploratory-data-analysis-and-machine-learning-projects/blob/main/air-pollution-sao-paulo-project/images/iema_logo.png" alt="Material Bread logo">
<p>
    
Notebooks da pasta `air-pollution-sao-paulo-project` e seus assuntos:

- `eda-air-pollution-sao-paulo.ipynb`: Análise Exploratória de Dados;

- `NO2-prediction-SJC.ipynb`: Machine Learning (Random Forest e XGBoost).    

Página do meu perfil no Kaggle associada: https://www.kaggle.com/code/samirnunesdasilva/eda-polui-o-do-ar-no-estado-de-s-o-paulo

Neste repositório, realiza-se uma análise exploratória (EDA - Exploratory Data Analysis) acerca dos dados de poluição atmosférica no estado de São Paulo, no Brasil, e a utilização de modelos de aprendizado de máquina para prever a concentração de NO2 em São José dos Campos (SP) com base na hora e na concentração de NO na mesma hora do dia. Os dados foram retirados da Plataforma Qualidade do Ar, do Instituto de Energia e Meio Ambiente (IEMA) - uma organização sem fins lucrativos brasileira, fundada em 2006 e com sede em São Paulo (SP) que, segundo seu site, "tem contribuído para a melhoria da qualidade ambiental de forma socialmente justa e sustentável por meio da geração e da sistematização de conhecimento técnico e científico, subsidiando a formulação e a avaliação de políticas públicas nas áreas de energia elétrica e de transportes".

Fonte dos dados: https://energiaeambiente.org.br/qualidadedoar#secao-14

## Dados e Descrições

No sentido de desenvolver uma visão geral da situação da poluição atmosférica em todo o estado de São Paulo, justificando, por exemplo, seus impactos na saúde das pessoas, buscou-se os dados mais atualizados sobre as medições das concentrações de poluentes em diversas localidades e durante um longo período de tempo (início de 2015 até o fim de 2021) de forma a se verificar a evolução desse grande problema ambiental e de saúde no estado. O resultado dessa busca foi um conjunto de dados com mais de 10 milhões de linhas englobando 87 estações de medição distintas distribuídas ao longo do estado de São Paulo que podem medir a concentração de 9 poluentes: MP10, O3, NO2, MP2.5, CO, SO2, NO, FMC e PTS. 

A análise dos dados no notebook é organizada em quatro partes essenciais. São elas:

- Data Profiling: visão e estatísticas gerais;
- Análise univariada: estudando cada coluna separadamente;
- Análise multivariada: estudando colunas em conjunto;
- Conclusões.

A seguir, encontram-se as primeiras linhas do conjunto de dados:
    
<p align="center">
    <img width="850" src="https://github.com/Samirnunes/exploratory-data-analysis-and-machine-learning-projects/blob/main/air-pollution-sao-paulo-project/images/visao_geral_dataframe.PNG" alt="Material Bread logo">
<p>

Suas colunas possuem as seguintes descrições:

- `ID`: identificação, via índice inteiro, de cada registro do dataframe;

- `Data`: data em que foi feita a medição da concentração do poluente;

- `Hora`: hora em que foi feita a medição;

- `Estação`: local em que foi feita a medição;

- `Código`: código associado à estação em que foi realizada a medição;

- `Poluente`: poluente cuja concentração foi medida;

- `Valor`: valor, na unidade especificada, da concentração do poluente;

- `Unidade`: unidade de concentração utilizada;

- `Tipo`: como foi realizada a medição (de forma automática ou manual).

## Tecnologias e Bibliotecas Utilizadas

- Jupyter Notebook
- Python
- Pandas
- SQL (SQLite)
- Matplotlib
- Scikit-Learn (modelos Random Forest e XGBoost)
 
# 2) Análise Univariada em SQL: Layoffs nas Empresas de Tecnologia (2022-2023)

<p align="center">
    <img width="300" src="https://github.com/Samirnunes/exploratory-data-analysis-and-machine-learning-projects/blob/main/sql-tech-layoffs-project/images/sqlite.jpg alt="Material Bread logo">
<p>
 
Fonte dos dados: https://www.kaggle.com/datasets/salimwid/technology-company-layoffs-20222023-data

Arquivos na pasta `sql-tech-layoffs-project`.
    
O objetivo deste projeto é utilizar os dados atuais acerca das demissões nas áreas de tecnologia de 
várias empresas para praticar o uso da linguagem SQL (Standard Query Language) para análise de dados.

Escolheu-se realizar apenas uma análise univariada das colunas do conjunto de dados de forma a simplificar
o contexto da análise e focar nas habilidades técnicas envolvendo o banco de dados SQLite e a linguagem SQL, como
preparação dos dados, criação do banco de dados e consultas a esse banco de dados utilizando a biblioteca
sqlite3 do Python em associação à biblioteca Pandas.

## Dados e Descrições

A seguir, encontram-se as primeiras linhas do conjunto de dados:

<p align="center">
    <img width="1000" src="https://github.com/Samirnunes/exploratory-data-analysis-and-machine-learning-projects/blob/main/sql-tech-layoffs-project/images/visao_geral_dataframe.PNG" alt="Material Bread logo">
<p>

Suas colunas possuem as seguintes descrições:

- `ID`: chave primária inteira;

- `company`: empresa que realizou o layoff (demissão em massa);

- `total_layoffs`: número de funcionários demitidos até janeiro de 2023;

- `impacted_workforce_percentage`: porcentagem da força de trabalho total da empresa que foi demitida pelos layoffs;

- `reported_date`: data em que o primeiro layoff ou planos de layoff da empresa foram anunciados;

- `industry`: segmentos de atuação da empresa;

- `headquarter_location`: localização da sede da empresa;

- `sources`: fonte dos dados;

- `status`: se a empresa é pública ou privada.

- `additional_notes`: notas adicionais sobre o plano de layoffs da empresa.

## Tecnologias e Bibliotecas Utilizadas

- Jupyter Notebook
- SQL (SQLite)
- Python
- Pandas
- Matplotlib
