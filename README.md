# Análise Exploratória de Dados: Poluição Atmosférica no Estado de São Paulo

<p align="center">
    <img width="400" src="https://github.com/Samirnunes/sp_polution_eda/blob/main/images/iema_logo.png" alt="Material Bread logo">
<p>

Neste repositório, realiza-se uma análise exploratória (EDA - Exploratory Data Analysis) acerca dos dados de poluição atmosférica no estado de São Paulo, no Brasil. Os dados foram retirados da Plataforma Qualidade do Ar, do Instituto de Energia e Meio Ambiente (IEMA) - uma organização sem fins lucrativos brasileira, fundada em 2006 e com sede em São Paulo (SP) que, segundo seu site, "tem contribuído para a melhoria da qualidade ambiental de forma socialmente justa e sustentável por meio da geração e da sistematização de conhecimento técnico e científico, subsidiando a formulação e a avaliação de políticas públicas nas áreas de energia elétrica e de transportes".

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
    <img width="850" src="https://github.com/Samirnunes/sp_polution_eda/blob/main/images/visao_geral_dataframe.PNG" alt="Material Bread logo">
<p>

Suas colunas possuem as seguintes descrições:

- ID: identificação, via índice inteiro, de cada registro do dataframe;

- Data: data em que foi feita a medição da concentração do poluente;

- Hora: hora em que foi feita a medição;

- Estação: local em que foi feita a medição;

- Código: código associado à estação em que foi realizada a medição;

- Poluente: poluente cuja concentração foi medida;

- Valor: valor, na unidade especificada, da concentração do poluente;

- Unidade: unidade de concentração utilizada;

- Tipo: como foi realizada a medição (de forma automática ou manual).

## Tecnologias e Bibliotecas Utilizadas

- Jupyter Notebook
- Python
- Pandas
- SQL (SQLite)
- Matplotlib
 
