# Tech Challenge 5 - Datathon Passos Mágicos

## 📋 Sobre o Projeto

Análise de dados educacionais da **Associação Passos Mágicos**, organização com 32 anos de atuação na transformação da vida de crianças e jovens de baixa renda através da educação.

Este projeto utiliza dados de 2022, 2023 e 2024 para responder perguntas estratégicas sobre o desempenho educacional dos alunos e desenvolver um modelo preditivo de risco de defasagem.

## 🎯 Objetivos

- Análise exploratória dos indicadores educacionais (IAN, IDA, IEG, IAA, IPS, IPP, IPV, INDE)
- Identificação de padrões de risco e defasagem
- Desenvolvimento de modelo preditivo usando Machine Learning
- Storytelling com os dados para insights acionáveis

## 📁 Estrutura do Projeto

```
tech5/
├── BASES/                          # Bases de dados
│   ├── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
│   ├── Base Tratada.xlsx
│   └── dicionario.xlsx
├── GRÁFICOS/                       # Gráficos gerados nas análises
│   ├── alunos-atingem-ponto-de-virada-por-fase.png
│   ├── comparativo-desempenho-por-materia.png
│   ├── correlacao-engajamento-vs-desempenho-academico.png
│   ├── distribuicao_idade_alunos.png
│   └── ... (outros gráficos)
├── NOTEBOOKS/                      # Notebooks Jupyter
│   ├── analise_exploratoria.ipynb
│   ├── tratamento_base.ipynb
│   └── modelo_preditivo_risco.ipynb
├── instalar.bat                    # Script de instalação (Windows)
├── README.md                       # Documentação do projeto
├── storytelling.pdf                # Apresentação dos insights
└── streamlit_app.py                # Aplicação web interativa
```

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### 1. Instalar Dependências

**Opção A - Script automático (Windows):**
```bash
instalar.bat
```

**Opção B - Instalação manual:**
```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
```

**Para usar a aplicação Streamlit, instale também:**
```bash
python -m pip install streamlit reportlab
```

### 2. Executar Notebooks

```bash
jupyter notebook
```

Navegue até a pasta `NOTEBOOKS/` e abra:
- `tratamento_base.ipynb` - Tratamento e limpeza dos dados
- `analise_exploratoria.ipynb` - Análise exploratória dos dados
- `modelo_preditivo_risco.ipynb` - Modelo preditivo de risco

### 3. Executar Aplicação Web (Opcional)

```bash
streamlit run streamlit_app.py
```

A aplicação permite calcular índices educacionais de forma interativa para pessoas leigas.

## 📊 Principais Análises

### Indicadores Analisados

- **IAN** - Índice de Adequação do Nível
- **IDA** - Índice de Desempenho Acadêmico
- **IEG** - Índice de Engajamento
- **IAA** - Índice de Autoavaliação
- **IPS** - Índice Psicossocial
- **IPP** - Índice Psicopedagógico
- **IPV** - Índice de Ponto de Virada
- **INDE** - Índice de Desenvolvimento Educacional

### Modelo Preditivo

O modelo identifica alunos em risco de defasagem usando:
- Random Forest Classifier
- Gradient Boosting Classifier
- Features: indicadores educacionais, notas, engajamento, dados demográficos

**Resultados:**
- Classificação de risco: Baixo, Médio, Alto
- Probabilidade de risco para cada aluno
- Identificação dos indicadores mais importantes

## 🛠️ Tecnologias

- Python 3.8+
- Pandas - Manipulação de dados
- NumPy - Operações numéricas
- Matplotlib/Seaborn - Visualização
- Scikit-learn - Machine Learning
- Jupyter Notebook - Ambiente de desenvolvimento
- Streamlit - Aplicação web interativa
- ReportLab - Geração de relatórios PDF
- OpenPyXL - Leitura de arquivos Excel

## 📈 Principais Descobertas

- **29.41%** dos alunos estão em risco de defasagem
- Indicadores de **engajamento** e **aprendizado** são os maiores preditores de risco
- Alunos em risco apresentam:
  - 21.5% menor engajamento
  - 33.6% menor índice de aprendizado
  - 34.9% menor média em matemática

## 👥 Equipe

Airton da Silva Cruz Filho - RM 362447 
Gustavo Pitarello de Souza - RM 361594 
João Paulo Giacherini de Moraes - RM 361571
Victor Moreno Galves Marcondes - RM 362219
Thiago Ribeiro da Costa - RM 362845


## 📝 Licença

Este projeto é parte do programa educacional da FIAP.

---


