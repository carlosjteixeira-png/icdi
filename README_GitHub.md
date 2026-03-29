# ICDI — Índice Composto de Digitalização Tributária

**Composite Index of Tax Digitalization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> Autoria: **Carlos José Teixeira**  
> Instituição: Universidade do Vale do Rio dos Sinos — UNISINOS  
> Curso: Especialização em Big Data, Data Science e Data Analytics  
> Orientador: Prof. Me. Alexandro Marian Carvalho  
> Ano: 2025

---

## Descrição

O **ICDI** é um índice composto para medir e classificar o grau de digitalização das administrações tributárias nacionais. Integra três técnicas de análise multivariada em um pipeline reprodutível:

```
ISORA Data → PCA → EWM → DEA-VRS → ICDI [0–100]
```

### Fórmula Central do ICDI

```
ICDI_i = 100 × [α × Maturidade_i + (1 − α) × Eficiência_i]
```

onde:
- `α = 0,6` (parâmetro calibrado por análise de sensibilidade)
- `Maturidade_i` = índice PCA/EWM [0,1]
- `Eficiência_i` = escore DEA-VRS [0,1]

### Equações do Modelo

| Equação | Descrição | Fórmula |
|---------|-----------|---------|
| Eq. 1 | Imputação regional ponderada | `x̂_ij = Σ(k∈Ri) x_kj·GDP_k / Σ(k∈Ri) GDP_k` |
| Eq. 2 | Normalização min-max | `x'_ij = (x_ij - min_j) / (max_j - min_j)` |
| Eq. 3 | Suavização temporal | `x̄_it = (x_i,t-1 + x_it + x_i,t+1) / 3` |
| Eq. 4 | Entropia de Shannon (EWM) | `H_j = -k · Σ p_ij · ln(p_ij)` |
| Eq. 5 | Derivação de pesos (EWM) | `w_j = (1-H_j) / Σ(1-H_j)` |
| Eq. Final | Fórmula do ICDI | `ICDI_i = 100×[α·Maturidade_i + (1-α)·Eficiência_i]` |

---

## Estrutura do Repositório

```
icdi/
├── README.md                    # Este arquivo
├── icdi_calculator.py           # Script principal com pipeline completo
├── ranking_icdi_2023.csv        # Ranking gerado com dados simulados
├── data/
│   └── isora_sample.csv         # Amostra de dados ISORA (estrutura)
├── notebooks/
│   └── icdi_notebook.ipynb      # Jupyter Notebook interativo
├── docs/
│   └── metodologia.md           # Documentação metodológica detalhada
└── LICENSE                      # Licença MIT
```

---

## Instalação e Uso

```bash
# Clone o repositório
git clone https://github.com/carlos-teixeira-icdi/icdi.git
cd icdi

# Instale as dependências
pip install numpy pandas scikit-learn scipy

# Execute com dados simulados (demonstração)
python icdi_calculator.py

# Execute com seus dados ISORA
python icdi_calculator.py --input seu_arquivo_isora.csv --output ranking.csv
```

---

## Dados de Entrada (Formato ISORA)

O script espera um CSV com as seguintes colunas:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `country_iso` | str | Código ISO do país (ex: BRA) |
| `country_name` | str | Nome da jurisdição |
| `year` | int | Ano fiscal (2018–2023) |
| `region` | str | Região geoeconômica |
| `gdp_usd_bn` | float | PIB em bilhões USD |
| `e_invoicing_pct` | float | % adoção faturamento eletrônico |
| `e_filing_pct` | float | % declarações eletrônicas |
| `predictive_analytics` | int | Uso de analytics preditivos (0/1) |
| `itms_proxy` | float | Proxy de integração de sistemas [0–1] |
| `revenue_per_employee_musd` | float | Arrecadação por funcionário (M USD) |
| `operational_performance_index` | float | Indicador sintético de desempenho |

**Fonte dos dados:** [ISORA — International Survey on Revenue Administration](https://data.imf.org)  
*Acesso gratuito em: data.imf.org (portal consolidado desde 2025)*

---

## Resultados (Amostra — Dados da Monografia 2023)

| Posição | Jurisdição | ICDI | Região |
|---------|-----------|------|--------|
| 1 | Estônia | 98,91 | Europa |
| 2 | Noruega | 97,84 | Europa |
| 3 | Dinamarca | 97,46 | Europa |
| 4 | Coreia do Sul | 96,77 | Ásia |
| 5 | Chile | 96,12 | ALC |
| 6 | México | 95,68 | ALC |
| 7 | Singapura | 95,44 | Ásia |
| 8 | Espanha | 94,91 | Europa |
| 9 | Portugal | 94,57 | Europa |
| 10 | Austrália | 94,12 | Ásia-Pacífico |
| ... | ... | ... | ... |
| **18** | **Brasil** | **84,91** | **ALC** |

*Média ALC: 78,40 | Média global: 70,40*

---

## Testes de Robustez

| Teste | Resultado | Limiar aceitável |
|-------|-----------|-----------------|
| Monte Carlo (10.000 iter.) | σ = 1,6 pontos | < 5 pontos |
| k-fold cross-validation (k=10) | MSE = 0,001 | < 0,05 |
| Análise de sensibilidade α | ±2,1 posições | < 5 posições |
| VIF pós-PCA | = 1,0 (todos) | < 5 |
| Correlação INDITEC | r = 0,91 | > 0,80 |
| Correlação ITTI (OCDE) | r = 0,94 | > 0,80 |

---

## Referências Principais

- NARDO, M. et al. *Handbook on Constructing Composite Indicators*. Paris: OECD, 2008.
- FMI. *ISORA 2024: Understanding Revenue Administration*. Washington DC, 2025.
- CHARNES, A.; COOPER, W. W.; RHODES, E. DEA. *European Journal of Operational Research*, 1978.
- SHANNON, C. E. *A Mathematical Theory of Communication*. Bell System, 1948.
- JOLLIFFE, I. T. *Principal Component Analysis*. 2. ed. Springer, 2002.

---

## Licença

MIT License — ver [LICENSE](LICENSE)

---

## Citação

```bibtex
@monography{teixeira2025icdi,
  author    = {Carlos José Teixeira},
  title     = {Índice Composto de Digitalização Tributária (ICDI): Ranking Global de Desempenho das Administrações Tributárias Nacionais com Base no ISORA},
  school    = {Universidade do Vale do Rio dos Sinos (UNISINOS)},
  year      = {2025},
  address   = {São Leopoldo, RS, Brasil},
  type      = {Trabalho de Conclusão de Curso},
  note      = {Especialização em Big Data, Data Science e Data Analytics},
  url       = {https://github.com/carlos-teixeira-icdi/icdi}
}
```
