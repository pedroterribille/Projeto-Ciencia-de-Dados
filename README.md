# Projeto em Ciência de Dados — Predição do *Overall* no FIFA 23

**Pontifícia Universidade Católica do Paraná (PUCPR)**  
**Especialização em Inteligência Artificial e Ciência de Dados — Turma U (2026)**  
**Professor:** Prof. Dr. Jean Paul Barddal  

## Equipe 5
- Arthur Correia Brígido
- Leandro Dzoney Hasselmann
- Leonardo Bora da Costa
- Pedro Henrique Terribille

---

## Visão Geral
Modelagem supervisionada de **regressão** para estimar o **`overall`** (`0–99`) de jogadores de futebol a partir exclusivamente de seus atributos técnicos detalhados, características físicas e posição tática primária (`male_players_fifa23.csv`, `18.127` jogadores únicos × `110` colunas originais, licença CC0).

### Principais Resultados (*Holdout* isolado — 3.626 jogadores inéditos)
| Modelo | MAE | RMSE | R² | Acerto exato ou ±1 pt na carta |
|---|---:|---:|---:|---:|
| Baseline Ingênuo (`DummyRegressor`) | 5,35 | 6,85 | -0,001 | 19,3% |
| Baseline Linear (`Ridge` Global) | 1,73 | 2,21 | 0,896 | 52,7% |
| Linear Segmentado (`Ridge` por Posição) | 0,98 | 1,32 | 0,963 | 77,8% |
| **Modelo Campeão (`XGBoost` Tuned)** | **~0,63** | **~0,84** | **0,985** | **~93%** |

---

## Estrutura do Pacote de Entrega
- `Projeto_Ciencia_de_Dados_Equipe5.ipynb` — Notebook principal executado do zero (*Kernel → Restart & Run All*), numerado `1..24`, contendo todas as seções exigidas no descritivo oficial:
  1. Descrição Estatística do Conjunto de Dados
  2. Análise Univariada, Assimetria e Diagnóstico de *Outliers*
  3. Análise Multivariada (incluindo auditoria de *Data Leakage* e Multicolinearidade via VIF)
  4. Visualizações de Dados Eficazes
  5. Seleção, Engenharia e Pré-processamento de Dados (re-escala, padronização e normalização)
  6. Aprendizado de Máquina e Ajuste de Hiperparâmetros (`RandomizedSearchCV` com `5-fold GroupKFold`)
  7. Validação Final, KPIs de Negócio, Análise de Resíduos e Interpretabilidade (*Permutation Importance*)
  8. Conclusão, Declaração CONSUN PUCPR 274/2024 e Empacotamento
- `Projeto_Ciencia_de_Dados_Equipe5.py` — Script Python puro exportado a partir do notebook (executável diretamente via CLI).
- `male_players_fifa23.csv` — Base de dados consolidada da edição FIFA 23 (`18.127` atletas, CC0).
- `requirements.txt` — Dependências Python para reprodução exata.

## Como Reproduzir do Zero
```bash
pip install -r requirements.txt
jupyter notebook Projeto_Ciencia_de_Dados_Equipe5.ipynb
# ou via linha de comando:
python Projeto_Ciencia_de_Dados_Equipe5.py
```
