# energy-efficiency-pca-neural-network
# Previsione del fabbisogno di riscaldamento degli edifici (PCA + Rete Neurale)

Progetto realizzato per l'esame di Intelligenza Artificiale Applicata
(Laurea Triennale in Ingegneria Informatica, Università del Salento).

## Obiettivo
Prevedere il carico di riscaldamento di un edificio a partire dalle sue
caratteristiche architettoniche (dataset UCI "Energy Efficiency", 768 edifici,
8 feature), e valutare l'effetto della PCA come fase di pre-processing di una
rete neurale.

## Approccio
- Analisi esplorativa e studio delle correlazioni tra feature e target
- Standardizzazione delle feature (fit solo sul training set)
- PCA: 5 componenti principali per conservare il 95% della varianza
- Rete MLP profonda in PyTorch: inizializzazione He, Batch Normalization,
  Dropout, ottimizzatore Adam, learning rate scheduling esponenziale
- Confronto tra la stessa rete addestrata sulle 8 feature originali e
  sulle 5 componenti PCA

## Risultati (test set)
| Modello | RMSE | MAE | R² | Tempo |
|---|---|---|---|---|
| MLP senza PCA (8 feature) | 2,25 | 1,78 | 0,951 | 5,7 s |
| MLP con PCA (5 componenti) | 2,58 | 1,92 | 0,936 | 4,2 s |

La PCA riduce il tempo di addestramento di circa il 25% al costo di una
piccola perdita di accuratezza: con sole 8 feature originali il vantaggio
della riduzione di dimensionalità è limitato, mentre diventa rilevante su
dataset con molte più variabili.

## Tecnologie
Python, PyTorch, scikit-learn, pandas, NumPy, Matplotlib, Jupyter

## Esecuzione
pip install torch scikit-learn pandas numpy matplotlib jupyter
Posizionare `ENB2012_data.csv` nella stessa cartella del notebook ed eseguirlo.
