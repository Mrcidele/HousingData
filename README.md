# Clusterização Boston Housing com KMeans

Avaliação do 2º bimestre — clusterizador KMeans treinado na base Boston Housing.

## Arquivos

- `housing_clusterizador.py` — módulo de treinamento: trata dados faltantes, normaliza (MinMax nos atributos quantitativos e one-hot nos categóricos `CHAS` e `RAD`), determina o número ótimo de grupos pelo método do cotovelo (distorção = distância euclidiana média até o centroide; o k escolhido é o ponto da curva mais distante da reta entre o primeiro e o último ponto), treina e salva o modelo.
- `housing_inferencia.py` — módulo de inferência: recebe via teclado os dados de uma nova residência (Enter usa valores de exemplo), normaliza, infere o grupo e exibe o centroide desnormalizado.
- `cotovelo.png` — gráfico do método do cotovelo.
- `*.model` — normalizador, colunas categóricas e modelo KMeans salvos.

## Como executar

```bash
pip install -r requirements.txt
python housing_clusterizador.py
python housing_inferencia.py
```

Na inferência, digite os valores de cada atributo ou pressione Enter para usar o exemplo.
