# Clusterização Boston Housing com KMeans

Avaliação do 2º bimestre — clusterizador KMeans treinado na base Boston Housing.

## Arquivos

- `housing_clusterizador.py` — módulo de treinamento: trata dados faltantes, normaliza (MinMax nos atributos quantitativos e one-hot nos categóricos `CHAS` e `RAD`), determina o número ótimo de grupos pelo método do cotovelo (maior distância até a reta entre o primeiro e o último ponto da curva), treina e salva o modelo.
- `housing_inferencia.py` — módulo de inferência: recebe os dados de uma nova residência, normaliza, infere o grupo e exibe o centroide desnormalizado.
- `cotovelo.png` — gráfico do método do cotovelo.
- `*.model` — normalizador, colunas categóricas e modelo KMeans salvos.

## Como executar

```bash
pip install -r requirements.txt
python housing_clusterizador.py
python housing_inferencia.py
```

Para testar outra residência, altere o dicionário `nova_residencia` em `housing_inferencia.py`.
