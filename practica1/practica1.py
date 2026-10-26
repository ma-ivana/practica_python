import pandas as pd
import spacy
nlp = spacy.load("es_core_news_sm")
df = pd.read_csv("practica1\\corpus.csv")
#print(df)
df["tokens"] = df["genero"].apply(lambda t: [token.lemma_ for token in nlp(t)])
df["n_tokens"] = df["tokens"].apply(len)
print(df[["genero", "n_tokens"]].head())