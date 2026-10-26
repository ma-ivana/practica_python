import pandas as pd
import spacy
from collections import Counter
nlp = spacy.load("es_core_news_sm")
def procesar_texto(texto):
    doc = nlp(str(texto))
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop
        and not token.is_punct
        and len(token.text) > 2

    ]

    return {
        "tokens": tokens,
        "n_tokens": len(tokens),
        "sustantivos": [t.lemma_ for t in doc if t.pos_ == "NOUN" and not t.is_stop],
        "adjetivos": [t.lemma_ for t in doc if t.pos_ == "ADJ"],
        "entidades": [(t.text, t.label_) for t in doc.ents]
    }
df = pd.read_csv("practica2\\corpus.csv", encoding="utf-8")
resultados = df["texto"].apply(procesar_texto)
df["tokens"] = resultados.apply(lambda x: x["tokens"])
df["n_tokens"] = resultados.apply(lambda x: x["n_tokens"])
df["sustantivos"] = resultados.apply(lambda x: x["sustantivos"])
todas_palabras = [t for tokens in df["tokens"] for t in tokens]
frecuencias = Counter(todas_palabras).most_common(20)
print(frecuencias)