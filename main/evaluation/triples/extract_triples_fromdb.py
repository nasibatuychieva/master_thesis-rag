import math
import pandas as pd

def evaluate_file(path: str, method_name: str, confidence: float = 0.95):
    df = pd.read_csv(path)

    if "correct" not in df.columns:
        raise ValueError(f"Spalte 'correct' fehlt in {path}. Bitte 0/1-Labels hinzufügen.")

    # 0/1-Labels lesen
    labels = df["correct"].dropna().astype(int)
    n = len(labels)
    if n == 0:
        raise ValueError(f"Keine validen Labels in {path} gefunden.")

    # Accuracy = Mittelwert
    mu_hat = labels.mean()

    # Varianz für Bernoulli-Zufallsvariable
    sigma_hat_sq = mu_hat * (1 - mu_hat)

    # z-Wert für 95%-Konfidenzintervall
    if confidence == 0.95:
        z = 1.96
    else:
        from scipy.stats import norm
        z = norm.ppf(0.5 + confidence / 2)

    moe = z * math.sqrt(sigma_hat_sq / n)

    ci_lower = max(0.0, mu_hat - moe)
    ci_upper = min(1.0, mu_hat + moe)

    print(f"=== Ergebnis für {method_name} ===")
    print(f"n                 = {n}")
    print(f"Accuracy (mu_hat) = {mu_hat:.3f}")
    print(f"Varianz (sigma^2) = {sigma_hat_sq:.4f}")
    print(f"MoE (95% CI)      = {moe:.3f}")
    print(f"95%-Konfidenzintervall: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print()

if __name__ == "__main__":
    # Pfade zu deinen annotierten CSVs anpassen
    evaluate_file("triples_llmgraph_annotated.csv", "LLMGraphTransformer")
    evaluate_file("triples_path_annotated.csv", "SimpleLLMPathExtractor")
    evaluate_file("triples_simplekg_annotated.csv", "SimpleKGPipeline")
