# DeBERTa [CLS] + Handcrafted Features Synergy in XGBoost

## Finding
Concatenating DeBERTa [CLS] embedding (1024-dim) with handcrafted features (~35-dim) as XGBoost input outperforms either alone.

## Evidence
- Run4/091845: XGBoost input = hstack[stylo(26), read(4), pos(5), cls(1024)]
- XGBoost contributes ~20-35% weight in final ensemble

## Mechanism
- CLS embedding: "what is this text about?" (semantic)
- Handcrafted: "how is this text written?" (stylistic)
- XGBoost tree splits are scale-invariant → gives equal importance to all features
- Neural model concatenation would have CLS dominate handcrafted features in L2 norm

## Condition
Ensemble with tree model branch. Authorship attribution / style classification.
