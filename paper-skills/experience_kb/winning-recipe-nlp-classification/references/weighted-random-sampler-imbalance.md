# WeightedRandomSampler for Class Imbalance

## Finding
WeightedRandomSampler with inverse class frequency weights handles imbalance more stably than shuffle=True.

## Evidence
- Run 091845 top4: uses WeightedRandomSampler
- Earlier runs: use shuffle=True
- WRS ensures minority classes sampled proportionally every epoch

## Mechanism
- weight_i = 1.0 / count_i for each class
- sample_weight = class_weights[y_train] per sample
- WeightedRandomSampler(num_samples=len(samples), replacement=True)
- replacement=True: some samples appear multiple times, balance maintained

## Condition
Classification with class imbalance ratio > 1.5:1.
