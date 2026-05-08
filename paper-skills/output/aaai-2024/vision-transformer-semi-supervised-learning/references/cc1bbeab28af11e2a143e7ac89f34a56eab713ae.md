---
title: "DrFuse: Learning Disentangled Representation for Clinical Multi-Modal Fusion with Missing Modality and Modal Inconsistency"
source: "https://www.semanticscholar.org/paper/cc1bbeab28af11e2a143e7ac89f34a56eab713ae"
categories: ['vision-transformer-semi-supervised-learning', 'health-ai-methods-applications']
tags: ['clinical-ai', 'multimodal-fusion', 'missing-data', 'disentangled-representation']
venue: "AAAI 2024"
tldr: "Proposes a method for clinical multi-modal fusion that learns disentangled representations robust to missing modality and inconsistency."
---

# DrFuse: Learning Disentangled Representation for Clinical Multi-Modal Fusion with Missing Modality and Modal Inconsistency

**Source**: [https://www.semanticscholar.org/paper/cc1bbeab28af11e2a143e7ac89f34a56eab713ae](https://www.semanticscholar.org/paper/cc1bbeab28af11e2a143e7ac89f34a56eab713ae)

**TLDR**: Proposes a method for clinical multi-modal fusion that learns disentangled representations robust to missing modality and inconsistency.

## Abstract

The combination of electronic health records (EHR) and medical images is crucial for clinicians in making diagnoses and forecasting prognoses. Strategically fusing these two data modalities has great potential to improve the accuracy of machine learning models in clinical prediction tasks. However, the asynchronous and complementary nature of EHR and medical images presents unique challenges. Missing modalities due to clinical and administrative factors are inevitable in practice, and the significance of each data modality varies depending on the patient and the prediction target, resulting in inconsistent predictions and suboptimal model performance. To address these challenges, we propose DrFuse to achieve effective clinical multi-modal fusion. It tackles the missing modality issue by disentangling the features shared across modalities and those unique within each modality. Furthermore, we address the modal inconsistency issue via a disease-wise attention layer that produces the patient- and disease-wise weighting for each modality to make the final prediction. We validate the proposed method using real-world large-scale datasets, MIMIC-IV and MIMIC-CXR. Experimental results show that the proposed method significantly outperforms the state-of-the-art models.