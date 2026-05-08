---
title: "The Impact of Differential Privacy on Group Disparity Mitigation"
source: "https://aclanthology.org/2024.findings-naacl.249/"
categories: ['llm-ranking-benchmarking-adaptation', 'social-bias-mitigation-in-language-models', 'llm-fairness-bias-social-context']
tags: ['differential-privacy', 'fairness', 'group-disparity', 'privacy-utility-tradeoff']
venue: "NAACL 2024"
tldr: "Analyzes the impact of differential privacy on mitigating group performance disparities, extending beyond computer vision to NLP."
---

# The Impact of Differential Privacy on Group Disparity Mitigation

**Source**: [https://aclanthology.org/2024.findings-naacl.249/](https://aclanthology.org/2024.findings-naacl.249/)

**TLDR**: Analyzes the impact of differential privacy on mitigating group performance disparities, extending beyond computer vision to NLP.

## Abstract

AbstractThe performance cost of differential privacy has, for some applications, been shown to be higher for minority groups; fairness, conversely, has been shown to disproportionally compromise the privacy of members of such groups. Most work in this area has been restricted to computer vision and risk assessment. In response, we evaluate the impact of differential privacy on fairness across four diverse tasks, focusing on how attempts to mitigate privacy violations and between-group performance differences interact: Does privacy inhibit attempts to ensure fairness? To this end, we train (𝜀,𝛿)-differentially private models with empirical risk minimization and group distributionally robust training objectives. Consistent with previous findings, we find that differential privacy increases between-group performance differences in the baseline setting; more interestingly, differential privacy reduces between-group performance differences in the robust setting. We explain this by interpreting differential privacy as regularization.