---
title: "Conformal Thresholded Intervals for Efficient Regression"
source: "https://www.semanticscholar.org/paper/081cdbccad8e2baeb850ba2cac8301ccc2f81cce"
categories: ['conformal-intervals-gaussian-process-regression']
tags: ['conformal-prediction', 'regression', 'thresholded-intervals']
venue: "AAAI 2024"
tldr: "Proposes conformal thresholded intervals for efficient regression to produce small prediction sets with guaranteed coverage."
---

# Conformal Thresholded Intervals for Efficient Regression

**Source**: [https://www.semanticscholar.org/paper/081cdbccad8e2baeb850ba2cac8301ccc2f81cce](https://www.semanticscholar.org/paper/081cdbccad8e2baeb850ba2cac8301ccc2f81cce)

**TLDR**: Proposes conformal thresholded intervals for efficient regression to produce small prediction sets with guaranteed coverage.

## Abstract

This paper introduces Conformal Thresholded Intervals (CTI), a novel conformal regression method that aims to produce the smallest possible prediction set with guaranteed coverage. Unlike existing methods that rely on nested conformal frameworks and full conditional distribution estimation, CTI estimates the conditional probability density for a new response to fall into each interquantile interval using off-the-shelf multi-output quantile regression. By leveraging the inverse relationship between interval length and probability density, CTI constructs prediction sets by thresholding the estimated conditional interquantile intervals based on their length. The optimal threshold is determined using a calibration set to ensure marginal coverage, effectively balancing the trade-off between prediction set size and coverage. CTI's approach is computationally efficient and avoids the complexity of estimating the full conditional distribution. The method is theoretically grounded, with provable guarantees for marginal coverage and achieving the smallest prediction size given by Neyman-Pearson . Extensive experimental results demonstrate that CTI achieves superior performance compared to state-of-the-art conformal regression methods across various datasets, consistently producing smaller prediction sets while maintaining the desired coverage level. The proposed method offers a simple yet effective solution for reliable uncertainty quantification in regression tasks, making it an attractive choice for practitioners seeking accurate and efficient conformal prediction.