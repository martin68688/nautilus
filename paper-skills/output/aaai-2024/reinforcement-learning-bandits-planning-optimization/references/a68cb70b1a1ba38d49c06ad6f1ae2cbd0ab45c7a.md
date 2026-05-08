---
title: "Last-iterate Convergence in Regularized Graphon Mean Field Game"
source: "https://www.semanticscholar.org/paper/a68cb70b1a1ba38d49c06ad6f1ae2cbd0ab45c7a"
categories: ['reinforcement-learning-bandits-planning-optimization', 'federated-learning-optimization-and-robustness']
tags: ['mean-field-games', 'graphon', 'optimization', 'multi-agent']
venue: "AAAI 2024"
tldr: "Analyzes last-iterate convergence of regularized mirror descent in graphon mean-field games for modeling large-scale agent systems."
---

# Last-iterate Convergence in Regularized Graphon Mean Field Game

**Source**: [https://www.semanticscholar.org/paper/a68cb70b1a1ba38d49c06ad6f1ae2cbd0ab45c7a](https://www.semanticscholar.org/paper/a68cb70b1a1ba38d49c06ad6f1ae2cbd0ab45c7a)

**TLDR**: Analyzes last-iterate convergence of regularized mirror descent in graphon mean-field games for modeling large-scale agent systems.

## Abstract

To model complex real-world systems, such as traders in stock markets, or the dissemination of contagious diseases, graphon mean-field games (GMFG) have been proposed to model many agents. Despite the empirical success, our understanding of GMFG is limited. Popular algorithms such as mirror descent are deployed but remain unknown for their convergence properties. In this work, we give the first last-iterate convergence rate of mirror descent in regularized monotone GMFG. In tabular monotone GMFG with finite state and action spaces and under bandit feedback, we show a last-iterate convergence rate of O(T^{-1/4}). Moreover, when exact knowledge of costs and transitions is available, we improve this convergence rate to O(T^{-1}), matching the existing convergence rate observed in strongly convex games. In linear GMFG, our algorithm achieves a last-iterate convergence rate of O(T^{-1/5}). Finally, we verify the performance of the studied algorithms by empirically testing them against fictitious play in a variety of tasks.