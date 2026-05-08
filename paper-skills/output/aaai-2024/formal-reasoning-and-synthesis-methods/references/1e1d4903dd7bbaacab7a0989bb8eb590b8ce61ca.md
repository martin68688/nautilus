---
title: "Foundations of Reactive Synthesis for Declarative Process Specifications"
source: "https://www.semanticscholar.org/paper/1e1d4903dd7bbaacab7a0989bb8eb590b8ce61ca"
categories: ['formal-reasoning-and-synthesis-methods']
tags: ['reactive-synthesis', 'ltlf', 'formal-methods', 'controller-synthesis']
venue: "AAAI 2024"
tldr: "Studies the foundations of reactive synthesis for declarative process specifications given in Linear-time Temporal Logic over finite traces (LTLf)."
---

# Foundations of Reactive Synthesis for Declarative Process Specifications

**Source**: [https://www.semanticscholar.org/paper/1e1d4903dd7bbaacab7a0989bb8eb590b8ce61ca](https://www.semanticscholar.org/paper/1e1d4903dd7bbaacab7a0989bb8eb590b8ce61ca)

**TLDR**: Studies the foundations of reactive synthesis for declarative process specifications given in Linear-time Temporal Logic over finite traces (LTLf).

## Abstract

Given a specification of Linear-time Temporal Logic interpreted over finite traces (LTLf), the reactive synthesis problem asks to find a finitely-representable, terminating controller that reacts to the uncontrollable actions of an environment in order to enforce a desired system specification. In this paper we study, for the first time, the foundations of reactive synthesis for DECLARE, a well-established declarative, pattern-based business process modelling language grounded in LTLf. We provide a threefold contribution. First, we define a reactive synthesis problem for DECLARE. Second, we show how an arbitrary DECLARE specification can be polynomially encoded into an equivalent pure-past one in LTLf, and exploit this to define an EXPTIME algorithm for DECLARE synthesis. Third, we derive a symbolic version of this algorithm, by introducing a novel translation of pure-past temporal formulas into symbolic deterministic finite automata.