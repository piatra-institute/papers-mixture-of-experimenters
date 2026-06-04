"""Mixture of Experimenters — test-time residual-stream experimentation toy.

A small trained transformer on a synthetic in-context task, plus a toolkit for
forking, perturbing, scoring, and selecting residual-stream interventions at
inference time. The package exists to make the failure modes of the idea
measurable on a model whose ground truth is known by construction.
"""
