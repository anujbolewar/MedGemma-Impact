"""
safety — Output safety guards: entropy checking, grammar validation,
and confidence scoring.

Every LLM output passes through this pipeline before being voiced,
ensuring no low-confidence or malformed sentences reach the patient.
"""
