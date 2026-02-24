"""
llm — MedGemma model loading and intent-to-sentence reconstruction.

Loads google/medgemma-4b-it in 4-bit NF4 quantization and performs
constrained token-to-sentence reconstruction within the 2.5s latency budget.
"""
