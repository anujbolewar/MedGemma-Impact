"""
pipeline — Main orchestration controller.

The controller wires together all subsystems (input → encoder → llm →
safety → output) and drives the event loop within the latency budget.
"""
