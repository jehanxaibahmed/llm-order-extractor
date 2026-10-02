"""Application layer: the extract-order use case, its prompts, and the ports it depends on.

Depends only on the domain. Talks to the outside world through the protocols in
``ports.py``; concrete implementations live in ``order_extractor.adapters``.
"""
