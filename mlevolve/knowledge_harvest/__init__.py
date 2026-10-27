"""Collect dated public modelling knowledge for a task into a reviewable set.

Sources are read only as they stood before a cutoff date (the last training
era of the split), so nothing written after the selection or lockbox periods
can reach the agent. Pipeline: sources -> corpus -> extraction -> knowledge set.
"""
