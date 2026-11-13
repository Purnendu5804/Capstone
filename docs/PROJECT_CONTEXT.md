# Project context

Capstone: emulated onboard satellite pipeline that scores and prioritizes
image tiles under compute, memory and downlink limits, using Sentinel-2 data.

## Pipeline
tiles (from slicer) -> scorer(s) -> bounded priority queue -> downlink simulator -> "ground"

## Ownership
- Slicer: teammate (Vaibhav). We only consume its output. Until it is ready we use a stub.
- Heavy ML model: teammate (Ashutosh). We only call it through the Scorer interface. Until it is ready we use a mock.
- This repo's code (pipeline, scorers, queue, downlink sim, Docker, experiments, metrics) is ours.

## Scorers
- S1: pixel variance of a tile
- S2: edge density (Sobel) of a tile
- S3: tiny model, or the heavy model run on a low-resolution preview (mock for now)
Every scorer must be far cheaper than the heavy model. Cost (time per tile) is
recorded for every scorer on every run.

## Rules for the coding agent
- Python 3.11, type hints everywhere, small functions.
- Every module gets pytest tests. Do not move on until tests pass.
- No hidden global state. All randomness takes an explicit seed.
- Experiments are driven by config files. Results are written to results/ and never edited by hand.
- Do not invent features outside the current step.
- At the end of each step, update README.md with how to run it, and list what was done and what was not.