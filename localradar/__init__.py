"""local-model-radar -- find local models that fit the GPU and actually tool-call.

The local counterpart of model-radar. On a hosted API, "free" means zero price;
on an Ollama box, "free" means FITS IN VRAM -- a 27B model that offloads to CPU
isn't a saving, it's a slowdown. So the two primitives are:

  * radar:  list candidate models (Ollama library), filter by hardware fit,
            pull + probe the new ones (tool call + tok/s + context), and
            optionally adopt the best through the same gate as model-radar.
  * canary: watch the live local model -- is Ollama up, is the model resident
            in VRAM, does it still serve tool calls -- and fail over to a
            lighter model when the GPU dies or the resident model is evicted.

The three rules carry over unchanged: a 200 is not a capability test; unreachable
is not dead; free is not a saving if it is slow.
"""
from __future__ import annotations

__version__ = "0.1.0"
