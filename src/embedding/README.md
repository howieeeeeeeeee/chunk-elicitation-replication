# Reasoning Embedding Contract

This package defines one deterministic embedding entity for one reasoning
string at a simulation-session decision index and one sanitized OpenRouter
configuration.

`request_reasoning_embedding(reason_text, embedding_config)` sends exactly one
`POST` to `https://openrouter.ai/api/v1/embeddings`. It reads credentials only
from `OPENROUTER_API_KEY`, performs no retry, accepts only float encoding, and
returns the validated response id, resolved model, object type, float vector,
vector dimension, and complete JSON-normalized usage object. The usage `cost`
is stored as OpenRouter credits; an absent cost is `null` and is never
estimated.

Configurations support `model`, optional `dimensions`, optional `input_type`,
optional OpenRouter `provider` routing preferences, and
`encoding_format: "float"`. Unknown or secret-bearing keys are rejected before
the request. Canonical JSON determines the configuration hash. The entity id is
derived only from the simulation-session id, zero-based decision index, and
configuration hash; the reasoning text at that identity is immutable.

Entities store the input and sanitized configuration, top-level success/output
and latest usage, compact attempt history, and UTC timestamps. A source failure
may store `input_text: null` when the reasoning value is unavailable. Only a
successful vector is stored at top level; attempt history never duplicates
vectors or stores raw responses, request headers, or credentials.
