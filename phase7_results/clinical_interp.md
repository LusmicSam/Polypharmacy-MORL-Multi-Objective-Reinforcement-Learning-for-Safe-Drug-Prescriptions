
## Clinical interpretation

**Environment & decision abstraction.**  
The agent operates over a high-level treatment-policy selection environment (PolypharmacyEnv). The action space is binary (Discrete(2)) and corresponds to two treatment policies rather than individual drug selections. We therefore interpret actions as **policy-level decisions**:

- **Policy 0** — Conservative treatment policy (lower predicted tolerability burden).  
- **Policy 1** — Aggressive treatment policy (higher predicted tolerability burden).

**How policies behave in evaluation.**  
Policy call counts (total calls across all evaluated episodes):  
- Policy 0 total calls: 1951
- Policy 1 total calls: 49

Per-policy mean rewards (computed only over episodes where that policy was present):  
- Policy 0 mean rewards (obj0 , obj1 , obj2): obj0: 0.1289, obj1: -0.0280, obj2: -2.6073  
- Policy 1 mean rewards (obj0 , obj1 , obj2): obj0: 0.1331, obj1: 0.0000, obj2: -3.5370

> Interpretation: policy 0 yields relatively better tolerability (less negative `neg_tol`) than policy 1, at similar efficacy levels. Policy 1 tends to produce a more negative tolerability score, indicating higher adverse effect burden in episodes where it is present.

**Clinical implications and caveats.**  
1. These results do **not** directly translate to drug-level prescriptions. Instead, they indicate the agent’s preference between two coarse strategies (conservative vs aggressive).  
2. The tolerability outcome (`neg_tol`) is a proxy derived from dataset features and the environment’s reward function; it should be validated with clinicians before any actionable inference.  
3. The per-policy mean outcomes are associative: an episode where a policy appears and shows worse tolerability is not evidence that the policy causes worse outcomes without causal analysis or controlled experiments.

