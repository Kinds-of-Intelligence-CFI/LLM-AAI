This file contains the llm_scripting module, which specifies the language the LLM can use to control the agent in the environment.

**Why not just get the LLM to produce atomic actions?**

- It's not generally the way other LLM agents in 3D worlds have been successful (e.g. ref http://arxiv.org/abs/2404.02039 “for games with manipulative control, a translation module is required to translate LLM-generated action into lowlevel actions … In Minecraft, the high-level "approach" action uses an A∗ algorithm for path-finding and executes low-level actions”)
- Easy for the LLM to get lost (see early experiments where it moves in the right direction one step and then behaves apparently randomly)
- Writing code is something that is 'natural' for LLMs: we expect a lot of code to appear in the training set and as it's a big use case we can expect companies are optimising for it.

**Why not use a real programming language?**

This is done e.g. in Voyager (ref: http://arxiv.org/abs/2404.02039 "You are a helpful assistant that writes Mineflayer javascript code to complete any Minecraft task specified by me."). However that is at least in part because that allows it access to the Mineflayer API (along with queing the LLM to use any knowledge it has from training about mineflayer). We could imagine doing something similar with Unity scripts (e.g. https://gist.github.com/mminer/1331271/850c76f129996fba8c3c30063f16596f84a98182), but (1) it is unclear how we could use these scripts through the AAI API and (2) Real languages have a lot of complexity we don't need, possibly making it an unnecessarily harder task (note we're not testing code generating ability of LLMs).

It could be argued that LLMs would be better at generating real languages as they are in their training set. This is possible, but I'm making the assumption that this is 'language-like' enough to be able to utilise any general-purpose programming machinery the LLM might have. Strictly, I think the language could be classified as a (very simple) [Domain Specific Language](https://www.jetbrains.com/mps/concepts/domain-specific-languages).

**We could try to add this as a feature to unity mlagents https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/CONTRIBUTING.md**