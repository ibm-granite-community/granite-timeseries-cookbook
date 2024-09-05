# IBM Granite Timeseries Cookbook

The "Recipes" in the Granite Timeseries Cookbook showcase the capabilities of
the IBM Granite Timeseries models.

## Contents

See the [recipes](recipes/)

## Contributing

### General Requirements

For information about contributing to this repo, code of conduct guidelines, etc., see the community [CONTRIBUTING][CG] and [Code of Conduct][CoC] guides.  All commits require DCO-signoff (discussed [here][CG-legal]) _and_ GPG or SSH signing (discussed [here][CG-signing]).  The GitHub recommended code security settings are enforced on this public repository (which include the signing requirement).

### What is a "Recipe"?

A "cookbook" is composed of "recipes".
In this version of the Granite Timeseries Cookbook, a "recipe" is a Python notebook.

Under the implied cooking analogy, there are three key defining elements of a
good recipe:

1. It clearly states the required ingredients and tools up front
2. It is straightforward to reproduce efficiently
3. It results in something delicious to eat

The "ingredients" and "tools" of the first point mean the data and software at hand.

The second point is straightforward to map into the technical space:
The reader should be able to run the cells of the notebook sequentially
and the same result as what was published in the original notebook.
Our objective is to make these recipes reproducible in 15 minutes or less.
This may inform the decisions about recipe granularity.

The third point is more subtle, and is what sets this kind of writing apart.
In a technical sense, "something delicious to eat" means a system
that demonstrates useful functionality in such a way that it sets the
reader on the path to adopting it in their environment.  It clearly articulate the business value achieved by the resulting system.

For an "Energy Demand Forecasting" recipe, for instance, the recipe should:

- State at a high level how a forecasting capability could create "business value", and therefore be worth the investment.
- Provide example input data
- Provide an example forecast request
- Show the expected resulting prediction
- Provide enough code to walk through the whole process

For recipe authors with strong familiarity with a specific capability or tool,
the first inclination may be to write a recipe oriented around the tool.
Consider alternate ways to phrase the recipe so that the end result is showcased, rather than the tool.

Under the cooking analogy, that would mean writing a great soup recipe rather than one that talks about the features of a food processor.  If the soup tastes great and is easy to prepare, the reader will likely want to know more about how it was made.

Recipes will vary in complexity.
Some may be single inference calls.
Others may illustrate useful agentic workflows.

A "cookbook" is not intended to be a comprehensive guide to all
issues that may arise during development with Granite Timeseries.
Recipes will link to helpful external resources on topics including: distributed systems, UI, design, AI/ML theory, metrics, etc.

### Exceptions

While the majority of the content in the Granite Timeseries Cookbook
should be "recipes", there will be a need for other kinds of exposition.

Relating to Granite Timeseries, these might include:

- Supporting timeseries libraries and databases
- Application frameworks such as LangChain (and LangServe, LangGraph, ...) and LlamaIndex

### Granite Timeseries

The common element of all the recipes in this Cookbook is either direct usage of a Granite Timeseries model, or strong relevance to those models.

### Everything Local (... except, perhaps, the model)

Minimize the reader's exposure to sign-up flows and stateful network calls.

In most cases, the user should expect to do a single `git clone`,
run a notebook, and then optionally do some `pip install` early in
the execution of the notebook.

The recipes should not depend on the model being either remotely hosted
or local.  It is OK for a recipe to make a choice, but providing a link
to instructions for the other choice is useful.

### Automated Validation

The code in the recipes will be automatically validated (coming soon).
Minimally, this should guarantee that no unexpected errors occur.
Checking deeper assertions about the behavior of the code
in the cells is preferred.

### Performance

Where available and appropriate, cite specific relevant expectations
and claims for quality, latency, cost, or any other relevant metrics.

In some cases, it may be appropriate to include a demonstration
of establishing those benchmarks.

### Other general guidance

- The Developer Certificate of Origin (DCO) applies to the code, documentation, and any example data provided. See community CONTRIBUTING guide [legal section][CG-legal] for more on the DCO.
- Commits must be GPG or SSH signed. See the community CONTRIBUTING guide [signing commits section][CG-signing] for details.
- Prefer an opinionated recipe over one that is flexible.
- With that said, sometimes offering examples from multiple domains (eg for "text to sql") can be helpful.  If that brings too much complexity, split into smaller recipes.
- Keep in mind a specific user persona when writing a recipe.  Rather than writing for a general audience, can you imagine that one specific user would find the recipe valuable from beginning to end?
- Take an iterative approach to the development of this Cookbook.
- Expect that over time, recipes will be split, merged, made uniform, deprecated, replace, or deleted.
- The `ibm-granite-community` org does not host OSS "code" other than what is directly visible in the recipes and simple utility functions.

## Licenses

The Granite Time Series Cookbook's base license is CC BY 4.0.

Code in this repository, including in notebook cells, is licensed under Apache 2.0.

Any example datasets committed to this repository are licensed under CDLA Permissive 2.0.

## IBM Public Repository Disclosure

All content in these repositories including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.

[CoC]: https://github.com/ibm-granite-cookbooks/community/blob/main/CODE_OF_CONDUCT.md
[CG]: https://github.com/ibm-granite-cookbooks/community/blob/main/CONTRIBUTING.md
[CG-legal]: https://github.com/ibm-granite-cookbooks/community/blob/main/CONTRIBUTING.md#legal
[CG-signing]: https://github.com/ibm-granite-cookbooks/community/blob/main/CONTRIBUTING.md#signing-commits