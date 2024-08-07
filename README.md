# Granite Timeseries Cookbook

We'll start by implementing "recipe" forms of these TS notebooks: https://github.com/ibm-granite/granite-tsfm/tree/main/notebooks.

How do we prioritize which ones to do first?
* Most relevant use cases for industry.
* Easiest to get done.
* Good datasets already exist.

Some TBD items:
* Can we actually host TS models on Replicate, etc. or are they too different from LLMs for this purpose?
* If they are to different, should we sketch out the design of an OSS TS server?
* On the other hand, TS models are typically much smaller than LLMs, so is local inference completely sufficient for our purposes?
* Even if that's true, is there still virtue in creating an industry-leading, scalable, OSS TS server, i.e., like vLLM for LLMs?

For information about contributing to this repo, code of conduct guidelines, etc., see the [community](https://github.com/granite-cookbooks/community) project.
