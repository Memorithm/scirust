# Feature-gated API review prompts

For public feature-gated APIs, document/review:

- exact Cargo feature(s);
- dependency/backend activated;
- whether the API disappears or changes behavior without the feature;
- feature interactions/conflicts;
- target restrictions;
- CI evidence for the feature combination.

Examples should be executed under the required feature in an applicable job.
The general lexicon should eventually expose configuration-specific reachability
rather than presenting every gated item as universally available.
