# API documentation CI artifacts

Documentation workflows should retain machine-readable observations when they
help reproduce a review:

- callable/source lexicon JSON;
- example observation JSON;
- aggregate documentation/example metrics;
- source revision/fingerprints;
- negative-control logs when a documentation audit also fixes a verified bug.

Do not retain generated HTML merely to claim documentation exists when rustdoc
can be rebuilt from the revision. Artifacts should support auditability and
metric reproduction, not duplicate the repository indefinitely.
