# API documentation versioning policy

SciRust documentation describes the code revision it accompanies.

When an API's behavior/signature changes, update its Rustdoc and examples in the
same change. Generated lexicon/example artifacts must be regenerated from the
final source revision. Evidence from earlier revisions remains historical and
must retain its commit/date rather than being silently edited into a current
claim.

For format/protocol/checkpoint APIs, document compatibility explicitly rather
than assuming repository versioning implies wire/file compatibility. For
experimental APIs, state expected instability but still document the current
contract precisely.
