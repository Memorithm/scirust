# Storage/database API review prompts

For public persistent storage/database APIs, document/review:

- key/schema/namespace conventions;
- transaction/atomicity/isolation semantics;
- ordering/iteration guarantees;
- serialization/versioning;
- concurrency/locking;
- durability/flush behavior;
- corruption/recovery/errors;
- resource/path ownership.

Examples should use temporary/local storage with an exact insert/read or
transaction property plus a distinct missing/conflict/error case.
