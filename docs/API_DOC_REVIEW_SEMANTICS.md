# Semantic review versus comment coverage

Comment coverage asks whether documentation text is present. Semantic review asks
whether the text correctly and sufficiently describes the callable.

A callable with a one-line summary can still be undocumented in practice when
callers cannot infer units, shape order, interpolation convention, error
conditions, backend support or safety requirements. Conversely, a private helper
may have no public Rustdoc obligation.

The #1431 program uses source counters to locate work, then semantic review and
compiler reachability to decide completion.
