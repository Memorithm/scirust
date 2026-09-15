# Validation-order review prompts

Public boundary validation should occur before operations that can panic,
overflow, truncate or misattribute an error.

Review order such as:

1. existence/identity;
2. rank/basic shape;
3. zero/range/checked narrowing;
4. cross-operand compatibility;
5. checked size/storage arithmetic;
6. representation/backend support;
7. execution.

Examples/tests should verify typed failure before dangerous arithmetic. This
pattern prevented zero KV heads from reaching modulo and large dimensions from
being silently narrowed in the attention audit.
