# Intended property

Euclid's algorithm: replace (x, y) by (y, x % y) until y is 0, at which point
x is gcd(a, b). For a > 0 and b > 0 the answer is gcd_spec(a, b), the source's
own recursive function -- already tail-recursive in the source, so
pl/*/verification.k's `gcdSpec(A, B)` is a transliteration of it with no change
of shape.
