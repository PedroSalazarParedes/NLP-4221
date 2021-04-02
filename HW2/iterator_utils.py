import random
from typing import TypeVar, Iterator, List, Tuple, Callable, Dict
from itertools import chain, tee
from functools import reduce

A = TypeVar("A")
B = TypeVar("B")

def apply_and_preserve (f: Callable[[Iterator[A]], B], xs: Iterator[A]) -> Tuple[B, Iterator[A]]:
    ys, zs = tee(xs)
    return f(ys), zs

def size (xs: Iterator[A]) -> int:
    return reduce(lambda acc, _: acc + 1, xs, 0)

def buffer (xs: Iterator[A], n: int) -> Iterator[List[A]]:
    chunk = []
    for x in xs:
        if len(chunk) == n:
            yield chunk
            chunk = [x]
        else:
            chunk.append(x)
    if len(chunk) > 0:
        yield chunk

def shuffle (xs: Iterator[A], buffer_size: int) -> Iterator[A]:
    chunks = buffer(xs, buffer_size)
    def shuf (xs: List[A]):
        return random.sample(xs, len(xs))
    rand_chunks = map(shuf, chunks)
    return chain.from_iterable(rand_chunks)

def take (n: int, xs: Iterator[A]) -> Iterator[A]:
    for i, x in enumerate(xs):
        if i < n:
            yield x
        else:
            break

def drop (n: int, xs: Iterator[A]) -> Iterator[A]:
    for i, x in enumerate(xs):
        if i >= n:
            yield x

def partition2 (xs: Iterator[A], part1_size: int) -> Tuple[Iterator[A], Iterator[A]]:
    p1 = take(part1_size, xs)
    return p1, xs

def chain_unique (xs: Iterator[Iterator[A]]) -> Iterator[A]:
    seen = set()
    chained = chain.from_iterable(xs)
    for x in chained:
        if x not in seen:
            seen.add(x)
            yield x

def unique (xs: Iterator[A], key=lambda x : x) -> Iterator[A]:
    seen = set()
    for x in xs:
        if key(x) not in seen:
            seen.add(x)
            yield x

def count_occurrences (xs: Iterator[A]) -> Dict[A, int]:
    count = {}
    for x in xs:
        count[x] = count[x] + 1 if x in count else 1
    return count

# xs = (i for i in range(100))
# n, ys = apply_and_preserve(size, xs)
# print(n, list(xs))
# shuffled, zs = apply_and_preserve(lambda xs: shuffle(xs, 10), ys)
# print(list(shuffled))
# p1, p2 = partition2(zs, 10)
# print(list(p1), list(p2))