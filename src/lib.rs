#![allow(clippy::unused_unit)]
#![forbid(unsafe_code)]
//#![feature(trait_alias)]
//#![feature(fn_traits)]
//#![feature(unboxed_closures)]
// #![feature(f128)]
// #![feature(portable_simd)]

use std::char::from_u32;
pub use std::cmp::{max};
pub use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Iter;
use std::default::Default;
use std::fmt::{Binary, Debug, LowerHex, Octal};
use std::fs::{read_to_string, write};
use std::hash::Hash;
pub use std::iter::{Enumerate, Filter, IntoIterator, Sum, zip};
use std::ops::{Index, IndexMut, Neg, Range};
use std::path::Path;
use std::str::FromStr;
pub use regex::Regex;

// Not safe for cryptography
use fxhash::FxBuildHasher as DefaultHasher;  // faster than std HashMap, at least for small hashes.
pub use fxhash::hash;
// use hashbrown::HashMap; - extension of std HashMap, but std is actually only api around this
pub use itertools::itertools::Itertools;  // one extra level of itertools because it is a submodule
use itertools::itertools::traits::HomogeneousTuple;

pub mod hashlib;

// skiped async functions and all those that depend on OOP / inheritance

// list.append = vec.push, list.extend = vec.append

// it is not possible to have kwargs in rust, but we can pass dict (but then if called with defaults, we still need to pass empty one)
// or use macro, or use builder pattern and call optional methods
// or use a pattern like https://github.com/alexpusch/rust-magic-patterns/tree/master/axum-style-magic-function-param

pub trait Naruto: Clone {}
impl<T: Clone> Naruto for T {}
#[allow(dead_code)]
pub trait Kakashi: Copy {}
impl<T: Copy> Kakashi for T {}

//#[macro_export] macro_rules! comprehension {
//    // usage: comprehension![{ print(x); } for x in 0..10]
//    ($body:block for $var:ident in $iterable:expr) => {
//        $iterable.into_iter().for_each(|$var| $body)
//    };
//    ($body:block for $key:ident, $val:ident in $iterable:expr) => {};  // dict
//}


// there is mapcomp crate providing vecc![] macro (5 months ago)
// or list_comprehension_macro = "0.1.1" (updated 2 years ago), cute = "0.3.0" (c![] macro; 6 ys ago)
#[macro_export]
macro_rules! comp {
    // usage: println!("{:?}", comprehension![{ x + 1 } for x in 0..10]);
    //($body:block for $var:ident in $iterable:expr) => {
    //    $iterable.into_iter().map(|$var| $body).collect::<Vec<_>>()
    //};
    //($body:block for $key:ident, $val:ident in $iterable:expr) => {
    //    $iterable.into_iter().map(|($key, $val)| $body).collect::<Vec<_>>()
    //};
    (@eval {$mapping:expr} for {$pattern:pat} in {$iterator:expr} $(if {$condition:expr})*) => {
        <_ as ::core::iter::Iterator>::filter_map(
            <_ as ::core::iter::IntoIterator>::into_iter($iterator),
            |$pattern| (true $(&& ($condition))*).then(|| ($mapping))
        )
    };

    (@eval {$mapping:expr}
        for {$pattern:pat} in {$iterator:expr} $(if {$condition:expr})*
        $(for {$pattern2:pat} in {$iterator2:expr} $(if {$condition2:expr})*)+
    ) => {
        <_ as ::core::iter::Iterator>::flatten(
            <_ as ::core::iter::Iterator>::filter_map(
                <_ as ::core::iter::IntoIterator>::into_iter($iterator),
                |$pattern| (true $(&& ($condition))*).then(|| (
                    comp!(@eval {$mapping} $(for {$pattern2} in {$iterator2} $(if {$condition2})*)+)
                ))
            )
        )
    };

    (@scan mapping [] [$($group:tt)*] for $($tail:tt)*) => {
        comp!(@scan pattern [{$($group)*}] [] $($tail)*)
    };
    (@scan pattern [$($done:tt)*] [$($group:tt)*] in $($tail:tt)*) => {
        comp!(@scan iterator [$($done)* for {$($group)*}] [] $($tail)*)
    };
    (@scan iterator [$($done:tt)*] [$($group:tt)*]) => {
        comp!(@eval $($done)* in {$($group)*})
    };
    (@scan iterator [$($done:tt)*] [$($group:tt)*] if $($tail:tt)*) => {
        comp!(@scan condition [$($done)* in {$($group)*}] [] $($tail)*)
    };
    (@scan iterator [$($done:tt)*] [$($group:tt)*] for $($tail:tt)*) => {
        comp!(@scan pattern [$($done)* in {$($group)*}] [] $($tail)*)
    };
    (@scan condition [$($done:tt)*] [$($group:tt)*]) => {
        comp!(@eval $($done)* if {$($group)*})
    };
    (@scan condition [$($done:tt)*] [$($group:tt)*] if $($tail:tt)*) => {
        comp!(@scan condition [$($done)* if {$($group)*}] [] $($tail)*)
    };
    (@scan condition [$($done:tt)*] [$($group:tt)*] for $($tail:tt)*) => {
        comp!(@scan pattern [$($done)* if {$($group)*}] [] $($tail)*)
    };

    (@scan $kind:ident [$($done:tt)*] [$($group:tt)*] $head:tt $($tail:tt)*) => {
        comp!(@scan $kind [$($done)*] [$($group)* $head] $($tail)*)
    };

    // fallback to prevent infinite loop
    (@ $($token:tt)*) => {
        compile_error!(::core::concat!("comp!(@", ::core::stringify!($($token)*), ")"))
    };

    ($($token:tt)*) => {
        <_ as std::iter::Iterator>::collect::<Vec<_>>(
            comp!(@scan mapping [] [] $($token)*)
        )
    }
}


#[macro_export]
macro_rules! dict_comprehension {
    // usage: dict_comprehension![{ (x, y) } for x, y in (0..10).zip(0..10)]
    ($body:block for $key:ident, $val:ident in $iterable:expr) => {
        $iterable.into_iter().map(|($key, $val)| $body).collect::<Dict<_, _>>()
    };
}


// access tuple items dynamically: https://crates.io/crates/seq-macro
// seq!(N in 0..=2 {
//     sum += tuple.N;
// });

// TODO implement "from" dict into bool: if hash_map.is_empty() => false

pub type Set<T> = HashSet<T, DefaultHasher>;
pub type Dict<K, V> = HashMap<K, V, DefaultHasher>;  // TODO impl fromkeys()

pub trait Items<K, V> {
    fn items(&self) -> Iter<K, V>;
}

impl<K, V> Items<K, V> for Dict<K, V> {
    fn items(&self) -> Iter<K, V> {
        self.iter()
    }
}



pub fn abs<T: Ord + Default + Neg<Output = T>>(x: T) -> T {
    if x < T::default() {
        -x
    } else {
        x
    }
}


pub fn all<I: IntoIterator<Item = bool> + Clone>(iterable: &I) -> bool {
    iterable.clone().into_iter().all(|x| x)
}


pub fn any<I: IntoIterator<Item = bool> + Clone>(iterable: &I) -> bool {
    iterable.clone().into_iter().any(|x| x)
}


pub fn bin<T: Binary>(x: T) -> String {
    format!("{x:b}")
}


pub fn chr(code_point: u32) -> char {
    // TODO not generic - would need macro like
    // https://github.com/rust-itertools/itertools/blob/1fb979b51e24a9f78d719c0f07d14af18b12be3f/src/tuple_impl.rs#L336
    from_u32(code_point).unwrap()
}


pub fn dict<K, V>() -> Dict<K, V> {
    // TODO or do smt more clever than returning empty dict?
    HashMap::with_hasher(DefaultHasher::default())
}


// pub fn enumerate<T>(iterable: &Vec<T>) -> Enumerate<std::slice::Iter<'_, T>> {
pub fn enumerate<I: IntoIterator + Clone>(
    iterable: &I,
) -> Enumerate<<I as IntoIterator>::IntoIter> {
    // TODO performance of Clone / Copy?
    iterable.clone().into_iter().enumerate()  // in Vec case, we can use iter() instead of into_iter()
}


pub fn filter<I: IntoIterator + Clone, F>(
    callable: F, iterable: &I,
) -> Filter<<I as IntoIterator>::IntoIter, F>
where
    F: FnMut(&I::Item) -> bool,
{
    iterable.clone().into_iter().filter(callable)
}


pub type FrozenSet<T> = HashMap<usize, Set<T>, DefaultHasher>;

pub trait Frozen<T> {
    fn add(&mut self, other: Set<T>) -> ();
    fn insert(&mut self, other: Set<T>) -> ();
    fn remove(&mut self, other: &Set<T>) -> bool;
    // TODO rest of methods or use frozenset (or hashable) crate
    fn contains(&self, item: &T) -> bool;  // contains_key is not equivalent to in
}

impl<T: Debug + Ord> Frozen<T> for FrozenSet<T> {
    // usage:
    // let mut my_set_of_frozensets = set_of_frozensets();
    // <dyn Frozen<_>>::insert(my_set_of_frozensets, set());
    // because of clash with other traits methods / struct methods names
    fn add(&mut self, other: Set<T>) -> () {
        <dyn Frozen<_>>::insert(self, other);
    }

    fn contains(&self, item: &T) -> bool {
        self.contains_key(&hash(&format!("{item:?}")))
    }

    fn insert(&mut self, other: Set<T>) -> () {
        let to_hash = format!("{:?}", other.iter().sorted());  // sorted to make it really unique
        HashMap::insert(self, hash(&to_hash), other);
    }

    fn remove(&mut self, other: &Set<T>) -> bool {
        let to_hash = format!("{:?}", other.iter().sorted());
        HashMap::remove(self, &hash(&to_hash)).is_some()
    }
}

// cannot implement outer traits for outer types
//impl<T> Hash for FrozenSet<T> {
//    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//        for (k, v) in self {
//            k.hash(state);
//            v.hash(state);
//        }
//    }
//}

pub fn set_of_frozensets<T>() -> FrozenSet<T> {
    dict()
}


// reimported
// pub fn hash(to_hash: &[u8]) -> u64 {
//     // TODO can write various input types
//     let mut hasher = DefaultHasher::new();
//     hasher.write(to_hash);
//     hasher.finish()
// }


pub fn hex<T: LowerHex>(x: T) -> String {
    format!("{x:x}")
}


pub fn int<T: FromStr>(s: &str) -> T
where
    <T as FromStr>::Err: Debug,
{
    // need to specify the type by caller
    s.parse().unwrap()
}


pub fn iter<T: Clone + IntoIterator>(iterable: &T) -> T::IntoIter {
    iterable.clone().into_iter()
}


pub fn len<I>(iterable: &I) -> usize
where
    I: IntoIterator + Clone,
    I::IntoIter: ExactSizeIterator,  // TODO too strict
    I::Item: Clone,
{
    // TODO do not return usize explicitly, but some generic integer; as it is done in the int() fcn.
    // Or return directly int(current return)
    iterable.clone().into_iter().len()
}


/// collect iterator to list (as in python collect map / filter result)
pub fn list<T: Iterator<Item = U>, U>(iterator: T) -> Vec<U> {
    iterator.collect::<Vec<_>>()
}


pub fn map<I, F, T>(callable: F, iterable: I) -> impl Iterator<Item = T>
where
    I: IntoIterator,
    F: Fn(I::Item) -> T,
{
    iterable.into_iter().map(callable)
}


// max, min reimported; but they take only 2 arguments, not list of any len


pub fn next<I: Iterator>(iterator: &mut I) -> Option<I::Item> {
    // TODO do we need to return option? Probably yes, we "raise StopIteration".
    iterator.next()
}


pub fn oct<T: Octal>(x: T) -> String {
    format!("{x:o}")
}


#[derive(PartialEq)]
enum FileMode {
    Read,
    Write,
    Append,
}


pub struct FileIO<'a> {
    fname: &'a str,
    mode: FileMode,
}

impl FileIO<'_> {
    pub fn readlines(&self) -> Vec<String> {
        // not sure if the map is needed
        // TODO also not sure if we want to return list or iterator
        self.read().lines().map(|l| l.to_string()).collect()
    }

    pub fn read(&self) -> String {
        if self.mode != FileMode::Read {
            panic!("File not opened in read mode");
        }
        // let mut f = File::open(Path::new(self.fname)).unwrap();
        // let mut buffer = String::new();
        // f.read_to_string(&mut buffer).unwrap();
        
        // buffer
        
        read_to_string(Path::new(self.fname)).unwrap()
    }

    pub fn write<C: AsRef<[u8]>>(&self, contents: C) -> () {
        if self.mode != FileMode::Write || self.mode != FileMode::Append {
            panic!("File not opened in readable mode");
        }
        // let mut buffer = File::create(self.fname).unwrap();
        // buffer.write(b"some bytes").unwrap();
        write(self.fname, contents).unwrap();  // calling .unwrap() to panic in case of error
        // TODO what in case of append?
    }
}


pub fn open<'a>(fname: &'a str, mode: &'a str) -> FileIO<'a> {
    // TODO fname could be AsRef<Path> or AsRef<str> to be more generic
    // ideally, the open should be called in its own scope, so that the file is closed when going out of scope
    let mode_enum = match mode {
        "r" => FileMode::Read,
        "w" => FileMode::Write,
        "a" => FileMode::Append,
        // TODO other modes?
        _ => panic!("Unknown mode"),
    };
    FileIO { fname, mode: mode_enum }
}


pub fn ord(c: char) -> u32 {
    c as u32
}


//pub fn pow<T: std::ops::Mul<Output = T> + Copy + From<u32>, P: Into<u32>>(x: T, y: P) -> T {
pub fn pow<T, P>(x: T, y: P) -> i64  // T: From<i64>, P: From<P>
// TODO return T, but that requires TryFrom handling
where
    i64: TryFrom<T>, u32: TryFrom<P>,
{
    let input = i64::try_from(x);
    let exponent = u32::try_from(y);
    match exponent {
        Ok(exp) => match input {
            Ok(val) => val.pow(exp),
            Err(_) => panic!("Failed to convert input"),
        },
        Err(_) => panic!("Failed to convert exponent"),
    }
}


pub fn print<T: Debug>(printable: &T) -> () {
    println!("{printable:?}");
}


pub mod pprint {
    use std::fmt::Debug;
    pub fn pprint<T: Debug>(printable: &T) -> () {
        println!("{printable:#?}");
    }
}


pub fn range<T: Default>(end: T) -> Range<T> {
    T::default()..end
}


/*
pub fn reversed<I, T>(iterable: &I) -> I::IntoIter
where
    I: Clone + IntoIterator<Item = T, IntoIter = std::vec::IntoIter<T>>,  // , IntoIter = std::slice::Iter<'a, T>
    I::IntoIter: DoubleEndedIterator //  + FromIterator<T>,
{
    iterable.clone().into_iter().rev().collect::<Vec<_>>().into_iter()
}
*/
pub fn reversed<I, T>(iterable: &I) -> I::IntoIter
where
    I: Clone + IntoIterator<Item = T>,
    I::IntoIter: DoubleEndedIterator + FromIterator<T>,
{
    // TODO the I::IntoIter: FromIterator<T> is not sensible, we need smt more generic
    iterable.clone().into_iter().rev().collect()
}


// pub fn round<T: Float>(x: T, ndigits: usize) -> T {
// should return int in case ndigits is 0

// slice - not sure how to implement


// Would have to be macro, to have 0 or 1 param
// pub fn set<T: IntoIterator<Item = U>, U: std::hash::Hash + Eq>(iterable: T) -> std::collections::HashSet<U> {
//     // cast to set
//     iterable.into_iter().collect::<std::collections::HashSet<_>>()
// }
pub fn set<T>() -> HashSet<T, DefaultHasher> {
    // empty set
    HashSet::with_hasher(DefaultHasher::default())
}


// This one was not generic enough
// pub fn sorted<I, F, T, U: Ord>(iterable: &I, key: F) -> I::IntoIter
// where
//     I: Clone + IntoIterator<Item = T>,
//     I::IntoIter: FromIterator<T>,
//     F: FnMut(&I::Item) -> U,
// {
//     iterable.clone().into_iter().sorted_by_key(key).collect()
// }


pub fn sorted<I, F, T: Clone, U: Ord>(iterable: &I, key: F) -> impl Iterator<Item = T> + Clone
where
    I: Clone + IntoIterator<Item = T, IntoIter = std::vec::IntoIter<T>>,
    F: FnMut(&I::Item) -> U,
{
    // not sure if all generics are needed; there is a lot of them in the signature
    // call with key = lambda x: x or std::convert::identity for default
    // in case of lifetime problems for iterable<&str>, use |x| x.to_owned()
    iterable.clone().into_iter().sorted_by_key(key)
}


pub fn sum<I>(iterable: &I) -> I::Item
where
    I: IntoIterator + Clone,
    I::Item: Sum,
{
    //iterable.clone().iter().sum()  // will probably be working for Vec only
    iterable.clone().into_iter().sum()
}


//pub fn sum_bools<I>(iterable: &I) -> u32
//where
//    I: IntoIterator + Clone,
//    I::Item = bool,
//{
//    map(int::<u32>, iterable.clone().into_iter()).sum()
//}


pub fn tuple<I, T: HomogeneousTuple<Item = I::Item>>(iterable: I) -> T  // (I::Item, I::Item)
where
    I: IntoIterator + Clone,
    I::IntoIter: Itertools,
{
    // cast to tuple
    // itertools are able to return tuple of size up to 12
    iterable.into_iter().collect_tuple().unwrap()
}


// zip reimported

// anyhow - for any error propagation to main (not to be used in actual libraries)
// base64 = "0.22.1"
// rand = "0.8.5" or getrandom = "0.2.15"
// serde = "1.0.210", serde_derive = "1.0.210"
// serde_json = "1.0.128"
// regex-syntax = "0.8.5" | regex
// clap = "4.5.20" - argparse
// time = "0.3.36"
// either = "1.13.0" - left, right option - functional
// reqwest = "0.12.8"
// tokio = "1.40.0", async-std = "1.13.0" - async
// sqlx = "0.8.2" - umoznuje mimo jiné compile-time checking sql dotazů
// sqlformat = "0.3.0"
// Diesel - orm
// glob = "0.3.1"
// hyperfine = "1.18.0" - benchmarking
// aws-sdk-<service>
// r2d2 - universal connections to databases and more. Has separate crates for specific databases
// bracoxide - expand string with braces
// axum - web framework
// pathfinding - find paths in graph
// fast_paths
// chrono - datetime
// rayon - parallelism of iterators ((into_)par_iter() instead of (into_)iter()) or more general parallelism (mutex...)
// inline_python - run python code blocks from Rust via macro - not super stable
// wiremock - mock http server
// image - image processing
// tailcall - rewrites tail-recursive functions to use iteration
// stacker - spills over to the heap if the stack has hit its recursion limit
// phf - perfect hash functions
// matrix_operations = "0.1.4"
// std::ops::Deref can be used as callable class, if that callable does not take args: build = Builder(...); build();


pub trait SetAdd<T> {
    // alias
    fn add(&mut self, other: T) -> ();
}

pub trait SetUpdate<I> {
    fn update(&mut self, other: I) -> ();
}

impl<T: Hash + Eq> SetAdd<T> for Set<T> {
    fn add(&mut self, other: T) -> () {
        self.insert(other);  // .clone()
    }
}

impl<I: IntoIterator<Item = T>, T: Hash + Eq> SetUpdate<I> for Set<T> {
    fn update(&mut self, other: I) -> () {
        for item in other {
            self.insert(item);
        }
    }
}


pub mod decimal {}


pub mod heapq {
    // heap is implemented to pop greatest element in Rust - use Reverse
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    pub fn heapify<T: Ord>(iterable: Vec<T>) -> BinaryHeap<Reverse<T>> {
        let mut heap = BinaryHeap::new();
        for item in iterable {
            heap.push(Reverse(item));
        }

        heap
    }

    pub fn heappop<T: Ord>(heap: &mut BinaryHeap<Reverse<T>>) -> T {
        Reverse(heap.pop().unwrap().0).0
    }

    pub fn heappush<T: Ord>(heap: &mut BinaryHeap<Reverse<T>>, item: T) -> () {
        heap.push(Reverse(item));
    }
}


pub mod collections {
    pub use std::collections::VecDeque;

    // namedtuple
    // chainmap - I could implement an "inheritance" of attributes with that (traits only provide methods)

    #[allow(dead_code)]
    trait DequeRotate<T> {
        // alias
        fn rotate(&mut self, n: usize) -> ();
    }

    impl<T> DequeRotate<T> for VecDeque<T>
    {
        fn rotate(&mut self, n: usize) -> () {
            // TODO make n isize and switch direction if n < 0
            // Also add % to not have bigger n than len
            self.rotate_left(n);
        }
    }

    pub fn deque<I: IntoIterator<Item = T>, T>(iterable: I) -> VecDeque<T> {
        // cannot use VecDeque::append(), other arg needs to be VecDeque itself
        let mut queue = VecDeque::new();
        for item in iterable {
            queue.push_back(item);
        }

        queue
    }

    // ChainMap
    // Counter - TODO this needs to be implemented, I think I use it somewhere
    // indexmap = "2.6.0" - OrderedDict, orderedset
    // OrderedDict
    // defaultdict
    // UserDict
    // UserList
    // UserString
}


pub mod functools {
    use crate::{Dict, dict};
    use std::hash::Hash;
    use std::cmp::Eq;
    use std::cell::RefCell;

    #[allow(clippy::type_complexity)]
    pub struct LruCache<'a, K, V> {
        // Dynamic dispatch
        data: RefCell<Dict<K, V>>,  // interior mutability
        wrapped: Box<dyn Fn(&dyn Fn(K) -> V, K) -> V + 'a>,
        // TODO So far only supports one arg to the wrapped, so either 1 hashable or tuple of more args
        // TODO multithread version - but maybe works already
        // TODO we could overcome the need for hashable key by using custom struct, see advent_24_21
    }

    impl<'a, K: Clone + Eq + Hash, V: Clone> LruCache<'a, K, V> {
        pub fn new<F>(to_wrap: F) -> Self
        where
            F: Fn(K) -> V + 'a,
        {
            // The lifetime on struct is only for this closure
            let wrapped = move |_: &dyn Fn(K) -> V, k: K| to_wrap(k);
            LruCache {
                data: dict().into(),
                wrapped: Box::new(wrapped),
            }
        }

        #[allow(clippy::type_complexity)]
        pub fn new_recursive(to_wrap: Box<dyn Fn(&dyn Fn(K) -> V, K) -> V>) -> Self {
            // Borrow checker is having hard time assuming lifetimes
            // if parameters passed by ref. Can be overcome by passing other indirection like Rc
            LruCache {
                data: dict().into(),
                wrapped: to_wrap,
            }
        }

        /// Call the cache
        pub fn call(&self, key: K) -> V {
            if let Some(value) = self.data.borrow().get(&key) {
                //println!("Cache hit");
                return value.clone();
            }

            //println!("Cache miss");
            let cache_fn = |k: K| self.call(k);
            let value = (self.wrapped)(&cache_fn, key.clone());

            self.data.borrow_mut().insert(key, value.clone());
            value
        }
    }

    //pub fn lru_cache<F, K, V>(cache_fn: F, k: K) -> LruCache<K, V>
    //where
    //    F: Fn(K) -> V + 'static,
    //    K: Clone + Eq + Hash,
    //    V: Clone,
    //{
    //    LruCache::new_recursive(Box::new(|cache_fn, k| {
    //        cache_fn(k)
    //    }))
    //}

    //pub fn reduce<F, T>(iterable: &mut dyn Iterator<Item = T>, init: T, f: F) -> T
    pub fn reduce<F, I, T>(iterable: I, f: F) -> T
    where
        F: FnMut(T, T) -> T,
        I: IntoIterator<Item = T>,
    {
        iterable.into_iter().reduce(f).unwrap()
    }
}


pub mod itertools {
    // add also more-itertools. Minimally distinct_permutations
    // https://github.com/more-itertools/more-itertools/blob/v10.5.0/more_itertools/more.py#L661
    use std::iter::Cycle;

    pub use itertools;
    use itertools::iproduct;
    use itertools::structs::{Combinations, Product};
    pub use itertools::Itertools;

    use crate::Naruto;

    //count(start) // optional stop
    //repeat(elem) // endlessly or up to n times
    //accumulate(iterable) // optional func
    //batched(p, n)
    //chain(p, q) // originally supports any number of arguments to chain
    //chain.from_iterable(iterable)
    //compress(data, selectors)
    //dropwhile(predicate, seq)  // predicate: Callable
    //filterfalse(predicate, seq)
    //groupby(iterable)  // optional key
    //islice(seq, stop)
    //pairwise(iterable: I)
    //starmap(func, seq)  // probably just map with some flavour
    //takewhile(predicate, seq)
    //tee(it, n)
    //zip_longest(arbitrary number of args)
    //permutations(p)  // optional r
    //combinations_with_replacement(p, r)

    pub fn combinations<T: IntoIterator + Clone>(
        iterable: &T, r: usize,
    ) -> Combinations<<T as IntoIterator>::IntoIter>
    where
        T::Item: Clone,
    {
        iterable.clone().into_iter().combinations(r)
    }

    pub fn cycle<T: IntoIterator + Naruto>(iterable: &T) -> Cycle<<T as IntoIterator>::IntoIter>
    where
        <T as std::iter::IntoIterator>::IntoIter: Naruto,
    {
        iterable.clone().into_iter().cycle()
    }

    pub fn product<T: IntoIterator + Clone, U: IntoIterator + Clone>(
        iter1: &T, iter2: &U,
    ) -> Product<<T as IntoIterator>::IntoIter, <U as IntoIterator>::IntoIter>
    where
        T::Item: Clone,
        U::Item: Clone, <U as IntoIterator>::IntoIter: Clone,
    {
        iproduct!(iter1.clone().into_iter(), iter2.clone().into_iter())
    }
}


pub mod json {
    //pub fn load() {}
    //pub fn loads() {}
    //pub fn dump() {}
    //pub fn dumps() {}
}


pub mod list {
    // TODO generic - should be possible by using associated type to a trait, not generic return
    //pub fn index<T, I>(lst: I, item: T) -> usize
    //where I: IntoIterator<Item = T>
    pub fn index(lst: &[i32], item: i32) -> usize {
        lst.iter().position(|&x| x == item).unwrap()
    }
    // remove?
}


pub mod math {
    // TODO not sure about the efficiency
    // num::integer crate
    pub fn lcm(a: usize, b: usize) -> usize {
        a * b / gcd(a, b)
    }

    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    pub fn sgn<T: Into<i32>>(n: T) -> i8 {  // PartialOrd
        // TODO generic, e.g. using macro https://docs.rs/num-traits/0.2.19/src/num_traits/sign.rs.html#44
        match n.into() {
            n if n > 0 => 1,
            0 => 0,
            _ => -1,
        }
    }
}


pub mod np {
    // ndarray-0.16.1
    use std::cmp::Ord;
    use std::default::Default;
    use std::ops::{Add, AddAssign, Neg, Sub};
    use crate::abs;

    // if not using ndarray crate, implement traits for vec allowing +, -, *, /, ** operations

    // transpose

    pub fn cumsum<T: Iterator<Item = U> + Clone, U: Default + AddAssign + Copy>(input: T) -> Vec<U>
    {
        // a_vec.iter_mut().fold(0, |acc, x| {
        // input.fold(U::default(), |acc, x| acc + x)
        // not sure if possible with fold instead of scan, but scan works
        // btw we do not have to avoid using for loops in Rust, but iterators are neat
        input.scan(U::default(), |acc, x| {
            *acc += x;
            Some(*acc)
        }).collect()
    }

    pub fn diag() {  // https://numpy.org/doc/2.1/reference/generated/numpy.diag.html
        // use https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.diag
        // https://docs.rs/ndarray/latest/ndarray/type.Array.html#method.windows
        todo!()
    }

    pub fn diff_pairs<T>(input: &[(T, T)]) -> Vec<(T, T)>
    where
        T: Sub<Output = T> + Copy,
    {
        // Diff for vector of pairs - applies diff element-wise to each component
        // assuming axis = 0
        // TODO more diff functionality
        if input.len() < 2 {
            return vec![];
        }
        
        input.windows(2)
            .map(|window| {
                let (x1, y1) = window[0];
                let (x2, y2) = window[1];
                (x2 - x1, y2 - y1)
            })
            .collect()
    }

    pub fn logical_not(a: Vec<bool>) -> Vec<bool> {
        let mut res = vec![false; a.len()];
        for (count, aval) in a.iter().enumerate() {
            //res.push(aval.not());
            res[count] = !aval;
        }
        res
    }

    pub fn substract<T: Sub<Output = T>>(a: Vec<T>, b: Vec<T>) -> Vec<T> {
        // element-wise
        // TODO prealocate like in logical_not
        let mut res = vec![];
        for (aval, bval) in a.into_iter().zip(b) {
            res.push(aval - bval);
        }
        res
    }

    pub fn vec_abs<T: Neg<Output = T> + Default + Ord>(a: Vec<T>) -> Vec<T> {
        // element-wise
        // TODO prealocate
        let mut res = vec![];
        for aval in a {
            res.push(abs(aval));
        }
        res
    }

    pub fn prod<T: Iterator<Item = i32> + Clone>(input: T) -> i32 {  // should also be in "math"
        // there is identity trait in some crates to make it more generic
        // https://users.rust-lang.org/t/identity-trait/11922
        // input.clone().fold(1, |acc, x| acc * x)
        input.clone().product::<i32>()
    }

    //pub enum Dtype {
    //    Bool,
    //    Int,
    //    Float,
    //}

    pub fn ones<T: Default + Clone + Add<i32, Output = T>>(size: (usize, usize), _: T) -> Vec<Vec<T>> {  // , dtype: Dtype
        // TODO more dimensions
        //match dtype {
        //    Dtype::Bool => vec![vec![true; size.1]; size.0],
        //    Dtype::Int => vec![vec![1; size.1]; size.0],
        //    Dtype::Float => vec![vec![1.0; size.1]; size.0],
        //}
        //https://docs.rs/num/latest/num/trait.One.html
        vec![vec![T::default() + 1; size.1]; size.0]
    }

    // in case normal sum does not work for list of lists
    //pub fn sum<I>(iterable: &I) -> I::Item
    //where
    //    I: IntoIterator + Clone,
    //    I::Item: Sum,
    //{
    //    iterable.clone().into_iter().sum()
    //}

    pub fn zeros<T: Default + Clone>(size: (usize, usize), _: T) -> Vec<Vec<T>> {
        vec![vec![T::default(); size.1]; size.0]
    }

    pub fn rot90<T: Clone>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
        let rows = matrix.len();
        let cols = matrix[0].len();
        let mut result = vec![vec![matrix[0][0].clone(); rows]; cols];
        for r in 0..rows {
            for (c, item) in result.iter_mut().enumerate().take(cols) {
                item[rows - 1 - r] = matrix[r][c].clone();
            }
        }
        result
    }

    ///numpy.lib.stride_tricks.sliding_window_view
    pub fn sliding_window_view<T: Clone>(
        array: &[Vec<T>], window_shape: (usize, usize),
    ) -> Vec<Vec<Vec<T>>> {
        let (window_height, window_width) = window_shape;
        let mut windows = Vec::new();

        for i in 0..=array.len() - window_height {
            for j in 0..=array[0].len() - window_width {
                let mut window = Vec::new();
                for k in 0..window_height {
                    window.push(array[i + k][j..j + window_width].to_vec());
                }
                windows.push(window);
            }
        }

        windows
    }
}

pub mod pstr {
    // str.format() ?

    pub fn strip(item: &str) -> &str {
        item.trim()
    }

    //pub fn split(item: &str, sep: &str) -> Vec<&str> {
    //    item.()
    //}

    //pub fn startswith() {}

    pub fn rstrip(item: &str) -> &str {
        item.trim_end()
    }

    pub fn lstrip(item: &str) -> &str {
        item.trim_start()
    }

    pub fn count(slf: &str, item: &str) -> usize {
        // TODO this only works if item is one char
        // TODO we should also have this method for lists
        slf.chars().filter(|x| *x == item.chars().next().unwrap()).collect::<Vec<_>>().len()
    }

    //pub fn replace(item: &str, old: &str, new: &str) -> String {
    //    item.replace(old, new)
    //}
}


pub mod re {
    use regex::Regex;
    // Match is Capture I think

    pub fn compile(pattern: &str) -> Regex {
        Regex::new(pattern).unwrap()
    }
    //pub fn match(pattern: Regex, string: &str) -> Match {
    //}
    //pub fn matchall() -> Match {}
    //pub fn search() -> Match {}
    //pub fn findall() -> Vec<String> {}
    //pub fn finditer() -> Iterator<Item=String> {}
    //pub fn sub() -> String {}
    //pub fn subn(pattern, repl, string) -> (String, usize) {}
    //pub fn split() -> Vec<String> {}
    //pub fn escape() {}
}


pub mod scipy {
    use std::collections::VecDeque;

    /// Connected component labeling implementation similar to scipy.ndimage.measurements.label()
    /// 
    /// # Arguments
    /// * `input` - 2D array where non-zero elements are considered as objects to be labeled
    /// * `structure` - Optional connectivity structure (None defaults to 4-connectivity)
    /// 
    /// # Returns
    /// * `(labeled_array, num_features)` - Tuple containing the labeled array and number of components
    pub fn label<T: std::cmp::PartialEq<usize>>(
        input: &[Vec<T>],  // &Vec<Vec<bool>>
        structure: Option<&Vec<Vec<bool>>>,
    ) -> (Vec<Vec<usize>>, usize) {  // TODO: generic return?
        if input.is_empty() || input[0].is_empty() {
            return (vec![], 0);
        }
        
        let rows = input.len();
        let cols = input[0].len();
        let mut labeled = vec![vec![0; cols]; rows];
        let mut current_label = 1;
        
        // Default 4-connectivity structure if none provided
        let default_structure = connectivity_4();
        let connectivity = structure.unwrap_or(&default_structure);
        
        // Get neighbor offsets from structure
        let mut neighbors = Vec::new();
        let center_r = connectivity.len() / 2;
        let center_c = connectivity[0].len() / 2;
        
        for (i, row) in connectivity.iter().enumerate() {
            for (j, &connected) in row.iter().enumerate() {
                if connected && !(i == center_r && j == center_c) {
                    let dr = i as i32 - center_r as i32;
                    let dc = j as i32 - center_c as i32;
                    neighbors.push((dr, dc));
                }
            }
        }

        // Flood fill for each unvisited non-zero element
        for r in 0..rows {
            for c in 0..cols {
                if input[r][c] != 0 && labeled[r][c] == 0 {
                    // Start BFS from this point
                    let mut queue = VecDeque::new();
                    queue.push_back((r, c));
                    labeled[r][c] = current_label;
                    
                    while let Some((cur_r, cur_c)) = queue.pop_front() {
                        // Check all neighbors
                        for &(dr, dc) in &neighbors {
                            let new_r = cur_r as i32 + dr;
                            let new_c = cur_c as i32 + dc;
                            
                            // Check bounds
                            if new_r >= 0 && new_r < rows as i32 && new_c >= 0 && new_c < cols as i32 {
                                let nr = new_r as usize;
                                let nc = new_c as usize;
                                
                                // If neighbor is non-zero and unlabeled
                                if input[nr][nc] != 0 && labeled[nr][nc] == 0 {
                                    labeled[nr][nc] = current_label;
                                    queue.push_back((nr, nc));
                                }
                            }
                        }
                    }
                    
                    current_label += 1;
                }
            }
        }
        
        (labeled, current_label - 1)
    }

    /// 4-connectivity structure (default)
    pub fn connectivity_4() -> Vec<Vec<bool>> {
        vec![
            vec![false, true, false],
            vec![true, true, true],
            vec![false, true, false],
        ]
    }

    /// 8-connectivity structure
    pub fn connectivity_8() -> Vec<Vec<bool>> {
        vec![
            vec![true, true, true],
            vec![true, true, true],
            vec![true, true, true],
        ]
    }
}


// pub mod statistics


// not really python lib, but needed for negative indexing (not used now)
pub struct ReverseIndex(pub usize);

impl<T> Index<ReverseIndex> for [T] {
    type Output = T;
    
    fn index(&self, idx: ReverseIndex) -> &T {
        let _len = self.len();
        &self[_len - 1 - idx.0]
    }
}

impl<T> IndexMut<ReverseIndex> for [T] {
    fn index_mut(&mut self, idx: ReverseIndex) -> &mut T {
        let _len = self.len();
        &mut self[_len - 1 - idx.0]
    }
}
