//! Items related to expressions.

use fnv::{FnvHashMap, FnvHashSet};
use petgraph::{self, Incoming, Outgoing};
use petgraph::visit::{Dfs, EdgeRef, Topo};
use rand::Rng;
use std::mem;

/// A node/expression type that can be evaluated to a single value.
pub trait Evaluate<E> {
    /// The type of the value produced by the node type.
    type Value;
    /// Evaluate this node in terms of the given inputs to produce the given value.
    fn evaluate(&self, inputs: &[&Self::Value], env: &E) -> Self::Value;
}

/// The directed graph type used to represent an expression.
///
/// Each node within the graph is either a `Function` or a  `Terminal`. `Function`s are branch
/// nodes that have one or more input expressions stored on `Incoming` edges. `Terminal`s are
/// leaf nodes, often either inputs to the expression or a constant value.
pub type DiGraph<N> = petgraph::graph::DiGraph<N, (), u32>;

/// The node index type used within the expr DiGraph type.
pub type NodeIndex = petgraph::graph::NodeIndex<u32>;

/// The edge index type used within the expr DiGraph type.
pub type EdgeIndex = petgraph::graph::EdgeIndex<u32>;

/// Generate a random expression.
///
/// This function will randomly choose between using `gen::full_tree` and `gen::grow_tree`.
pub fn gen<R, N>(rng: &mut R, max_depth: u32) -> DiGraph<N>
where
    R: Rng,
    N: gen::Node,
    N: ::std::fmt::Debug,
{
    match rng.gen_range(0, 2) {
        0 => gen::full_tree(rng, max_depth),
        1 => gen::grow_tree(rng, max_depth),
        _ => unreachable!(),
    }
}

/// Evaluate the given expression.
pub fn eval<N, E>(expr: &DiGraph<N>, env: &E) -> FnvHashMap<NodeIndex, N::Value>
where
    N: Evaluate<E>,
    N: ::std::fmt::Debug + gen::Arity,
{
    let mut topo = Topo::new(expr);
    let mut evaluated = FnvHashMap::with_capacity_and_hasher(expr.node_count(), Default::default());
    while let Some(nx) = topo.next(&expr) {
        let value = {
            let inputs = expr.edges_directed(nx, Incoming)
                .map(|e| &evaluated[&e.source()])
                .collect::<Vec<_>>();
            if inputs.len() != expr[nx].arity() as _ {
                panic!("node disparity between number of inputs {} and arity {}\n{:#?}",
                       inputs.len(), expr[nx].arity(), expr);
            }
            N::evaluate(&expr[nx], &inputs[..], env)
        };
        evaluated.insert(nx, value);
    }
    evaluated
}

/// Close the subtree whose root is at the given node into a new directed graph.
pub fn clone_subtree<N>(tree: &DiGraph<N>, subtree_root: NodeIndex) -> DiGraph<N>
where
    N: Clone,
{
    // Initialise the graph.
    let mut subtree = petgraph::graph::DiGraph::<N, (), u32>::new();

    // Add the root without adding any outgoing edges.
    let subtree_root = subtree.add_node(tree[subtree_root].clone());

    // For all others, add both nodes and their edges into their parent.
    let mut curr = vec![subtree_root];
    let mut next = vec![];
    while !curr.is_empty() {
        for a in curr.drain(..) {
            for e in tree.edges_directed(a, Incoming) {
                let b = subtree.add_node(tree[e.source()].clone());
                subtree.add_edge(b, a, e.weight().clone());
                next.push(b);
            }
        }
        mem::swap(&mut curr, &mut next);
    }

    subtree
}

/// Replace the node at `nx` with the given subtree.
pub fn replace_subtree<N>(tree: &DiGraph<N>, nx: NodeIndex, subtree: DiGraph<N>) -> DiGraph<N>
where
    N: Clone,
{
    // Collect the nodes we don't want to include in the new tree..
    let mut disclude = FnvHashSet::default();
    let mut dfs = Dfs::new(tree, nx);
    while let Some(n) = dfs.next(tree) {
        disclude.insert(n);
    }

    // Initialise the new tree by cloning the original, discluding the node's subtree.
    let mut new_tree = tree.filter_map(
        |nx, nw| if disclude.contains(&nx) { None } else { Some(nw.clone()) },
        |_, ew| Some(ew.clone()),
    );

    // Attach the root of the subtree.
    let root_old = NodeIndex::new(0);
    let root_new = new_tree.add_node(subtree[root_old].clone());
    for e in tree.edges_directed(nx, Outgoing) {
        new_tree.add_edge(root_new, e.target(), e.weight().clone());
    }

    // Clone the rest of the subtree.
    let mut curr = vec![(root_old, root_new)];
    let mut next = vec![];
    while !curr.is_empty() {
        for (dst_old, dst_new) in curr.drain(..) {
            for e in subtree.edges_directed(dst_old, Incoming) {
                let src_old = e.source();
                let src_new = new_tree.add_node(subtree[src_old].clone());
                new_tree.add_edge(src_new, dst_new, e.weight().clone());
                next.push((src_old, src_new));
            }
        }
        mem::swap(&mut curr, &mut next);
    }

    new_tree
}

/// Trim the given tree to the given maximum depth.
///
/// Also replaces any leaf nodes with terminals if necessary (determined by whether or not their
/// arity is already `0`).
pub fn trim_tree_to_depth<R, N>(rng: &mut R, tree: &mut DiGraph<N>, max_depth: u32)
where
    R: Rng,
    N: gen::Arity + gen::Terminal,
{
    // If the tree is empty or the max depth requires an empty tree, return early.
    if max_depth == 0 {
        tree.clear();
    }
    if tree.node_count() == 0 {
        return;
    }

    // If the max depth only expects a root, ensure it is a terminal.
    if max_depth == 1 {
        if tree.node_count() != 1 || tree[NodeIndex::new(0)].arity() != 0 {
            tree.clear();
            tree.add_node(gen::Terminal::generate(rng));
        }
        return;
    }

    println!("tree pre-trim node count: {}", tree.node_count());

    // Loop in BFS order and determine what to keep.
    let root = NodeIndex::new(0);
    let mut next = vec![root];
    let mut curr = vec![];
    let mut depth = 0;
    let mut keep = FnvHashSet::default();
    while depth == 0 || !curr.is_empty() {
        mem::swap(&mut curr, &mut next);
        if depth >= max_depth {
            break;
        }
        for a in curr.drain(..) {
            keep.insert(a);
            for e in tree.edges_directed(a, Incoming) {
                next.push(e.source());
            }
        }
        depth += 1;
    }

    // Trim the tree.
    tree.retain_nodes(|_, nx| keep.contains(&nx));

    if tree.node_count() == 0 {
        panic!("tree was trimmed down to nothing:\nmax_depth: {}\nkeep: {:#?}\ncurr: {:#?}", max_depth, keep, curr);
    }
    assert!(tree.node_count() > 0, "tree was trimmed down to nothing");

    // Replace the leaves with terminals if necessary.
    for nx in curr {
        if tree[nx].arity() > 0 {
            tree[nx] = gen::Terminal::generate(rng);
        }
    }
}

/// Functions for generating program graphs.
pub mod gen {
    use petgraph::graph::Graph;
    use rand::Rng;
    use std::mem;
    use super::DiGraph;

    /// Node types that know their number of inputs / arguments.
    pub trait Arity {
        /// The number of arguments to the node.
        ///
        /// Function nodes will return 1 or more. Terminal nodes will return 0.
        fn arity(&self) -> u32;
    }

    /// Function types that may be generated for use within an expression.
    pub trait Function: Arity {
        /// Generate an instance of this Function type.
        fn generate<R>(rng: &mut R) -> Self where R: Rng;
    }

    /// Terminal types that may be generated for use within an expression.
    pub trait Terminal {
        /// Generate an instance of this Terminal type.
        fn generate<R>(rng: &mut R) -> Self where R: Rng;
    }

    /// Expression nodes that may be generated.
    pub trait Node: Function + Terminal {}

    impl<T> Node for T where T: Function + Terminal {}

    /// Generate an expression tree using the "full" approach.
    ///
    /// All branches will end with `Terminal`s at the given `max_depth`, while all other nodes
    /// will be `Function`s.
    ///
    /// Inputs to a `Function` node can be found by iterating over the incoming edges.
    ///
    /// The "root" or "output" of the expression will be at the node with index `0`.
    ///
    /// Nodes are generated in breadth-first order.
    pub fn full_tree<R, N>(rng: &mut R, depth: u32) -> DiGraph<N>
    where
        R: Rng,
        N: Node,
    {
        // The graph that will contain the tree.
        let mut g = Graph::new();

        // Handle the low depth cases.
        match depth {
            0 => return g,
            1 => {
                g.add_node(Terminal::generate(rng));
                return g;
            }
            _ => ()
        }

        // Fill each depth level one at a time.
        let mut curr = vec![g.add_node(Function::generate(rng))];
        let mut next = vec![];
        for _ in 1..depth {
            for a in curr.drain(..) {
                for _ in 0..g[a].arity() {
                    let b = g.add_node(Function::generate(rng));
                    g.add_edge(b, a, ());
                    next.push(b);
                }
            }
            mem::swap(&mut curr, &mut next);
        }

        // Generate the final depth of nodes.
        for a in curr.drain(..) {
            for _ in 0..g[a].arity() {
                let b = g.add_node(Terminal::generate(rng));
                g.add_edge(b, a, ());
            }
        }

        g
    }

    /// Generate an expression tree using the "grow" approach.
    ///
    /// The tree will be "grown" by randomly generating functions and terminals for each node
    /// until the maximum depth is reached.
    pub fn grow_tree<R, N>(rng: &mut R, depth: u32) -> DiGraph<N>
    where
        R: Rng,
        N: Node,
    {
        // The graph that will contain the tree.
        let mut g = Graph::new();

        // Handle the low depth cases.
        match depth {
            0 => return g,
            1 => {
                g.add_node(Terminal::generate(rng));
                return g;
            }
            _ => ()
        }

        // Fill each depth level one at a time.
        let mut curr = vec![g.add_node(Function::generate(rng))];
        let mut next = vec![];
        for d in 1..depth {
            for a in curr.drain(..) {
                for _ in 0..g[a].arity() {
                    let node = match rng.gen_range(0, depth - d) {
                        0 => Terminal::generate(rng),
                        _ => Function::generate(rng),
                    };
                    let b = g.add_node(node);
                    g.add_edge(b, a, ());
                    next.push(b);
                }
            }
            mem::swap(&mut curr, &mut next);
        }

        g
    }
}
