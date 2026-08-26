"""Experiments for "Goldfarb/Idnani Revisited: Invariants, Implementation, and Certificates".

The paper that these support is a derivation: every claim in it is proved rather
than measured. The experiments here exist for the two jobs that a proof cannot do
on its own.

The first is to check that the object proved about is the object that runs. A
derivation states identities the solver is supposed to satisfy; whether the
shipped code satisfies them is a separate question, and one worth answering
numerically because the answer is not obviously yes -- the identities hold in
exact arithmetic, and the code runs in floating point with a hand-rolled packed
triangular layout and BLAS calls that write through to their arguments' buffers.
``experiment_qp_identities`` answers it for every identity in the paper.

The second is to measure the quantities the paper reasons about but cannot
predict: how often a guessed active set is certifiable, how many repairs it
takes, what a warm-started solve actually costs.

Every module writes graphs/ and tables/ that the paper inputs, following the same
conventions as cg/, rmt/ and nncg_note/: run as a module from experiment/, honour
EXPERIMENT_SMOKE and EXPERIMENT_OUT via common.util.runner.
"""
