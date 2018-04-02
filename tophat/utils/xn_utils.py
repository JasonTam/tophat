import tensorflow as tf
import itertools as it
from typing import Dict, Iterable, Any, Mapping


def preset_interactions(fields_d: Dict[Any, Iterable[str]],
                        interaction_type: str='inter',
                        max_order: int=2
                        ) -> Iterable[frozenset]:
    """Convenience feature interaction planner for common interaction types
    
    Args:
        fields_d: Dictionary of group_name to iterable of feat_names that
            belong to that group
            ex) `fields_d = {'user': {'gender', 'age'},
            'item': {'brand', 'pcat', 'price'}}`
        interaction_type: One of {'intra', 'inter'}

            - intra: Include interactions for all features agnostic of group
            - inter: Include only interactions between different groups
              (this will mimic the formulation of [1]_)
        max_order: Max order of interactions
            If `interaction_type` is `inter`, `max_order` should be no larger
            than 3

    Returns:
        Iterable of interaction sets (set will contain feature names)

    References:
        .. [1] Kula, Maciej. "Metadata embeddings for user and item cold-start
           recommendations." arXiv preprint arXiv:1507.08439 (2015).

    """

    if interaction_type == 'intra':  # includes intra field
        feature_pairs = it.chain(
            *(it.combinations(
                it.chain(*fields_d.values()), order)
                for order in range(2, max_order + 1)))
    elif interaction_type == 'inter':  # only inter field
        feature_pairs = it.chain(
            *(it.product(*fields)
              for fields in it.chain(
                *(it.combinations(fields_d.values(), o)
                  for o in range(2, max_order + 1)))
              ))
    else:
        raise ValueError

    return map(frozenset, feature_pairs)


def muls_via_xn_sets(interaction_sets: Iterable[frozenset],
                     emb_d: Mapping[Any, tf.Tensor]
                     ) -> Dict[frozenset, tf.Tensor]:
    """ Computes element-wise product of embeddings (with node-reuse)
    
    Args:
        interaction_sets: Interactions to process
        emb_d: Dictionary of embedding tensors

    Returns:
        Dictionary of multiplication ops

    """

    interaction_sets = list(interaction_sets)
    # Populate xn nodes with single terms (order 1)
    xn_nodes: Dict[frozenset, tf.Tensor] = {
        frozenset({k}): v for k, v in emb_d.items()}
    unq_orders = set(map(len, interaction_sets))
    if unq_orders:
        for order in range(min(unq_orders), max(unq_orders) + 1):
            with tf.name_scope(f'xn_order_{order}'):
                for xn in interaction_sets:
                    if len(xn) == order:
                        xn_l = list(xn)
                        xn_nodes[xn] = tf.multiply(
                            # cached portion (if ho)
                            xn_nodes[frozenset(xn_l[:-1])],
                            # last as new term
                            xn_nodes[frozenset({xn_l[-1]})],
                            name='X'.join(xn)
                        )
    return xn_nodes


def kernel_via_xn_muls(xn_nodes: Dict[frozenset, tf.Tensor]) -> tf.Tensor:
    """Reduce nodes (typically element-wise sums) with addition
    (effectively yields dot product)
    
    Args:
        xn_nodes: Interaction nodes

    Returns:
        Reduced interactions
    """

    if len(xn_nodes):
        # Reduce nodes of order > 1
        contrib_dot = tf.add_n([
            tf.reduce_sum(node, 1, keepdims=False)
            for s, node in xn_nodes.items() if len(s) > 1
        ], name='contrib_dot')
    else:
        contrib_dot = tf.zeros(None, name='contrib_dot')
    return contrib_dot


def kernel_via_xn_sets(interaction_sets: Iterable[frozenset],
                       emb_d: Mapping[Any, tf.Tensor]) -> tf.Tensor:
    """Computes arbitrary order interaction terms
    Reuses lower order terms

    Differs from typical HOFM [2]_ as we will reuse lower order embeddings
    (not actually sure if this is OK in terms of expressiveness).
    In theory, we're supposed to use a new param matrix for each order
    Much like how we use a bias param for order=1
            
    Args:
        interaction_sets: Interactions to create nodes for
        emb_d: Dictionary of embedding tensors

    Returns:
        Interactions that have passed through the kernel
        
    References:
        .. [2] Blondel, Mathieu, et al. "Higher-Order Factorization Machines."
           Advances in Neural Information Processing Systems. 2016.

    """
    # TODO: for now we assume that all dependencies of previous order are met
    return kernel_via_xn_muls(muls_via_xn_sets(interaction_sets, emb_d))
