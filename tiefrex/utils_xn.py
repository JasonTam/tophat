import tensorflow as tf
import itertools as it
from typing import Dict, Iterable


def preset_interactions(fields_d: Dict[str, Iterable[str]],
                        interaction_type: str='inter',
                        max_order: int=2
                        ) -> Iterable[frozenset]:
    """
    Convenience feature interaction planner for common interaction types
    :param fields_d: dictionary of group_name to iterable of feat_names that 
        belong to that group
        ex) `fields_d = {'user': {'gender', 'age'}, 
                         'item': {'brand', 'pcat', 'price'}}`
    :param interaction_type: preset type
    :param max_order: max order of interactions
        if `interaction_type` is `inter`, `max_order` should
        not be larger than 3
    :return: Iterable of interaction sets
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
                     emb_d: Dict[str, tf.Tensor]
                     ) -> Dict[frozenset, tf.Tensor]:
    """ Returns the element-wise product of embeddings (with node-reuse)
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
    """ Reduce nodes (typically element-wise sums) with addition
        (effectively yields dot product)
    """
    if len(xn_nodes):
        # Reduce nodes of order > 1
        contrib_dot = tf.add_n([
            tf.reduce_sum(node, 1, keep_dims=False)
            for s, node in xn_nodes.items() if len(s) > 1
        ], name='contrib_dot')
    else:
        contrib_dot = tf.zeros(None, name='contrib_dot')
    return contrib_dot


def kernel_via_xn_sets(interaction_sets: Iterable[frozenset],
                       emb_d: Dict[str, tf.Tensor]) -> tf.Tensor:
    """
    Computes arbitrary order interaction terms
    Reuses lower order terms
    
    Differs from typical HOFM as we will reuse lower order embeddings
        (not actually sure if this is OK in terms of expressiveness)
        In theory, we're supposed to use a new param matrix for each order
            Much like how we use a bias param for order=1
    :param interaction_sets: interactions to create nodes for
    :param emb_d: dictionary of embedding tensors
        NOTE: would include an additional `order` key to match HOFM literature
    :return: 
    
    References:
        Blondel, Mathieu, et al. "Higher-Order Factorization Machines." 
            Advances in Neural Information Processing Systems. 2016.
    """
    # TODO: for now we assume that all dependencies of previous order are met
    return kernel_via_xn_muls(muls_via_xn_sets(interaction_sets, emb_d))
