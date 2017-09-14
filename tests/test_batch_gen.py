import unittest

import numpy as np
from tiefrex.data import TrainDataLoader
from tiefrex.nets import EmbeddingMap, BilinearNet, BilinearNetWithNum, BilinearNetWithNumFC
from tiefrex.core import FactModel
from tiefrex.config_parser import Config

from tiefrex import naive_sampler


class TestBG(unittest.TestCase):
    def setUp(self):
        self.config = Config('tiefrex/config/config_test.py')
        self.train_data_loader = TrainDataLoader(self.config)

        self.embedding_map = EmbeddingMap(self.train_data_loader, embedding_dim=16)

        self.model = FactModel(net=BilinearNetWithNum(
            embedding_map=self.embedding_map,
            num_meta=self.train_data_loader.num_meta)
        )
        self.batch_size = self.model.net.embedding_map.data_loader.batch_size
        self.sampler = naive_sampler.PairSampler(
            self.train_data_loader,
            self.model.input_pair_d,
            batch_size=self.batch_size,
            method='uniform',
        )
        self.feed_dict_gen = iter(self.sampler)

    def test_sample(self):
        pos_xns = {tuple(xn) for xn in self.train_data_loader.interactions_df.values}

        for _ in range(100):
            z = next(self.feed_dict_gen)
            user_ind_batch = z[self.model.input_pair_d['user.user_id']]
            user_id_batch = [self.train_data_loader.cats_d['user_id'][ii] for ii in user_ind_batch]
            pos_item_ind_batch = z[self.model.input_pair_d['pos.item_id']]
            pos_item_id_batch = [self.train_data_loader.cats_d['item_id'][ii] for ii in pos_item_ind_batch]

            # All sampled positive interactions should actually be positive interactions
            assert all(xn in pos_xns for xn in zip(user_id_batch, pos_item_id_batch))

            # Numerical features from the test data should be the same as item_id + [0.1,0.2,0.3,0.4]
            # (this is just how the test data is)
            expected_num_feats = np.tile(np.arange(1, 5)[None, :]*0.1, (self.batch_size, 1))\
                                 + np.array(pos_item_id_batch)[:, None]
            pos_item_num_batch = z[self.model.input_pair_d['pos.item_num_feats']]
            np.testing.assert_array_equal(pos_item_num_batch, expected_num_feats)


if __name__ == '__main__':
    unittest.main()

