import unittest
from dataset import MyDataModule

class MyDataModuleUT(unittest.TestCase):

    @unittest.skip("disabled")
    def test_speed(self):
        m = MyDataModule()
        dl = m.train_dataloader()
        for x, y in dl:
            self.assertTrue('KK' in m.cols)
            self.assertTrue('KK_v' in m.cols)

    def test_test_shape(self):
        m = MyDataModule()
        dl = m.test_dataloader()
        print("test_dl.length", len(dl))
        self.assertGreater(len(dl), 0)
        for x, y in dl:
            self.assertEqual(y.shape, (m.test_batch_size, m.test_L, len(m.cols)))
            break
