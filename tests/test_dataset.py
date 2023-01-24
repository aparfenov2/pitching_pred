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
        m = MyDataModule(
            fn_train = "tests/data/sin_train.dat",
            fn_test = "tests/data/sin_train.dat",
            test_only=True,
            test_config = {
                "L": 100
            }
        )
        dl = m.test_dataloader()
        # print("test_dl.length", len(dl))
        self.assertGreater(len(dl), 0)
        for batch in dl[0]:
            y = batch["y"]
            self.assertEqual(y.shape, (m.test_batch_size, 100, len(m.cols)))
            break

    def test_inst(self):
        MyDataModule(
            fn_train = "tests/data/sin_train.dat",
            fn_test = "tests/data/sin_train.dat",
            train_config={
                "L": 1000,
                "stride": 0.5,
                "multiply": 10
            }
        )
