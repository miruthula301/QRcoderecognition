# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 17:35:51 2021

@author: Miruthula
"""

import qr
import numpy as np

from nose.tools import assert_equals


def test_qrCodeMatrix():
    """QR code matrices should have the specified size"""

    data = "kg823nASp7"
    matrix = qr.qrCodeMatrix(data)
    assert_equals((qr.IMAGE_SIZE, qr.IMAGE_SIZE), matrix.shape)


def test_dataToVector():
    """One-hot encoded vectors should have the proper shape"""

    for ind, char in enumerate(qrcodes.CHARACTER_SET):
        data = char + "random_suffix"
        vec = qr.dataToVector(data)

        assert_equals(len(qr.CHARACTER_SET), len(vec))
        assert_equals(1.0, sum(vec))
        assert_equals(1.0, vec[ind])


def test_getRandomBatch():
    """Batch tensors should have the correct shape"""

    batchSize = 22
    X, y = qr.getRandomBatch(batchSize)

    y = np.asarray(y)

    assert_equals((batchSize, qr.IMAGE_SIZE, qr.IMAGE_SIZE, 1), X.shape)
    assert_equals((batchSize, len(qr.CHARACTER_SET)), y.shape)


if __name__ == "__main__":
    import nose
    nose.run()