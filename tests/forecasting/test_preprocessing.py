import unittest

import numpy as np
from pytest import fixture

from sigllm.forecasting.preprocessing import Signal2String

class Signal2StringTest(unittest.TestCase):

    def test_transform_default(self):
        converter = Signal2String()

        data = np.array([
            1, 2, 3, 4, 5
        ])
        expected = '1,2,3,4,5'
        
        output = converter.transform(data)

        assert converter.sep == ','
        assert converter.space == False
        assert converter.decimal == 0
        assert converter.rescale == False

        assert output == expected

    
    def test_transform_space(self):
        converter = Signal2String(space=True)

        data = np.array([
            1, 2, 3, 4, 5
        ])
        expected = '1 , 2 , 3 , 4 , 5'
        
        output = converter.transform(data)

        assert converter.space == True
        assert output == expected

    
    def test_transform_sep(self):
        converter = Signal2String(sep='.')

        data = np.array([
            1, 2, 3, 4, 5
        ])
        expected = '1.2.3.4.5'
        
        output = converter.transform(data)

        assert converter.sep == '.'
        assert output == expected


    def test_transform_sep(self):
        converter = Signal2String(decimal=2)

        data = np.array([
            1, 2, 3, 4, 5
        ])
        expected = '100,200,300,400,500'
        
        output = converter.transform(data)

        assert converter.decimal == 2
        assert output == expected

    
    def test_transform_rescale(self):
        converter = Signal2String(rescale=True)

        data = np.array([
            -1, 2, 3, 4, 5
        ])
        expected = '0,3,4,5,6'
        
        output = converter.transform(data)

        assert converter.rescale == True
        assert output == expected


    def test_reverse_transform_default(self):
        converter = Signal2String()

        data = '1,2,3,4,5'
        expected = np.array([
            1, 2, 3, 4, 5
        ])
        
        output = converter.reverse_transform(data)

        assert converter.sep == ','
        assert converter.space == False
        assert converter.decimal == 0
        assert converter.rescale == False

        np.testing.assert_equal(output, expected)
