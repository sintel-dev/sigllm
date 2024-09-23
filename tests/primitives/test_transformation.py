import unittest

import numpy as np
import pytest

from sigllm.primitives.transformation import (
    Float2Scalar, Scalar2Float, _from_string_to_integer, format_as_integer, format_as_string,)


class FormatAsStringTest(unittest.TestCase):

    def test_format_as_string_default(self):

        data = np.array([[
            1, 2, 3, 4, 5
        ]])
        expected = np.array([
            '1,2,3,4,5'
        ])

        output = format_as_string(data)

        np.testing.assert_array_equal(output, expected)

    def test_format_as_string_space(self):

        data = np.array([[
            1, 2, 3, 4, 5
        ]])
        expected = np.array([
            '1 , 2 , 3 , 4 , 5'
        ])

        output = format_as_string(data, space=True)

        np.testing.assert_array_equal(output, expected)

    def test_format_as_string_sep(self):

        data = np.array([[
            1, 2, 3, 4, 5
        ]])
        expected = np.array([
            '1.2.3.4.5'
        ])

        output = format_as_string(data, sep='.')

        assert output == expected

    def test_format_as_string_decimal(self):

        data = np.array([[
            100, 200, 300, 400, 500
        ]])
        expected = np.array([
            '100,200,300,400,500'
        ])

        output = format_as_string(data)

        assert output == expected


class FromStringToIntegerTest(unittest.TestCase):

    def test__from_string_to_integer_default(self):

        data = '1,2,3,4,5'

        expected = np.array([
            1, 2, 3, 4, 5
        ])

        output = _from_string_to_integer(data)

        np.testing.assert_equal(output, expected)

    def test__from_string_to_integer_trunc_all(self):

        data = '1,2,3,4,5'

        expected = np.array([
            1, 2, 3, 4, 5
        ])

        output = _from_string_to_integer(data, trunc=5)

        np.testing.assert_equal(output, expected)

    def test__from_string_to_integer_trunc_two(self):

        data = '134,252,3,412,51'

        expected = np.array([
            134, 252
        ])

        output = _from_string_to_integer(data, trunc=2)

        np.testing.assert_equal(output, expected)

    def test__from_string_to_integer_ignore(self):

        data = '1 , 2!e , 3,4,5'

        expected = np.array([
            1, 3, 4, 5
        ])

        output = _from_string_to_integer(data, errors='ignore')

        np.testing.assert_equal(output, expected)

    def test__from_string_to_integer_filter(self):

        data = '1 , 2!e , 3,4,5'

        expected = np.array([
            1, 2, 3, 4, 5
        ])

        output = _from_string_to_integer(data, errors='filter')

        np.testing.assert_equal(output, expected)

    def test__from_string_to_integer_coerce(self):

        data = '1 , 2!e , 3,4,5'

        expected = np.array([
            1, np.nan, 3, 4, 5
        ])

        output = _from_string_to_integer(data, errors='coerce')

        np.testing.assert_equal(output, expected)

    def test__from_string_to_integer_raise(self):

        data = '1 , 2!e , 3,4,5'

        with pytest.raises(ValueError):
            _from_string_to_integer(data, errors='raise')

    def test__from_string_to_integer_error(self):

        data = '1 , 2 , 3,4,5'

        with pytest.raises(KeyError):
            _from_string_to_integer(data, errors='unknown')


def test_format_as_integer_one():
    data = ['1,2,3,4,5']

    with pytest.raises(ValueError):
        format_as_integer(data)


def test_format_as_integer_list():
    data = [['1,2,3,4,5']]

    expected = np.array([[[
        1, 2, 3, 4, 5
    ]]])

    output = format_as_integer(data)

    np.testing.assert_equal(output, expected)


def test_format_as_integer_2d_shape_mismatch():
    data = [['1,2,3,4,5'], ['1, 294., 3 , j34,5'], ['!232, 23,3,4,5']]

    expected = np.array([
        [np.array([1., 2, 3, 4, 5])],
        [np.array([1., 3, 5])],
        [np.array([23., 3, 4, 5])]
    ], dtype=object)

    output = format_as_integer(data)

    for out, exp in list(zip(output, expected)):
        for o, e in list(zip(out, exp)):
            np.testing.assert_equal(o, e)


def test_format_as_integer_2d_trunc():
    data = [['1,2,3,4,5'], ['1,294.,3,j34,5'], ['!232, 23,3,4,5']]

    expected = np.array([
        [[1, 2]],
        [[1, 3]],
        [[23, 3]]
    ])

    output = format_as_integer(data, trunc=2)

    np.testing.assert_equal(output, expected)


def test_format_as_integer_3d():
    data = [
        ['1,2,3,4,5', '6,7,8,9,10'],
        ['11,12,13,14,15', '16,17,18,19,20']
    ]

    expected = np.array([
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]
    ])

    output = format_as_integer(data)

    np.testing.assert_equal(output, expected)


class Float2ScalarTest(unittest.TestCase):

    def test_transform_default(self):
        converter = Float2Scalar()

        data = np.array([
            1.05, 2., 3.1, 4.8342, 5, 0
        ])
        expected = np.array([
            105, 200, 310, 483, 500, 0
        ])
        print(converter)

        converter.fit(data)
        output, minimum, decimal = converter.transform(data)

        assert converter.decimal == 2
        assert converter.rescale is True
        assert converter.minimum == 0

        np.testing.assert_array_equal(output, expected)

    def test_transform_decimal_zero(self):
        converter = Float2Scalar(decimal=0)

        data = np.array([
            1.05, 2., 3.1, 4.8342, 5, 0
        ])
        expected = np.array([
            1, 2, 3, 4, 5, 0
        ])

        converter.fit(data)
        output, minimum, decimal = converter.transform(data)

        assert converter.decimal == 0
        assert converter.rescale is True
        assert converter.minimum == 0

        np.testing.assert_array_equal(output, expected)

    def test_transform_minimum_not_zero(self):
        converter = Float2Scalar()

        data = np.array([
            1.05, 2., 3.1, 4.8342, 5
        ])
        expected = np.array([
            0, 95, 204, 378, 395
        ])

        converter.fit(data)
        output, minimum, decimal = converter.transform(data)

        assert converter.decimal == 2
        assert converter.rescale is True
        assert converter.minimum == 1.05

        np.testing.assert_array_equal(output, expected)

    def test_transform_rescale_false(self):
        converter = Float2Scalar(rescale=False)

        data = np.array([
            1.05, 2., 3.1, 4.8342, 5
        ])
        expected = np.array([
            105, 200, 310, 483, 500
        ])

        converter.fit(data)
        output, minimum, decimal = converter.transform(data)

        assert converter.decimal == 2
        assert converter.rescale is False
        assert converter.minimum == 1.05

        np.testing.assert_array_equal(output, expected)

    def test_transform_negative(self):
        converter = Float2Scalar()

        data = np.array([
            1.05, 2., 3.1, 4.8342, 5, -2
        ])
        expected = np.array([
            305, 400, 509, 683, 700, 0
        ])

        converter.fit(data)
        output, minimum, decimal = converter.transform(data)

        assert converter.decimal == 2
        assert converter.rescale is True
        assert converter.minimum == -2

        np.testing.assert_array_equal(output, expected)

    def test_transform_fit_different(self):
        converter = Float2Scalar()

        data = np.array([
            1.05, 2., 3.1, 4.8342, 5
        ])
        expected = np.array([
            55, 150, 260, 433, 450
        ])

        converter.fit([7, 3, 0.5])
        output, minimum, decimal = converter.transform(data)

        assert converter.decimal == 2
        assert converter.rescale is True
        assert converter.minimum == 0.5

        np.testing.assert_array_equal(output, expected)


class Scalar2FloatTest(unittest.TestCase):

    def test_transform_default(self):
        converter = Scalar2Float()

        data = np.array([
            105, 200, 310, 483, 500, 0
        ])
        expected = np.array([
            1.05, 2., 3.1, 4.83, 5, 0
        ])

        output = converter.transform(data)

        np.testing.assert_array_equal(output, expected)

    def test_transform_decimal_zero(self):
        converter = Scalar2Float()

        data = np.array([
            1, 2, 3, 4, 5, 0
        ])
        expected = np.array([
            1., 2., 3., 4., 5., 0.
        ])

        output = converter.transform(data, decimal=0)

        np.testing.assert_array_equal(output, expected)

    def test_transform_minimum_not_zero(self):
        converter = Scalar2Float()

        data = np.array([
            105, 200, 310, 483, 500, 0
        ])
        expected = np.array([
            0.05, 1., 2.1, 3.83, 4., -1.
        ])

        output = converter.transform(data, minimum=-1)

        np.testing.assert_allclose(output, expected)


def test_float2scalar_scalar2float_integration():
    decimal = 2
    rescale = True

    float2scalar = Float2Scalar(decimal, rescale)

    data = np.array([
        1.05, 2., 3.1, 4.8342, 5, -1
    ])

    expected = np.array([
        1.05, 2., 3.10, 4.83, 5., -1.
    ])

    float2scalar.fit(data)
    transformed, minimum, decimal = float2scalar.transform(data)

    scalar2float = Scalar2Float()

    output = scalar2float.transform(transformed, minimum, decimal)

    np.testing.assert_allclose(output, expected, rtol=1e-2)
