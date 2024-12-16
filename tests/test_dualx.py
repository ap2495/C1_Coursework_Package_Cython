import pytest
import numpy as np
import re
from dual_autodiff_x.dual import Dual_x
from dual_autodiff_x.dual import Dual_x_array

# Implement a test function for every method in dual
# For scalars


# Tests for Dual_x class
def test_init():
    # Test initialization of Dual_x with real and dual parts
    test_number = Dual_x(5.0, 7.0)
    assert test_number.real == 5.0
    assert test_number.dual == 7.0

def test_add():
    # Test addition of two Dual_x numbers
    test_number1 = Dual_x(5.0, 7.0)
    test_number2 = Dual_x(3.0, 2.0)
    test_sum = test_number1 + test_number2
    assert test_sum.real == 8.0
    assert test_sum.dual == 9.0

def test_sub():
    # Test subtraction of two Dual_x numbers
    test_number1 = Dual_x(5.0, 7.0)
    test_number2 = Dual_x(3.0, 2.0)
    test_diff = test_number1 - test_number2
    assert test_diff.real == 2.0
    assert test_diff.dual == 5.0

def test_mul():
    # Test multiplication of two Dual_x numbers
    test_number1 = Dual_x(5.0, 7.0)
    test_number2 = Dual_x(3.0, 2.0)
    test_prod = test_number1 * test_number2
    expected_real = 5.0 * 3.0
    expected_dual = 5.0 * 2.0 + 7.0 * 3.0
    assert test_prod.real == expected_real
    assert test_prod.dual == expected_dual

def test_pow():
    # Test power operation on a Dual_x number
    test_number = Dual_x(5.0, 1.0)
    power = test_number ** 3
    expected_real = 5.0 ** 3
    expected_dual = 3 * 5.0 ** (3 - 1) * 1.0
    assert power.real == expected_real
    assert power.dual == expected_dual

def test_sin():
    # Test sine operation on a Dual_x number
    test_number = Dual_x(5.0, 1.0)
    sin_test = test_number.sin()
    expected_real = np.sin(5.0)
    expected_dual = np.cos(5.0) * 1.0
    assert sin_test.real == pytest.approx(expected_real, rel=1e-6)
    assert sin_test.dual == pytest.approx(expected_dual, rel=1e-6)

def test_cos():
    # Test cosine operation on a Dual_x number
    test_number = Dual_x(5.0, 1.0)
    cos_test = test_number.cos()
    expected_real = np.cos(5.0)
    expected_dual = -np.sin(5.0) * 1.0
    assert cos_test.real == pytest.approx(expected_real, rel=1e-6)
    assert cos_test.dual == pytest.approx(expected_dual, rel=1e-6)

def test_tan():
    # Test tangent operation on a Dual_x number
    test_number = Dual_x(5.0, 1.0)
    tan_test = test_number.tan()
    expected_real = np.tan(5.0)
    expected_dual = (1 / np.cos(5.0)) ** 2 * 1.0
    assert tan_test.real == pytest.approx(expected_real, rel=1e-6)
    assert tan_test.dual == pytest.approx(expected_dual, rel=1e-6)

    # Test ValueError for undefined tangent at pi/2 + n*pi
    invalid_number_test = Dual_x(np.pi / 2, 1.0)
    with pytest.raises(ValueError, match=re.escape("Real value too close to pi/2 + n*pi.")):
        invalid_number_test.tan()

    # Test RuntimeWarning for near-undefined tangent
    almost_invalid = Dual_x(np.pi / 2 + 1e-8, 1.0)
    with pytest.warns(RuntimeWarning, match=re.escape("Real value close to pi/2 + n*pi; numerical instability possible.")):
        tan_almost = almost_invalid.tan()
        expected_real = np.tan(almost_invalid.real)
        expected_dual = (1 / np.cos(almost_invalid.real)) ** 2 * 1.0
        assert tan_almost.real == pytest.approx(expected_real, rel=1e-6)
        assert tan_almost.dual == pytest.approx(expected_dual, rel=1e-6)

def test_log():
    # Test natural logarithm on a Dual_x number
    test_number = Dual_x(5.0, 1.0)
    log_test = test_number.log()
    expected_real = np.log(5.0)
    expected_dual = 1 / 5.0 * 1.0
    assert log_test.real == pytest.approx(expected_real, rel=1e-6)
    assert log_test.dual == pytest.approx(expected_dual, rel=1e-6)

    # Test ValueError for zero input
    invalid_number1 = Dual_x(0.0, 1.0)
    with pytest.raises(ValueError) as excinfo:
        invalid_number1.log()
    assert "Log cannot take 0 or negative real part." in str(excinfo)

    # Test ValueError for negative input
    invalid_number2 = Dual_x(-5.0, 1.0)
    with pytest.raises(ValueError) as excinfo2:
        invalid_number2.log()
    assert "Log cannot take 0 or negative real part." in str(excinfo2)

    # Test ValueError for inputs near zero (below tolerance_exception)
    small_number = Dual_x(1e-11, 1.0)
    with pytest.raises(ValueError, match=re.escape("Real value less than 1e-10. Potential overflow in log.")):
        small_number.log()

    # Test RuntimeWarning for inputs close to zero (above tolerance_exception)
    almost_zero = Dual_x(1e-7, 1.0)
    with pytest.warns(RuntimeWarning, match=re.escape("Log input close to zero; numerical instability possible.")):
        log_almost_zero = almost_zero.log()
        expected_real = np.log(1e-7)
        expected_dual = 1 / 1e-7 * 1.0
        assert log_almost_zero.real == pytest.approx(expected_real, rel=1e-6)
        assert log_almost_zero.dual == pytest.approx(expected_dual, rel=1e-6)

def test_exp():
    # Test exponential operation on a Dual_x number
    test_number = Dual_x(5.0, 1.0)
    exp_test = test_number.exp()
    expected_real = np.exp(5.0)
    expected_dual = np.exp(5.0) * 1.0
    assert exp_test.real == expected_real
    assert exp_test.dual == expected_dual



# Tests for Dual_x class with array inputs
def test_init_array():
    # Test initialization of Dual_x with arrays
    test_number = Dual_x(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert np.all(test_number.real == np.array([1.0, 2.0]))
    assert np.all(test_number.dual == np.array([3.0, 4.0]))

def test_add_array():
    # Test addition of two Dual_x numbers with arrays
    test_number1 = Dual_x(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    test_number2 = Dual_x(np.array([5.0, 6.0]), np.array([7.0, 8.0]))
    test_sum = test_number1 + test_number2
    assert np.all(test_sum.real == np.array([6.0, 8.0]))
    assert np.all(test_sum.dual == np.array([10.0, 12.0]))

def test_sub_array():
    # Test subtraction of two Dual_x numbers with arrays
    test_number1 = Dual_x(np.array([5.0, 6.0]), np.array([7.0, 8.0]))
    test_number2 = Dual_x(np.array([3.0, 2.0]), np.array([1.0, 2.0]))
    test_diff = test_number1 - test_number2
    assert np.all(test_diff.real == np.array([2.0, 4.0]))
    assert np.all(test_diff.dual == np.array([6.0, 6.0]))

def test_mul_array():
    # Test multiplication of two Dual_x numbers with arrays
    test_number1 = Dual_x(np.array([5.0, 2.0]), np.array([3.0, 1.0]))
    test_number2 = Dual_x(np.array([4.0, 3.0]), np.array([2.0, 2.0]))
    test_prod = test_number1 * test_number2
    expected_real = np.array([5.0 * 4.0, 2.0 * 3.0])
    expected_dual = np.array([5.0 * 2.0 + 3.0 * 4.0, 2.0 * 2.0 + 1.0 * 3.0])
    assert np.all(test_prod.real == expected_real)
    assert np.all(test_prod.dual == expected_dual)

def test_pow_array():
    # Test power operation on a Dual_x number with arrays
    test_number = Dual_x(np.array([2.0, 3.0]), np.array([1.0, 1.0]))
    power = test_number ** 2
    expected_real = np.array([4.0, 9.0])
    expected_dual = np.array([2 * 2.0 * 1.0, 2 * 3.0 * 1.0])
    assert np.all(power.real == expected_real)
    assert np.all(power.dual == expected_dual)

def test_sin_array():
    # Test sine operation on a Dual_x number with arrays
    test_number = Dual_x(np.array([0.0, np.pi / 4]), np.array([1.0, 1.0]))
    sin_test = test_number.sin()
    expected_real = np.sin(np.array([0.0, np.pi / 4]))
    expected_dual = np.cos(np.array([0.0, np.pi / 4]))
    assert sin_test.real == pytest.approx(expected_real, rel=1e-6)
    assert sin_test.dual == pytest.approx(expected_dual, rel=1e-6)

def test_cos_array():
    # Test cosine operation on a Dual_x number with arrays
    test_number = Dual_x(np.array([0.0, np.pi / 4]), np.array([1.0, 1.0]))
    cos_test = test_number.cos()
    expected_real = np.cos(np.array([0.0, np.pi / 4]))
    expected_dual = -np.sin(np.array([0.0, np.pi / 4]))
    assert cos_test.real == pytest.approx(expected_real, rel=1e-6)
    assert cos_test.dual == pytest.approx(expected_dual, rel=1e-6)

def test_tan_array():
    # Test tangent operation on a Dual_x number with arrays
    test_number = Dual_x(np.array([0.0, np.pi / 4]), np.array([1.0, 1.0]))
    tan_test = test_number.tan()
    expected_real = np.tan(np.array([0.0, np.pi / 4]))
    expected_dual = (1 / np.cos(np.array([0.0, np.pi / 4]))) ** 2
    assert tan_test.real == pytest.approx(expected_real, rel=1e-6)
    assert tan_test.dual == pytest.approx(expected_dual, rel=1e-6)

    # Test exception for undefined tangent at pi/2 + n*pi
    invalid_number_test = Dual_x(np.array([np.pi / 2, 3 * np.pi / 2]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match=re.escape("Real value too close to pi/2 + n*pi.")):
        invalid_number_test.tan()

    # Test warning for near-undefined tangent
    almost_invalid = Dual_x(np.array([np.pi / 2 + 1e-8, 3 * np.pi / 2 - 1e-8]), np.array([1.0, 1.0]))
    with pytest.warns(RuntimeWarning, match=re.escape("Real value close to pi/2 + n*pi; numerical instability possible.")):
        tan_almost = almost_invalid.tan()
        expected_real = np.tan(almost_invalid.real)
        expected_dual = (1 / np.cos(almost_invalid.real)) ** 2 * almost_invalid.dual
        assert tan_almost.real == pytest.approx(expected_real, rel=1e-6)
        assert tan_almost.dual == pytest.approx(expected_dual, rel=1e-6)

def test_log_array():
    # Test natural logarithm on a Dual_x number with arrays
    test_number = Dual_x(np.array([2.0, 3.0]), np.array([1.0, 1.0]))
    log_test = test_number.log()
    expected_real = np.log(np.array([2.0, 3.0]))
    expected_dual = 1 / np.array([2.0, 3.0])
    assert log_test.real == pytest.approx(expected_real, rel=1e-6)
    assert log_test.dual == pytest.approx(expected_dual, rel=1e-6)

    # Test exception for zero and negative real values
    invalid_numbers = Dual_x(np.array([1.0, -5.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match=re.escape("Log cannot take 0 or negative real part.")):
        invalid_numbers.log()

    # Test exception for real values near zero
    small_values = Dual_x(np.array([1e-11, 1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match=re.escape("Real value less than 1e-10. Potential overflow in log.")):
        small_values.log()

    # Test warning for values near zero but greater than the threshold
    almost_zero = Dual_x(np.array([1e-7, 3e-7]), np.array([1.0, 1.0]))
    with pytest.warns(RuntimeWarning, match=re.escape("Log input close to zero; numerical instability possible.")):
        log_almost_zero = almost_zero.log()
        expected_real = np.log(np.array([1e-7, 3e-7]))
        expected_dual = 1 / np.array([1e-7, 3e-7]) * almost_zero.dual
        assert log_almost_zero.real == pytest.approx(expected_real, rel=1e-6)
        assert log_almost_zero.dual == pytest.approx(expected_dual, rel=1e-6)

def test_exp_array():
    # Test exponential operation on a Dual_x number with arrays
    test_number = Dual_x(np.array([2.0, 3.0]), np.array([1.0, 1.0]))
    exp_test = test_number.exp()
    expected_real = np.exp(np.array([2.0, 3.0]))
    expected_dual = np.exp(np.array([2.0, 3.0]))
    assert exp_test.real == pytest.approx(expected_real, rel=1e-6)
    assert exp_test.dual == pytest.approx(expected_dual, rel=1e-6)

def test_shape_mismatch_exception():
    # Test exception for shape mismatch between real and dual parts
    real = np.array([1.0, 2.0, 3.0])
    dual = np.array([4.0, 5.0])  # Mismatched shape
    with pytest.raises(ValueError, match="Shape mismatch"):
        Dual_x(real, dual)
# Tests for Dual_x_array class with array inputs

def test_init_array_adapt():
    # Test initialization of Dual_x_array with arrays
    test_number = Dual_x_array(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    # Verify that real and dual parts are correctly stored
    assert np.all(test_number.real == np.array([1.0, 2.0]))
    assert np.all(test_number.dual == np.array([3.0, 4.0]))

def test_add_array_adapt():
    # Test element-wise addition of Dual_x_array objects
    test_number1 = Dual_x_array(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    test_number2 = Dual_x_array(np.array([5.0, 6.0]), np.array([7.0, 8.0]))
    test_sum = test_number1 + test_number2
    # Verify that real and dual parts are added correctly
    assert np.all(test_sum.real == np.array([6.0, 8.0]))
    assert np.all(test_sum.dual == np.array([10.0, 12.0]))

def test_sub_array_adapt():
    # Test element-wise subtraction of Dual_x_array objects
    test_number1 = Dual_x_array(np.array([5.0, 6.0]), np.array([7.0, 8.0]))
    test_number2 = Dual_x_array(np.array([3.0, 2.0]), np.array([1.0, 2.0]))
    test_diff = test_number1 - test_number2
    # Verify that real and dual parts are subtracted correctly
    assert np.all(test_diff.real == np.array([2.0, 4.0]))
    assert np.all(test_diff.dual == np.array([6.0, 6.0]))

def test_mul_array_adapt():
    # Test element-wise multiplication of Dual_x_array objects
    test_number1 = Dual_x_array(np.array([5.0, 2.0]), np.array([3.0, 1.0]))
    test_number2 = Dual_x_array(np.array([4.0, 3.0]), np.array([2.0, 2.0]))
    test_prod = test_number1 * test_number2
    # Verify that real and dual parts follow the dual number multiplication rule
    expected_real = np.array([5.0 * 4.0, 2.0 * 3.0])
    expected_dual = np.array([5.0 * 2.0 + 3.0 * 4.0, 2.0 * 2.0 + 1.0 * 3.0])
    assert np.all(test_prod.real == expected_real)
    assert np.all(test_prod.dual == expected_dual)

def test_pow_array_adapt():
    # Test element-wise power operation for Dual_x_array
    test_number = Dual_x_array(np.array([2.0, 3.0]), np.array([1.0, 1.0]))
    power = test_number ** 2
    # Verify that real and dual parts are calculated correctly
    expected_real = np.array([4.0, 9.0])
    expected_dual = np.array([2 * 2.0 * 1.0, 2 * 3.0 * 1.0])
    assert np.all(power.real == expected_real)
    assert np.all(power.dual == expected_dual)

def test_sin_array_adapt():
    # Test element-wise sine function for Dual_x_array
    test_number = Dual_x_array(np.array([0.0, np.pi / 4]), np.array([1.0, 1.0]))
    sin_test = test_number.sin()
    expected_real = np.sin(np.array([0.0, np.pi / 4]))
    expected_dual = np.cos(np.array([0.0, np.pi / 4]))
    assert sin_test.real == pytest.approx(expected_real, rel=1e-6)
    assert sin_test.dual == pytest.approx(expected_dual, rel=1e-6)

def test_cos_array_adapt():
    # Test element-wise cosine function for Dual_x_array
    test_number = Dual_x_array(np.array([0.0, np.pi / 4]), np.array([1.0, 1.0]))
    cos_test = test_number.cos()
    expected_real = np.cos(np.array([0.0, np.pi / 4]))
    expected_dual = -np.sin(np.array([0.0, np.pi / 4]))
    assert cos_test.real == pytest.approx(expected_real, rel=1e-6)
    assert cos_test.dual == pytest.approx(expected_dual, rel=1e-6)

def test_tan_array_adapt():
    # Test element-wise tangent function for Dual_x_array
    test_number = Dual_x_array(np.array([0.0, np.pi / 4]), np.array([1.0, 1.0]))
    tan_test = test_number.tan()
    expected_real = np.tan(np.array([0.0, np.pi / 4]))
    expected_dual = (1 / np.cos(np.array([0.0, np.pi / 4]))) ** 2
    assert tan_test.real == pytest.approx(expected_real, rel=1e-6)
    assert tan_test.dual == pytest.approx(expected_dual, rel=1e-6)

    # Test exception for values where tangent is undefined
    invalid_number_test = Dual_x_array(np.array([np.pi / 2, 3 * np.pi / 2]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match=re.escape("Real value too close to pi/2 + n*pi.")):
        invalid_number_test.tan()

    # Test warning for values near undefined tangent points
    almost_invalid = Dual_x_array(np.array([np.pi / 2 + 1e-8, 3 * np.pi / 2 - 1e-8]), np.array([1.0, 1.0]))
    with pytest.warns(RuntimeWarning, match=re.escape("Real value close to pi/2 + n*pi; numerical instability possible.")):
        tan_almost = almost_invalid.tan()
        expected_real = np.tan(almost_invalid.real)
        expected_dual = (1 / np.cos(almost_invalid.real)) ** 2 * almost_invalid.dual
        assert tan_almost.real == pytest.approx(expected_real, rel=1e-6)
        assert tan_almost.dual == pytest.approx(expected_dual, rel=1e-6)

def test_log_array_adapt():
    # Test element-wise logarithm function for Dual_x_array
    test_number = Dual_x_array(np.array([2.0, 3.0]), np.array([1.0, 1.0]))
    log_test = test_number.log()
    expected_real = np.log(np.array([2.0, 3.0]))
    expected_dual = 1 / np.array([2.0, 3.0])
    assert log_test.real == pytest.approx(expected_real, rel=1e-6)
    assert log_test.dual == pytest.approx(expected_dual, rel=1e-6)

    # Test exceptions for invalid values
    invalid_numbers = Dual_x_array(np.array([1.0, -5.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match=re.escape("Log cannot take 0 or negative real part.")):
        invalid_numbers.log()

    # Test exceptions for values too close to zero
    small_values = Dual_x_array(np.array([1e-11, 1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match=re.escape("Real value less than 1e-10. Potential overflow in log.")):
        small_values.log()

    # Test warnings for near-zero values
    almost_zero = Dual_x_array(np.array([1e-7, 3e-7]), np.array([1.0, 1.0]))
    with pytest.warns(RuntimeWarning, match=re.escape("Log input close to zero; numerical instability possible.")):
        log_almost_zero = almost_zero.log()
        expected_real = np.log(np.array([1e-7, 3e-7]))
        expected_dual = 1 / np.array([1e-7, 3e-7]) * almost_zero.dual
        assert log_almost_zero.real == pytest.approx(expected_real, rel=1e-6)
        assert log_almost_zero.dual == pytest.approx(expected_dual, rel=1e-6)

def test_exp_array_adapt():
    # Test element-wise exponential function for Dual_x_array
    test_number = Dual_x_array(np.array([2.0, 3.0]), np.array([1.0, 1.0]))
    exp_test = test_number.exp()
    # Verify that real and dual parts are calculated correctly
    expected_real = np.exp(np.array([2.0, 3.0]))
    expected_dual = np.exp(np.array([2.0, 3.0]))
    assert exp_test.real == pytest.approx(expected_real, rel=1e-6)
    assert exp_test.dual == pytest.approx(expected_dual, rel=1e-6)

def test_shape_mismatch_exception_adapt():
    # Test exception for mismatched shapes between real and dual arrays
    real = np.array([1.0, 2.0, 3.0])
    dual = np.array([4.0, 5.0])  # Mismatched shape
    with pytest.raises(ValueError, match="Shape mismatch"):
        Dual_x_array(real, dual)
