import numpy as np
import warnings

class Dual:
    r"""A class representing dual numbers for automatic differentiation.

    Attributes:
        real (float, int): The real part of the dual number.
        dual (float, int): The dual part of the dual number.

    Note:
        For mathematical operations like sine, cosine, and logarithm, the real and dual parts of the output
        are evaluated according to the following formula:
        
        .. math::

            f(a + b\epsilon) = f(a) + f'(a)b\epsilon

        This formula describes how dual numbers are processed through a given mathematical function \(f\).
    """

    def __init__(self, real, dual):
        """Initialize an object of the Dual class.

        Args:
            real (float, int): The real part of the dual number.
            dual (float, int): The dual part of the dual number.

        Note:
            Retrieve the real part of the Dual class object with `object.real` and the dual part with `object.dual`.
        """
        self.real = real
        self.dual = dual

    def __add__(self, other):
        """Add two Dual numbers.

        Operator:
            Uses the :math:`+` operator.

        Returns:
            Dual: A new Dual number representing their sum.
        """
        return Dual(self.real + other.real, self.dual + other.dual)

    def __sub__(self, other):
        """Subtract one Dual number from another.

        Operator:
            Uses the :math:`-` operator.

        Returns:
            Dual: A new Dual number representing the difference.
        
        Note:
            For addition and subtraction, the real and dual parts are added or subtracted separately.
        """
        return Dual(self.real - other.real, self.dual - other.dual)

    def __mul__(self, other):
        r"""Multiply two Dual numbers.

        Operator:
            Uses the :math:`*` operator.

        Returns:
            Dual: A new Dual number representing the product :math:`(a + b\epsilon)(c + d\epsilon)`. 
            The real part of the product output is simply the product of the real parts of the arguments :math:`ab`. 
            The dual part of the output is the term that is first order in :math:`\epsilon` :math:`(ad + bc)`.
        """
        return Dual(
            self.real * other.real,
            self.real * other.dual + self.dual * other.real
        )

    def __pow__(self, exponent):
        """Raise a Dual number to a power.

        Operator:
            Uses the :math:`**` operator.

        Args:
            exponent (float, int): The exponent to raise the Dual number to. Must be a real number.

        Returns:
            Dual: A new Dual number raised to the power of the exponent.

        """
        return Dual(
            self.real ** exponent,
            exponent * self.real ** (exponent - 1) * self.dual
        )

    def sin(self):
        """Compute the sine of the Dual number.

        Returns:
            Dual: A new Dual number representing the sine.
        """
        return Dual(
            np.sin(self.real),
            np.cos(self.real) * self.dual
        )

    def cos(self):
        """Compute the cosine of the Dual number.

        Returns:
            Dual: A new Dual number representing the cosine.
        """
        return Dual(
            np.cos(self.real),
            -np.sin(self.real) * self.dual
        )

    def tan(self):
        """Compute the tangent of the Dual number.

        Returns:
            Dual: A new Dual number representing the tangent.

        Raises:
            ValueError: If the real part is within 1e-10 of (π/2 + nπ), where tangent is undefined.
            RuntimeWarning: If the real part is close to (π/2 + nπ) by less than 1e-6, which may cause numerical instability.
        """
        tolerance_exception = 1e-10
        tolerance_warning = 1e-6
        n = round((self.real - np.pi / 2) / np.pi)
        pi_over_2_plus_n_pi = np.pi / 2 + n * np.pi
        delta = abs(self.real - pi_over_2_plus_n_pi)
        if delta < tolerance_exception:
            raise ValueError(
                "Real value cannot be within 1e-10 of pi/2 + n*pi. Tan and 1/cos(real) are both undefined at these points."
            )
        elif delta < tolerance_warning:
            warnings.warn(
                "The proximity of the real value is less than 1e-6 to pi/2 + n*pi. Beware of potential numerical instability.",
                RuntimeWarning
            )
        return Dual(
            np.tan(self.real),
            (1 / np.cos(self.real)) ** 2 * self.dual
        )

    def log(self):
        """Compute the natural logarithm of the Dual number.

        Returns:
            Dual: A new Dual number representing the natural logarithm.

        Raises:
            ValueError: If the real part is less than or equal to zero.
            ValueError: If the real part is less than 1e-10.
            RuntimeWarning: If the real part is close to zero within 1e-6 but larger than 1e-10, to warn of potential numerical instability.
        """
        tolerance_exception = 1e-10
        tolerance_warning = 1e-6
        if self.real > 0 and self.real <= tolerance_exception:
            raise ValueError(
                "Real value is less than 1e-10. Log is undefined at zero, beware of potential overflow."
            )
        elif self.real > tolerance_exception and self.real < tolerance_warning:
            warnings.warn(
                "Log is undefined for x <= 0. The proximity of the real value to 0 is within 1e-6. Beware of potential numerical instability.",
                RuntimeWarning
            )
        elif self.real <= 0:
            raise ValueError(
                "Log cannot take in 0 or less than 0 for the real value. Real value must be greater than zero."
            )
        return Dual(
            np.log(self.real),
            (1 / self.real) * self.dual
        )

    def exp(self):
        """Compute the exponential of the Dual number.

        Returns:
            Dual: A new Dual number representing the exponential.
        """
        return Dual(
            np.exp(self.real),
            np.exp(self.real) * self.dual
        )
