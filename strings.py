from libc.math cimport sin, cos, tan, log, exp, pow
import numpy as np
cimport numpy as cnp
import warnings

cdef class Dual_x:
    r"""A class representing dual numbers for automatic differentiation.

    Attributes:
        real (float, int, or array-like): The real part of the dual number. 
            This can be a scalar (float or int) or an array-like object (e.g., list, tuple, numpy.ndarray).
        dual (float, int, or array-like): The dual part of the dual number.
            This can be a scalar (float or int) or an array-like object (e.g., list, tuple, numpy.ndarray).

    Note:
        For mathematical operations like sine, cosine, and logarithm, the real and dual parts of the output
        are evaluated according to the following formula:
        
        .. math::

            f(a + b\epsilon) = f(a) + f'(a)b\epsilon

        This formula describes how dual numbers are processed through a given mathematical function \(f\).
    """
    cdef public object real  # Allow scalars or arrays
    cdef public object dual

    def __init__(self, real, dual):
        """
        Initialize the Dual_x class to support dynamcially distinguish between scalars and arrays.
        If real and dual are arrays, enforce that they must be the same shape.
        """
        if isinstance(real, (list, tuple)):
            self.real = np.asarray(real, dtype=np.float64)
        elif isinstance(real, (float, int, np.ndarray)):
            self.real = real
        else:
            raise TypeError("Invalid type for 'real'. Must be scalar, list, tuple, or array.")

        if isinstance(dual, (list, tuple)):
            self.dual = np.asarray(dual, dtype=np.float64)
        elif isinstance(dual, (float, int, np.ndarray)):
            self.dual = dual
        else:
            raise TypeError("Invalid type for 'dual'. Must be scalar, list, tuple, or array.")

        # Check if both are arrays and their shapes match
        if isinstance(self.real, np.ndarray) and isinstance(self.dual, np.ndarray):
            if self.real.shape != self.dual.shape:
                raise ValueError(f"Shape mismatch: real {self.real.shape}, dual {self.dual.shape}")

    def __add__(self, other):
        """Initialize an object of the Dual class.

        Args:
            real (float, int, or array-like): The real part of the dual number.
                This can be a scalar or an array-like object.
            dual (float, int, or array-like): The dual part of the dual number.
                This can be a scalar or an array-like object.

        Raises:
            ValueError: If both `real` and `dual` are arrays (e.g., numpy.ndarray) but their shapes do not match.

        Note:
            If both `real` and `dual` are arrays, a check is performed to ensure their shapes match.
            This is to ensure that element-wise operations on the dual number are valid. If the shapes
            are mismatched, a `ValueError` is raised.
        """
        return Dual_x(self.real + other.real, self.dual + other.dual)

    def __sub__(self, other):
        """Subtract one Dual number from another.

        Operator:
            Uses the :math:`-` operator.

        Returns:
            Dual: A new Dual number representing the difference.
        
        Note:
            For addition and subtraction, the real and dual parts are added or subtracted separately.
        """
        return Dual_x(self.real - other.real, self.dual - other.dual)

    def __mul__(self, other):
        r"""Multiply two Dual numbers.

        Operator:
            Uses the :math:`*` operator.

        Returns:
            Dual: A new Dual number representing the product :math:`(a + b\epsilon)(c + d\epsilon)`. 
            The real part of the product output is simply the product of the real parts of the arguments :math:`ab`. 
            The dual part of the output is the term that is first order in :math:`\epsilon` :math:`(ad + bc)`.
        """
        return Dual_x(
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
        if isinstance(self.real, np.ndarray):
            return Dual_x(
                np.power(self.real, exponent),
                exponent * np.power(self.real, exponent - 1) * self.dual
            )
        else:
            return Dual_x(
                pow(self.real, exponent),
                exponent * pow(self.real, exponent - 1) * self.dual
            )

    cpdef Dual_x sin(self):
        """Compute the sine of the Dual number.

        Returns:
            Dual: A new Dual number representing the sine.
        """
        if isinstance(self.real, np.ndarray):
            return Dual_x(
                np.sin(self.real),
                np.cos(self.real) * self.dual
            )
        else:
            return Dual_x(
                sin(self.real),
                cos(self.real) * self.dual
            )

    cpdef Dual_x cos(self):
        """Compute the cosine of the Dual number.

        Returns:
            Dual: A new Dual number representing the cosine.
        """
        if isinstance(self.real, np.ndarray):
            return Dual_x(
                np.cos(self.real),
                -np.sin(self.real) * self.dual
            )
        else:
            return Dual_x(
                cos(self.real),
                -sin(self.real) * self.dual
            )

    cpdef Dual_x tan(self):
        """Compute the tangent of the Dual number.

        Returns:
            Dual: A new Dual number representing the tangent.

        Raises:
            ValueError: If the real part is within 1e-10 of (π/2 + nπ), where tangent is undefined.
            RuntimeWarning: If the real part is close to (π/2 + nπ) by less than 1e-6, which may cause numerical instability.
        """
        tolerance_exception = 1e-10
        tolerance_warning = 1e-6

        if isinstance(self.real, np.ndarray):
            n = np.round((self.real - np.pi / 2) / np.pi)
            pi_over_2_plus_n_pi = np.pi / 2 + n * np.pi
            delta = np.abs(self.real - pi_over_2_plus_n_pi)

            if np.any(delta < tolerance_exception):
                raise ValueError("Real value too close to pi/2 + n*pi.")
            elif np.any((delta >= tolerance_exception) & (delta < tolerance_warning)):
                warnings.warn("Real value close to pi/2 + n*pi; numerical instability possible.", RuntimeWarning)

            return Dual_x(
                np.tan(self.real),
                (1.0 / (np.cos(self.real) ** 2)) * self.dual
            )
        else:
            n = round((self.real - 3.141592653589793 / 2) / 3.141592653589793)
            pi_over_2_plus_n_pi = 3.141592653589793 / 2 + n * 3.141592653589793
            delta = abs(self.real - pi_over_2_plus_n_pi)

            if delta < tolerance_exception:
                raise ValueError("Real value too close to pi/2 + n*pi.")
            elif delta < tolerance_warning:
                warnings.warn("Real value close to pi/2 + n*pi; numerical instability possible.", RuntimeWarning)

            val = tan(self.real)
            deriv = (1.0 / (cos(self.real) * cos(self.real))) * self.dual
            return Dual_x(val, deriv)

    cpdef Dual_x log(self):
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

        if isinstance(self.real, np.ndarray):
            if np.any(self.real <= 0):
                raise ValueError("Log cannot take 0 or negative real part.")
            elif np.any(self.real <= tolerance_exception):
                raise ValueError("Real value less than 1e-10. Potential overflow in log.")
            elif np.any(self.real < tolerance_warning):
                warnings.warn("Log input close to zero; numerical instability possible.", RuntimeWarning)

            return Dual_x(
                np.log(self.real),
                (1.0 / self.real) * self.dual
            )
        else:
            if self.real <= 0:
                raise ValueError("Log cannot take 0 or negative real part.")
            elif self.real <= tolerance_exception:
                raise ValueError("Real value less than 1e-10. Potential overflow in log.")
            elif self.real < tolerance_warning:
                warnings.warn("Log input close to zero; numerical instability possible.", RuntimeWarning)

            return Dual_x(
                log(self.real),
                (1.0 / self.real) * self.dual
            )

    cpdef Dual_x exp(self):
        """Compute the exponential of the Dual number.

        Returns:
            Dual: A new Dual number representing the exponential.
        """
        if isinstance(self.real, np.ndarray):
            val = np.exp(self.real)
            return Dual_x(val, val * self.dual)
        else:
            val = exp(self.real)
            return Dual_x(val, val * self.dual)




cdef class Dual_x_array:
    cdef public object real  # Store real as Python object
    cdef public object dual  # Store dual as Python object

    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=1] real, cnp.ndarray[cnp.float64_t, ndim=1] dual):
        """
        Initialize the Dual_x_array class, assuming both inputs are 1D float arrays. Avoids dynamic type checking
        for boosted performance.

        Args:
            real (float, int, or array-like): The real part of the dual number.
                This can be a scalar or an array-like object.
            dual (float, int, or array-like): The dual part of the dual number.
                This can be a scalar or an array-like object.

        Raises:
            ValueError: Inputs `real` and `dual` are arrays (e.g., numpy.ndarray) but their shapes do not match.

        Note:
            The rest of the syntax functions the same as in Dual_x.

        """
        self.real = real
        self.dual = dual

        # Ensure their shapes match
        if real.shape[0] != dual.shape[0]:  # Check the first (and only) dimension
            raise ValueError(
                f"Shape mismatch: real.shape={real.shape[0]}, dual.shape={dual.shape[0]}"
            )

    def __add__(self, other):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r1 = self.real
        cdef cnp.ndarray[cnp.float64_t, ndim=1] d1 = self.dual
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r2 = other.real
        cdef cnp.ndarray[cnp.float64_t, ndim=1] d2 = other.dual

        return Dual_x_array(r1 + r2, d1 + d2)

    def __sub__(self, other):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r1 = self.real
        cdef cnp.ndarray[cnp.float64_t, ndim=1] d1 = self.dual
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r2 = other.real
        cdef cnp.ndarray[cnp.float64_t, ndim=1] d2 = other.dual

        return Dual_x_array(r1 - r2, d1 - d2)

    def __mul__(self, other):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r1 = self.real
        cdef cnp.ndarray[cnp.float64_t, ndim=1] d1 = self.dual
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r2 = other.real
        cdef cnp.ndarray[cnp.float64_t, ndim=1] d2 = other.dual

        return Dual_x_array(r1 * r2, r1 * d2 + d1 * r2)

    def __pow__(self, double exponent):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r = self.real
        cdef cnp.ndarray[cnp.float64_t, ndim=1] d = self.dual

        return Dual_x_array(
            np.power(r, exponent),
            exponent * np.power(r, exponent - 1) * d
        )

    cpdef Dual_x_array sin(self):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r = self.real
        cdef cnp.ndarray[cnp.float64_t, ndim=1] d = self.dual

        return Dual_x_array(np.sin(r), np.cos(r) * d)

    cpdef Dual_x_array cos(self):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r = self.real
        cdef cnp.ndarray[cnp.float64_t, ndim=1] d = self.dual

        return Dual_x_array(np.cos(r), -np.sin(r) * d)

    cpdef Dual_x_array tan(self):
        cdef double tolerance_exception = 1e-10
        cdef double tolerance_warning = 1e-6

        cdef cnp.ndarray[cnp.float64_t, ndim=1] n = np.round((self.real - np.pi / 2) / np.pi)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] pi_over_2_plus_n_pi = np.pi / 2 + n * np.pi
        cdef cnp.ndarray[cnp.float64_t, ndim=1] delta = np.abs(self.real - pi_over_2_plus_n_pi)

        if np.any(delta < tolerance_exception):
            raise ValueError("Real value too close to pi/2 + n*pi.")
        elif np.any((delta >= tolerance_exception) & (delta < tolerance_warning)):
            warnings.warn("Real value close to pi/2 + n*pi; numerical instability possible.", RuntimeWarning)
        return Dual_x_array(
            np.tan(self.real),
            (1.0 / (np.cos(self.real) ** 2)) * self.dual
        )
    cpdef Dual_x_array log(self):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r = self.real
        cdef cnp.ndarray[cnp.float64_t, ndim=1] d = self.dual

        if np.any(r <= 0):
            raise ValueError("Log cannot take 0 or negative real part.")

        return Dual_x_array(np.log(r), (1.0 / r) * d)

    cpdef Dual_x_array exp(self):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r = self.real
        cdef cnp.ndarray[cnp.float64_t, ndim=1] d = self.dual

        val = np.exp(r)
        return Dual_x_array(val, val * d)



