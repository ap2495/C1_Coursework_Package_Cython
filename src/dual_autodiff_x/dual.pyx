
from libc.math cimport sin, cos, tan, log, exp, pow
import warnings

cdef class Dual:
    cdef double real
    cdef double dual

    def __cinit__(self, double real, double dual):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        return Dual(self.real + other.real, self.dual + other.dual)

    def __sub__(self, other):
        return Dual(self.real - other.real, self.dual - other.dual)

    def __mul__(self, other):
        return Dual(
            self.real * other.real,
            self.real * other.dual + self.dual * other.real
        )

    def __pow__(self, exponent):
        return Dual(
            pow(self.real, exponent),
            exponent * pow(self.real, exponent - 1) * self.dual
        )

    cpdef Dual sin(self):
        return Dual(
            sin(self.real),
            cos(self.real) * self.dual
        )

    cpdef Dual cos(self):
        return Dual(
            cos(self.real),
            -sin(self.real) * self.dual
        )

    cpdef Dual tan(self):
        cdef double tolerance_exception = 1e-10
        cdef double tolerance_warning = 1e-6
        cdef double n = round((self.real - 3.141592653589793 / 2) / 3.141592653589793)
        cdef double pi_over_2_plus_n_pi = 3.141592653589793 / 2 + n * 3.141592653589793
        cdef double delta = abs(self.real - pi_over_2_plus_n_pi)
        if delta < tolerance_exception:
            raise ValueError("Real value too close to pi/2 + n*pi.")
        elif delta < tolerance_warning:
            warnings.warn("Real value close to pi/2 + n*pi; numerical instability possible.", RuntimeWarning)

        cdef double val = tan(self.real)
        cdef double deriv = (1.0 / (cos(self.real) * cos(self.real))) * self.dual
        return Dual(val, deriv)

    cpdef Dual log(self):
        cdef double tolerance_exception = 1e-10
        cdef double tolerance_warning = 1e-6

        if self.real <= 0:
            raise ValueError("Log cannot take 0 or negative real part.")
        elif self.real <= tolerance_exception:
            raise ValueError("Real value less than 1e-10. Potential overflow in log.")
        elif self.real < tolerance_warning:
            warnings.warn("Log input close to zero; numerical instability possible.", RuntimeWarning)

        return Dual(log(self.real), (1.0 / self.real) * self.dual)

    cpdef Dual exp(self):
        cdef double val = exp(self.real)
        return Dual(val, val * self.dual)
