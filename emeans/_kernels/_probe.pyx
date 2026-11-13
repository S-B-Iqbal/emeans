# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Build probe: was this extension compiled with OpenMP? (rule 07)."""


cdef extern from *:
    """
    #ifdef _OPENMP
    #include <omp.h>
    #define EMEANS_HAS_OPENMP 1
    static int emeans_max_threads(void) { return omp_get_max_threads(); }
    #else
    #define EMEANS_HAS_OPENMP 0
    static int emeans_max_threads(void) { return 1; }
    #endif
    """
    int EMEANS_HAS_OPENMP
    int emeans_max_threads()


def has_openmp():
    """Return True when the extension was compiled with OpenMP support."""
    return bool(EMEANS_HAS_OPENMP)


def max_threads():
    """Return the OpenMP thread limit (1 when built without OpenMP)."""
    return int(emeans_max_threads())
