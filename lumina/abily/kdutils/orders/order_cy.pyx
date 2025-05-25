# order_cy.pyx
import numpy as np
cimport numpy as np
from libc.math cimport isnan

# Initialize NumPy C-API (important for Cython modules using NumPy)
np.import_array()

# Type definitions for clarity and C-level performance
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t
ctypedef np.int32_t int32_t
ctypedef unsigned char bint # For boolean, often maps to C char

# Global constant for NaN, can be accessed efficiently
cdef double NAN_CY = np.nan

# Cython equivalent of determine_trade_result_numba
# nogil allows this function to run without the Python Global Interpreter Lock,
# making it faster if called from a nogil context (though our main loop will hold GIL for list appends).
cdef int _determine_trade_result_cy(float64_t entry_price, float64_t exit_price, int32_t direction) nogil:
    if isnan(entry_price) or isnan(exit_price):
        return 0  # unknown_price
    if direction == 1:  # Long
        if exit_price > entry_price: return 1  # win
        elif exit_price < entry_price: return 2  # loss
        else: return 3  # even
    elif direction == -1:  # Short
        if exit_price < entry_price: return 1  # win
        elif exit_price > entry_price: return 2  # loss
        else: return 3  # even
    return 4  # unknown_direction

# Cython equivalent of _position_next_order_numba_core
# Decorators for minor optimizations (use with care, ensure bounds are respected)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True) # Use C division for integers
def _core_logic_cy(
    float64_t[:] pos_values,         # Typed memoryview for efficient NumPy array access
    int64_t[:] trade_action_times_ns,
    float64_t[:] trade_action_prices
):
    # Python list to store output tuples. Operations on Python objects require GIL.
    cdef list orders_data = []

    cdef bint has_active_trade = False
    cdef int64_t active_entry_time_ns = 0
    cdef float64_t active_entry_price = 0.0
    cdef int32_t active_direction = 0

    cdef Py_ssize_t n = pos_values.shape[0]
    if n == 0:
        return orders_data

    # Declare all loop variables with C types for performance
    cdef Py_ssize_t i
    cdef float64_t current_pos_target
    cdef int64_t potential_trade_exec_time_ns
    cdef float64_t potential_trade_exec_price
    cdef bint is_execution_possible
    cdef float64_t previous_pos_target
    cdef int32_t new_direction
    cdef bint is_closing_signal, is_reversing_signal
    cdef int64_t exit_time_ns
    cdef float64_t exit_price
    cdef int result_code # Standard C int for the result code
    cdef int64_t old_leg_exit_time_ns
    cdef float64_t old_leg_exit_price
    cdef int result_code_old_leg

    for i in range(n):
        current_pos_target = pos_values[i]
        potential_trade_exec_time_ns = trade_action_times_ns[i]
        potential_trade_exec_price = trade_action_prices[i]

        # Check for valid execution time (must be > 0, NaT is np.iinfo(np.int64).min)
        # and valid price (not NaN)
        is_execution_possible = (potential_trade_exec_time_ns > 0) and \
                                (not isnan(potential_trade_exec_price))

        previous_pos_target = 0.0
        if i > 0:
            previous_pos_target = pos_values[i - 1]

        if not has_active_trade:
            if previous_pos_target == 0.0 and current_pos_target != 0.0:
                if is_execution_possible:
                    active_entry_time_ns = potential_trade_exec_time_ns
                    active_entry_price = potential_trade_exec_price
                    new_direction = 0 # Explicitly int32
                    if current_pos_target > 0.0: new_direction = 1
                    elif current_pos_target < 0.0: new_direction = -1

                    if new_direction != 0:
                        active_direction = new_direction
                        has_active_trade = True
        elif has_active_trade: # An active trade exists
            is_closing_signal = (current_pos_target == 0.0 and
                                 previous_pos_target == active_direction)
            is_reversing_signal = (current_pos_target == -active_direction and
                                   previous_pos_target == active_direction)

            if is_closing_signal:
                exit_time_ns = active_entry_time_ns
                exit_price = NAN_CY # Use our defined NaN
                if is_execution_possible:
                    exit_time_ns = potential_trade_exec_time_ns
                    exit_price = potential_trade_exec_price

                result_code = _determine_trade_result_cy(
                    active_entry_price, exit_price, active_direction)
                
                # Appending a Python tuple. Cython converts C types to Python objects.
                orders_data.append((
                    active_entry_time_ns,
                    active_entry_price,
                    <int64_t>1,  # Explicit cast for buy_cnt to match Numba's np.int64(1)
                    exit_time_ns,
                    exit_price,
                    result_code,
                    active_direction))
                has_active_trade = False
                active_direction = 0

            elif is_reversing_signal:
                old_leg_exit_time_ns = active_entry_time_ns
                old_leg_exit_price = NAN_CY
                if is_execution_possible:
                    old_leg_exit_time_ns = potential_trade_exec_time_ns
                    old_leg_exit_price = potential_trade_exec_price

                result_code_old_leg = _determine_trade_result_cy(
                    active_entry_price, old_leg_exit_price, active_direction)
                orders_data.append((
                    active_entry_time_ns, active_entry_price, <int64_t>1,
                    old_leg_exit_time_ns, old_leg_exit_price,
                    result_code_old_leg, active_direction))

                if is_execution_possible:
                    active_entry_time_ns = potential_trade_exec_time_ns
                    active_entry_price = potential_trade_exec_price
                    active_direction = <int32_t>(-active_direction) # Explicit cast
                else:
                    has_active_trade = False
                    active_direction = 0
    return orders_data