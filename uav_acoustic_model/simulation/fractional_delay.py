r"""Two independent zero-extended fractional-delay implementations.

Sign convention
---------------
A positive delay ``delta`` means ``y(t) = x(t-delta)``. In sample units,
``delay_samples = delta * fs`` and positive values move a waveform to the
right. The public functions accept delays in samples; callers with seconds
must multiply by the sampling rate explicitly.

Boundary model and valid region
-------------------------------
The finite input is treated as zero outside ``0 <= n < len(x)``. Both
functions return a requested finite output interval starting at time index
zero. ``fractional_delay_valid_region`` gives the conservative interval in
which a windowed-sinc interpolator does not touch either finite-input edge.

Frequency-domain method
-----------------------
The input is embedded between zero guards, transformed, multiplied by
``exp(-j*2*pi*k*delay/Nfft)``, and transformed back. The default guard is at
least eight input lengths and 1024 samples. The FFT length includes the input,
both guards, the integer delay, and any requested output extension; the
central linear-time interval is extracted, so no circularly shifted tail is
returned.

Windowed-sinc FIR method
------------------------
The default odd-length Kaiser-windowed FIR has 129 taps. For
``delay = integer + fraction`` its causal coefficient array has intrinsic
group delay ``(L-1)/2 + fraction`` samples. The API crops the fixed
``(L-1)/2``-sample latency and prepends the integer part, so the net requested
delay is preserved. Samples outside the finite input are zero. The first and
last ``(L-1)/2`` source-support samples are excluded from the conservative
valid region.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve

DEFAULT_FIR_LENGTH = 129
DEFAULT_KAISER_BETA = 8.6
DEFAULT_FREQUENCY_PADDING_FACTOR = 8
MINIMUM_FREQUENCY_PADDING = 1024


def _signal_1d(signal: ArrayLike) -> NDArray[np.float64]:
    samples = np.asarray(signal, dtype=float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(samples)):
        raise ValueError("signal must be finite")
    return samples


def _delay_samples(delay_samples: float) -> float:
    delay = float(delay_samples)
    if not np.isfinite(delay) or delay < 0.0:
        raise ValueError("delay_samples must be finite and non-negative")
    return delay


def _output_length(input_length: int, output_length: int | None) -> int:
    length = input_length if output_length is None else int(output_length)
    if length < 1:
        raise ValueError("output_length must be positive")
    return length


def _integer_shift(
    signal: NDArray[np.float64],
    integer_delay: int,
    output_length: int,
) -> NDArray[np.float64]:
    output = np.zeros(output_length, dtype=float)
    count = min(signal.size, max(0, output_length - integer_delay))
    if count:
        output[integer_delay : integer_delay + count] = signal[:count]
    return output


def _is_integer_delay(delay: float) -> bool:
    return bool(np.isclose(delay, np.rint(delay), rtol=0.0, atol=8.0 * np.finfo(float).eps))


def frequency_domain_delay(
    signal: ArrayLike,
    delay_samples: float,
    *,
    output_length: int | None = None,
    padding_samples: int | None = None,
) -> NDArray[np.float64]:
    """Delay a real finite signal with a zero-padded Fourier phase ramp.

    ``padding_samples`` is the number of explicit zeros placed on each side
    before accounting for the requested delay/output extension. If omitted,
    it is ``max(1024, 8*len(signal))``. Integer and zero delays use an exact
    zero-filled shift rather than an FFT round trip.
    """

    samples = _signal_1d(signal)
    delay = _delay_samples(delay_samples)
    length = _output_length(samples.size, output_length)
    if _is_integer_delay(delay):
        return _integer_shift(samples, int(np.rint(delay)), length)

    if padding_samples is None:
        guard = max(
            MINIMUM_FREQUENCY_PADDING,
            DEFAULT_FREQUENCY_PADDING_FACTOR * samples.size,
        )
    else:
        guard = int(padding_samples)
        if guard < 1:
            raise ValueError("padding_samples must be positive")
    integer_extent = int(np.ceil(delay))
    output_extension = max(0, length - samples.size)
    required_length = (
        guard + samples.size + guard + integer_extent + output_extension
    )
    transform_length = next_fast_len(required_length)
    buffer = np.zeros(transform_length, dtype=float)
    buffer[guard : guard + samples.size] = samples
    frequencies = np.fft.rfftfreq(transform_length)
    phase_ramp = np.exp(-2j * np.pi * frequencies * delay)
    delayed = np.fft.irfft(np.fft.rfft(buffer) * phase_ramp, n=transform_length)
    return delayed[guard : guard + length].copy()


def windowed_sinc_kernel(
    fractional_delay: float,
    *,
    fir_length: int = DEFAULT_FIR_LENGTH,
    kaiser_beta: float = DEFAULT_KAISER_BETA,
) -> NDArray[np.float64]:
    """Return a DC-normalised Kaiser-windowed sinc for a delay in ``[0, 1)``."""

    fraction = float(fractional_delay)
    if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError("fractional_delay must lie in [0, 1)")
    length = int(fir_length)
    if length < 3 or length % 2 == 0:
        raise ValueError("fir_length must be an odd integer of at least 3")
    beta = float(kaiser_beta)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("kaiser_beta must be finite and non-negative")
    half = (length - 1) // 2
    offsets = np.arange(-half, half + 1, dtype=float)
    kernel = np.sinc(offsets - fraction) * np.kaiser(length, beta)
    kernel /= np.sum(kernel)
    return kernel


def windowed_sinc_delay(
    signal: ArrayLike,
    delay_samples: float,
    *,
    output_length: int | None = None,
    fir_length: int = DEFAULT_FIR_LENGTH,
    kaiser_beta: float = DEFAULT_KAISER_BETA,
) -> NDArray[np.float64]:
    """Delay a zero-extended finite signal with a windowed-sinc FIR.

    The fixed FIR latency ``(fir_length-1)/2`` is compensated in the returned
    indexing. The remaining net group delay is exactly the requested integer
    plus fractional delay within the FIR approximation error.
    """

    samples = _signal_1d(signal)
    delay = _delay_samples(delay_samples)
    length = _output_length(samples.size, output_length)
    fir_length_checked = int(fir_length)
    if fir_length_checked < 3 or fir_length_checked % 2 == 0:
        raise ValueError("fir_length must be an odd integer of at least 3")
    if _is_integer_delay(delay):
        return _integer_shift(samples, int(np.rint(delay)), length)

    integer_delay = int(np.floor(delay))
    fraction = delay - integer_delay
    kernel = windowed_sinc_kernel(
        fraction,
        fir_length=fir_length_checked,
        kaiser_beta=kaiser_beta,
    )
    half = (fir_length_checked - 1) // 2
    integer_delayed = np.pad(samples, (integer_delay, 0))
    filtered = fftconvolve(integer_delayed, kernel, mode="full")
    output = np.zeros(length, dtype=float)
    available = max(0, min(length, filtered.size - half))
    if available:
        output[:available] = filtered[half : half + available]
    return output


def fractional_delay_valid_region(
    input_length: int,
    delay_samples: ArrayLike,
    *,
    output_length: int | None = None,
    boundary_guard_samples: int = (DEFAULT_FIR_LENGTH - 1) // 2,
) -> tuple[int, int]:
    """Return the common half-open valid region ``[start, stop)``.

    Every returned time index maps at least ``boundary_guard_samples`` inside
    the finite source support for every supplied delay. This conservative
    definition applies to either implementation and makes cross-method error
    comparisons independent of zero-extension transients.
    """

    length = int(input_length)
    if length < 1:
        raise ValueError("input_length must be positive")
    output = _output_length(length, output_length)
    guard = int(boundary_guard_samples)
    if guard < 0:
        raise ValueError("boundary_guard_samples must be non-negative")
    delays = np.atleast_1d(np.asarray(delay_samples, dtype=float))
    if delays.size == 0 or not np.all(np.isfinite(delays)) or np.any(delays < 0.0):
        raise ValueError("delay_samples must contain finite non-negative values")
    start = int(np.max(np.ceil(delays + guard)))
    stop = int(np.min(np.ceil(delays + length - guard)))
    start = min(max(start, 0), output)
    stop = min(max(stop, start), output)
    return start, stop
