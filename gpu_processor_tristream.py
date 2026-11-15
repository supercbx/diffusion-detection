from collections.abc import Mapping
import os
from numbers import Integral, Real
import warnings

import numpy as np
from PIL import Image
from scipy import ndimage as scipy_ndimage

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
except (ImportError, OSError):
    cp = None
    cp_ndimage = None

CUPY_AVAILABLE = cp is not None


class GPUProcessor_TriStream:


    def __init__(self, config=None, use_gpu=True):
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or None")
        supplied = dict(config) if config is not None else {}
        self.image_size = self._shape(supplied.get("image_size", (224, 224)), minimum=2)
        self.amp_strength = self._strength(
            supplied.get("amp_strength", supplied.get("alpha", 0.02)), "amp_strength"
        )
        self.phase_strength = self._strength(
            supplied.get("phase_strength", supplied.get("beta", 0.05)), "phase_strength"
        )
        window = supplied.get("phase_window", 3)
        if isinstance(window, (bool, np.bool_)) or not isinstance(window, Integral):
            raise ValueError("phase_window must be a positive odd integer")
        if window < 1 or window % 2 == 0:
            raise ValueError("phase_window must be a positive odd integer")
        self.phase_window = int(window)
        self.gaussian_sigma = self._strength(supplied.get("gaussian_sigma", 1.0), "gaussian_sigma")
        if self.gaussian_sigma == 0:
            raise ValueError("gaussian_sigma must be positive")
        self.seed = self._seed(supplied.get("seed", 42))
        self.config = dict(supplied, image_size=self.image_size,
                           amp_strength=self.amp_strength, phase_strength=self.phase_strength,
                           phase_window=self.phase_window, gaussian_sigma=self.gaussian_sigma,
                           seed=self.seed)
        self._rng = np.random.default_rng(self.seed)

        self.use_gpu = False
        if use_gpu and CUPY_AVAILABLE:
            try:
                self.use_gpu = cp.cuda.runtime.getDeviceCount() > 0
                if self.use_gpu:

                    cp.zeros(1, dtype=cp.float64)
            except Exception as exc:
                self.use_gpu = False
                warnings.warn(f"CuPy device unavailable; using CPU: {exc}", RuntimeWarning)
        self.xp = cp if self.use_gpu else np
        self.ndimage = cp_ndimage if self.use_gpu else scipy_ndimage

    @staticmethod
    def _shape(shape, minimum=1):
        try:
            values = tuple(shape)
        except TypeError as exc:
            raise ValueError("shape must be (height, width)") from exc
        if len(values) != 2 or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < minimum
            for v in values
        ):
            raise ValueError(f"shape must contain two integers >= {minimum}")
        return tuple(int(v) for v in values)

    @staticmethod
    def _strength(value, name):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
        return float(value)

    @staticmethod
    def _seed(seed):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        return int(seed)

    def _to_cpu(self, array):
        return cp.asnumpy(array) if self.use_gpu else np.asarray(array)

    def _to_gpu(self, array):

        return self.xp.asarray(array)

    def _prepare_image(self, path_or_arr):
        if isinstance(path_or_arr, (str, os.PathLike)):

            with Image.open(path_or_arr) as image:
                array = np.array(image.convert("RGB"), dtype=np.uint8)
        elif isinstance(path_or_arr, np.ndarray):
            array = path_or_arr
        else:
            raise TypeError("image must be a path or a NumPy RGB/grayscale array")
        if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] != 3):
            raise ValueError("image must have shape (H, W) or (H, W, 3)")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("image must not be empty")
        if array.dtype == np.uint8:
            rgb = array.astype(np.float32) / 255.0
        elif np.issubdtype(array.dtype, np.floating):
            if not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
                raise ValueError("floating-point image values must be finite and in [0, 1]")
            rgb = array.astype(np.float32, copy=True)
        else:
            raise TypeError("image dtype must be uint8 or floating point in [0, 1]")
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.shape[:2] != self.image_size:
            height, width = self.image_size
            rgb = np.stack([
                np.asarray(Image.fromarray(rgb[..., c]).resize(
                    (width, height), resample=Image.Resampling.BILINEAR
                ), dtype=np.float32)
                for c in range(3)
            ], axis=-1)
        gray = np.sum(rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32), axis=-1)
        return rgb, np.ascontiguousarray(gray, dtype=np.float32)

    def preprocess_image(self, image):

        _, gray = self._prepare_image(image)
        return self.xp.asarray(gray)

    def _map(self, array, name, minimum=1):
        array = self.xp.asarray(array, dtype=self.xp.float64)
        if array.ndim != 2 or min(array.shape) < minimum:
            raise ValueError(f"{name} must be a 2D array with dimensions >= {minimum}")
        if not bool(self.xp.all(self.xp.isfinite(array))):
            raise ValueError(f"{name} must contain only finite values")
        return array

    def _spectrum(self, image):
        image = self._map(image, "image", minimum=2)
        return self.xp.fft.fftshift(self.xp.fft.fft2(image))

    def frequency_mask(self, shape):

        height, width = self._shape(shape)
        y, x = self.xp.ogrid[:height, :width]
        radius = self.xp.sqrt((y - height // 2) ** 2 + (x - width // 2) ** 2)
        r_max = float(radius.max())
        return radius / r_max if r_max else self.xp.zeros((height, width), dtype=self.xp.float64)

    def phase_dispersion(self, phase):

        phase = self._map(phase, "phase", minimum=2)
        gradients = self.xp.gradient(phase, edge_order=1)
        variance = self.xp.zeros_like(phase)
        for gradient in gradients:
            mean = self.ndimage.uniform_filter(gradient, size=self.phase_window, mode="reflect")
            mean_square = self.ndimage.uniform_filter(
                gradient * gradient, size=self.phase_window, mode="reflect"
            )
            variance += mean_square - mean * mean

        return self.xp.sqrt(self.xp.maximum(variance, 0.0))

    def _perturb_amplitude(self, amplitude, strength, rng):

        mask = self._to_cpu(self.frequency_mask(amplitude.shape))
        multiplier = 1.0 + strength * mask * rng.normal(size=amplitude.shape)
        invalid = multiplier <= 0
        while np.any(invalid):
            multiplier[invalid] = 1.0 + strength * mask[invalid] * rng.normal(size=int(invalid.sum()))
            invalid = multiplier <= 0

        return amplitude * self.xp.asarray(multiplier)

    def inject_amplitude_noise(self, image, strength, *, rng=None):


        strength = self._strength(strength, "strength")
        amplitude = self.xp.abs(self._spectrum(image))
        return self._perturb_amplitude(amplitude, strength, self._rng if rng is None else rng)

    def denoise_amplitude_response(self, amplitude):

        amplitude = self._map(amplitude, "amplitude")
        if bool(self.xp.any(amplitude < 0)):
            raise ValueError("amplitude must be nonnegative")
        return self.ndimage.median_filter(amplitude, size=3, mode="reflect")

    def _perturb_phase(self, spectrum, strength, rng):
        phase = self.xp.angle(spectrum)
        delta = self.xp.asarray(rng.uniform(-1.0, 1.0, size=phase.shape))
        dispersed = phase + strength * self.phase_dispersion(phase) * delta
        wrapped = (dispersed + self.xp.pi) % (2 * self.xp.pi) - self.xp.pi
        modified = self.xp.abs(spectrum) * self.xp.exp(1j * wrapped)
        return self.xp.fft.ifft2(self.xp.fft.ifftshift(modified)).real

    def inject_phase_noise(self, image, strength, *, rng=None):

        strength = self._strength(strength, "strength")
        return self._perturb_phase(self._spectrum(image), strength, self._rng if rng is None else rng)

    def denoise_phase_response(self, spatial):

        spatial = self._map(spatial, "spatial")

        coordinates = self.xp.arange(-2, 3, dtype=self.xp.float64)
        weights = self.xp.exp(-0.5 * (coordinates / self.gaussian_sigma) ** 2)
        weights /= weights.sum()
        restored = self.ndimage.convolve1d(spatial, weights, axis=0, mode="reflect")
        return self.ndimage.convolve1d(restored, weights, axis=1, mode="reflect")

    def process_single_image(self, path_or_arr, is_train_mode=True, seed=None):


        if seed is not None:
            rng = np.random.default_rng(self._seed(seed))
        elif is_train_mode:
            rng = self._rng
        else:
            rng = np.random.default_rng(self.seed)
        rgb, gray_cpu = self._prepare_image(path_or_arr)
        gray = self.xp.asarray(gray_cpu)
        spectrum = self._spectrum(gray)
        amplitude = self.xp.abs(spectrum)
        perturbed_amplitude = self._perturb_amplitude(amplitude, self.amp_strength, rng)
        amplitude_residual = self.xp.abs(
            amplitude - self.denoise_amplitude_response(perturbed_amplitude)
        )
        spatial = self._perturb_phase(spectrum, self.phase_strength, rng)
        phase_residual = self.xp.abs(gray - self.denoise_phase_response(spatial))
        return (
            np.rint(rgb * 255.0).astype(np.uint8),
            self._to_cpu(amplitude_residual).astype(np.float32),
            self._to_cpu(phase_residual).astype(np.float32),
        )
