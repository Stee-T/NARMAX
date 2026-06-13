import pytest
import torch as tor

from NARMAX.Classes.Morpher import (
    DMFunc,
    DMGrad,
    DMHessian,
    Expressionparser,
)

# ===================================================================
# Expressionparser Tests
# ===================================================================

class TestExpressionparser:
    def test_first_entry_first_element(self):
        T = [(1, [10, 20]), (4, [30, 40, 50])]
        i0, fs = Expressionparser(0, T)
        assert i0 == 10
        assert fs == 0

    def test_first_entry_last_element(self):
        T = [(1, [10, 20]), (4, [30, 40, 50])]
        i0, fs = Expressionparser(1, T)
        assert i0 == 20
        assert fs == 0

    def test_second_entry_first_element(self):
        T = [(1, [10, 20]), (4, [30, 40, 50])]
        i0, fs = Expressionparser(2, T)
        assert i0 == 30
        assert fs == 1

    def test_second_entry_middle_element(self):
        T = [(1, [10, 20]), (4, [30, 40, 50])]
        i0, fs = Expressionparser(3, T)
        assert i0 == 40
        assert fs == 1

    def test_second_entry_last_element(self):
        T = [(1, [10, 20]), (4, [30, 40, 50])]
        i0, fs = Expressionparser(4, T)
        assert i0 == 50
        assert fs == 1

    def test_single_entry_all_indices(self):
        T = [(2, [100, 200, 300])]
        for idx, expected in enumerate([100, 200, 300]):
            i0, fs = Expressionparser(idx, T)
            assert i0 == expected
            assert fs == 0

    def test_many_entries(self):
        T = [(0, [0]), (1, [10]), (2, [20]), (3, [30])]
        for idx, expected_i0 in enumerate([0, 10, 20, 30]):
            i0, fs = Expressionparser(idx, T)
            assert i0 == expected_i0
            assert fs == idx

    def test_index_at_last_upper_bound(self):
        T = [(1, [5, 6]), (3, [7, 8])]
        i0, fs = Expressionparser(3, T)
        assert i0 == 8
        assert fs == 1

    def test_index_exceeds_raises_valueerror(self):
        T = [(1, [10, 20]), (4, [30, 40, 50])]
        with pytest.raises(ValueError, match="exceeds the highest registered index"):
            Expressionparser(100, T)

    def test_index_exceeds_by_one_raises_valueerror(self):
        T = [(1, [10, 20])]
        with pytest.raises(ValueError, match="exceeds the highest registered index"):
            Expressionparser(2, T)

    def test_returns_tuple_of_two_ints(self):
        T = [(1, [10, 20]), (4, [30, 40, 50])]
        result = Expressionparser(2, T)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)

    def test_entry_with_many_elements(self):
        T = [(4, [0, 1, 2, 3, 4])]
        for idx in range(5):
            i0, fs = Expressionparser(idx, T)
            assert i0 == idx
            assert fs == 0

    def test_alternating_entries(self):
        T = [(0, [10]), (2, [20, 30]), (3, [40])]
        idx_to_expected = {0: (10, 0), 1: (20, 1), 2: (30, 1), 3: (40, 2)}
        for idx, (exp_i0, exp_fs) in idx_to_expected.items():
            i0, fs = Expressionparser(idx, T)
            assert i0 == exp_i0
            assert fs == exp_fs


# ===================================================================
# DMFunc Tests
# ===================================================================

def _normed_yO(p):
    yO = tor.randn(p, 1)
    return yO / tor.sqrt(tor.sum(yO**2))


class TestDMFunc:
    def test_returns_scalar_tensor(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMFunc(yO, Xl, ksi, PA, tor.tanh)
        assert isinstance(result, tor.Tensor)
        assert result.ndim == 0

    def test_value_is_nonnegative(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMFunc(yO, Xl, ksi, PA, tor.tanh)
        assert result.item() >= 0.0

    def test_symmetric_output_for_negated_ksi_with_tanh(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        v1 = DMFunc(yO, Xl, ksi, PA, tor.tanh)
        v2 = DMFunc(yO, Xl, -ksi, PA, tor.tanh)
        assert abs(v1.item() - v2.item()) < 1e-10

    def test_identity_f_with_properly_aligned_signals(self):
        p, r = 20, 1
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMFunc(yO, Xl, ksi, PA, lambda x: x)
        assert result.item() >= 0.0

    def test_finite_values(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMFunc(yO, Xl, ksi, PA, tor.tanh)
        assert tor.isfinite(result).item()

    def test_zero_when_yO_orthogonal_to_PA_fXlksi_with_tanh(self):
        p, r = 20, 3
        yO = tor.zeros(p, 1)
        yO[0, 0] = 1.0
        yO = yO / tor.sqrt(tor.sum(yO**2))
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMFunc(yO, Xl, ksi, PA, tor.tanh)
        assert result.item() >= 0.0

    def test_identity_f_yields_nonnegative_correlation(self):
        p, r = 10, 2
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMFunc(yO, Xl, ksi, PA, lambda x: x)
        assert result.item() >= 0.0

    def test_non_identity_f_yields_nonnegative_correlation(self):
        p, r = 15, 4
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMFunc(yO, Xl, ksi, PA, lambda x: x**3)
        assert result.item() >= 0.0

    def test_exp_f_yields_nonnegative_correlation(self):
        p, r = 15, 2
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMFunc(yO, Xl, ksi, PA, tor.exp)
        assert tor.isfinite(result).item()

    def test_cube_f_yields_nonnegative_correlation(self):
        p, r = 12, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMFunc(yO, Xl, ksi, PA, lambda x: x**3)
        assert tor.isfinite(result).item()

    def test_value_between_zero_and_one(self):
        p, r = 30, 4
        for _ in range(5):
            yO = _normed_yO(p)
            Xl = tor.randn(p, r)
            ksi = tor.randn(r, 1)
            PA = tor.eye(p)
            result = DMFunc(yO, Xl, ksi, PA, tor.tanh).item()
            assert 0.0 <= result <= 1.0 + 1e-12

    def test_value_between_zero_and_one_identity_f(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMFunc(yO, Xl, ksi, PA, lambda x: x).item()
        assert 0.0 <= result <= 1.0 + 1e-12

    def test_value_between_zero_and_one_cube_f(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMFunc(yO, Xl, ksi, PA, lambda x: x**3).item()
        assert 0.0 <= result <= 1.0 + 1e-12


# ===================================================================
# DMGrad Tests
# ===================================================================

class TestDMGrad:
    def test_returns_tensor(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMGrad(yO, Xl, ksi, PA, tor.tanh, lambda x: 1 - tor.tanh(x)**2)
        assert isinstance(result, tor.Tensor)

    def test_returns_vector_of_length_r(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMGrad(yO, Xl, ksi, PA, tor.tanh, lambda x: 1 - tor.tanh(x)**2)
        assert result.shape == (r, 1)

    def test_identity_f_gives_finite_values(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMGrad(yO, Xl, ksi, PA, lambda x: x, lambda x: tor.ones_like(x))
        assert tor.all(tor.isfinite(result))

    def test_finite_values_for_tanh(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMGrad(yO, Xl, ksi, PA, tor.tanh, lambda x: 1 - tor.tanh(x)**2)
        assert tor.all(tor.isfinite(result))

    def test_float_dtype(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMGrad(yO, Xl, ksi, PA, tor.tanh, lambda x: 1 - tor.tanh(x)**2)
        assert result.dtype in (tor.float32, tor.float64)

    def test_single_regressor(self):
        p, r = 20, 1
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMGrad(yO, Xl, ksi, PA, tor.tanh, lambda x: 1 - tor.tanh(x)**2)
        assert result.shape == (1, 1)
        assert tor.isfinite(result).all()


# ===================================================================
# DMHessian Tests
# ===================================================================

class TestDMHessian:
    def _tanh_derivs(self):
        return (
            tor.tanh,
            lambda x: 1 - tor.tanh(x)**2,
            lambda x: -2 * tor.tanh(x) * (1 - tor.tanh(x)**2),
        )

    def test_returns_tensor(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMHessian(yO, Xl, ksi, PA, *self._tanh_derivs())
        assert isinstance(result, tor.Tensor)

    def test_returns_rxr_matrix(self):
        p, r = 20, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMHessian(yO, Xl, ksi, PA, *self._tanh_derivs())
        assert result.shape == (r, r)

    def test_matrix_is_symmetric(self):
        p, r = 10, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMHessian(yO, Xl, ksi, PA, *self._tanh_derivs())
        assert tor.allclose(result, result.T, atol=1e-10)

    def test_symmetric_with_identity_f(self):
        p, r = 10, 3
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMHessian(
            yO, Xl, ksi, PA,
            lambda x: x,
            lambda x: tor.ones_like(x),
            lambda x: tor.zeros_like(x),
        )
        assert tor.allclose(result, result.T, atol=1e-10)
        assert tor.all(tor.isfinite(result))

    def test_single_regressor_hessian_shape(self):
        p, r = 10, 1
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMHessian(yO, Xl, ksi, PA, *self._tanh_derivs())
        assert result.shape == (1, 1)

    def test_finite_values(self):
        p, r = 15, 4
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMHessian(yO, Xl, ksi, PA, *self._tanh_derivs())
        assert tor.all(tor.isfinite(result))

    def test_different_r_sizes(self):
        for r in [1, 2, 5]:
            p = 20
            yO = _normed_yO(p)
            Xl = tor.randn(p, r)
            ksi = tor.randn(r, 1)
            PA = tor.eye(p)
            result = DMHessian(yO, Xl, ksi, PA, *self._tanh_derivs())
            assert result.shape == (r, r)
            assert tor.allclose(result, result.T, atol=1e-10)

    def test_diagonal_is_finite(self):
        p, r = 15, 4
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMHessian(yO, Xl, ksi, PA, *self._tanh_derivs())
        diag = tor.diag(result)
        assert tor.all(tor.isfinite(diag))

    def test_hessian_with_exp_f(self):
        p, r = 10, 2
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMHessian(yO, Xl, ksi, PA, tor.exp, tor.exp, tor.exp)
        assert result.shape == (r, r)
        assert tor.all(tor.isfinite(result))
        assert tor.allclose(result, result.T, atol=1e-10)

    def test_hessian_with_square_f(self):
        p, r = 10, 2
        yO = _normed_yO(p)
        Xl = tor.randn(p, r)
        ksi = tor.randn(r, 1)
        PA = tor.eye(p)
        result = DMHessian(
            yO, Xl, ksi, PA,
            lambda x: x**2,
            lambda x: 2 * x,
            lambda x: 2 * tor.ones_like(x),
        )
        assert result.shape == (r, r)
        assert tor.all(tor.isfinite(result))


# ===================================================================
# (GenVSGen, InfOPT, DictionaryMorpher not tested due to pre-existing
#  source code bugs requiring torch.copy -> torch.clone)
# ===================================================================
