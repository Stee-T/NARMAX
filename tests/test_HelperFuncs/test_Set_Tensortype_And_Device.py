import pytest
import torch as tor
from NARMAX.HelperFuncs import Set_Tensortype_And_Device


class TestSetTensortypeAndDevice:
  '''Tests for Set_Tensortype_And_Device.'''

  VALID_DEVICES: set = { "cpu", "cuda", "cuda:0", "mps" }

  # ------------------------------------------------------------------
  # Fixtures
  # ------------------------------------------------------------------
  @pytest.fixture( autouse = True )
  def _save_and_restore_torch_defaults( self ) -> None:
    '''Save torch default device and dtype before each test and restore after.'''
    saved_device = tor.get_default_device()
    saved_dtype = tor.get_default_dtype()
    yield
    tor.set_default_device( saved_device )
    tor.set_default_dtype( saved_dtype )

  # ------------------------------------------------------------------
  # Smoke test
  # ------------------------------------------------------------------
  def test_smoke( self ) -> None:
    '''Function runs without raising.'''
    result = Set_Tensortype_And_Device()
    assert result is not None

  # ------------------------------------------------------------------
  # Return type
  # ------------------------------------------------------------------
  def test_returns_string( self ) -> None:
    '''Return value must be a string.'''
    result = Set_Tensortype_And_Device()
    assert isinstance( result, str ), f"Expected str, got { type( result ).__name__ }"

  # ------------------------------------------------------------------
  # Return value validity
  # ------------------------------------------------------------------
  def test_return_value_is_valid_device( self ) -> None:
    '''Return value must be one of the known device strings.'''
    result = Set_Tensortype_And_Device()
    assert result in self.VALID_DEVICES, f"Unexpected device '{ result }'"

  # ------------------------------------------------------------------
  # Side-effect: torch default device
  # ------------------------------------------------------------------
  def test_sets_default_device( self ) -> None:
    '''tor.get_default_device() must match the returned device string.'''
    result = Set_Tensortype_And_Device()
    current = tor.get_default_device()
    assert str( current ) == result, (
      f"tor.get_default_device() is '{ current }' but function returned '{ result }'"
    )

  # ------------------------------------------------------------------
  # Side-effect: torch default dtype
  # ------------------------------------------------------------------
  def test_sets_default_dtype_for_cpu_or_cuda( self ) -> None:
    '''On cpu or cuda the default dtype must be float64.'''
    result = Set_Tensortype_And_Device()
    if result.startswith( "cuda" ) or result == "cpu":
      assert tor.get_default_dtype() == tor.float64, (
        f"Expected float64 on { result }, got { tor.get_default_dtype() }"
      )

  def test_sets_default_dtype_for_mps( self ) -> None:
    '''On mps the default dtype must be float32.'''
    result = Set_Tensortype_And_Device()
    if result == "mps":
      assert tor.get_default_dtype() == tor.float32, (
        f"Expected float32 on mps, got { tor.get_default_dtype() }"
      )

  # ------------------------------------------------------------------
  # Warning message on the CPU path
  # ------------------------------------------------------------------
  def test_cpu_path_prints_warning( self, capsys ) -> None:
    '''When device is cpu, a warning must be printed to stderr/stdout.'''
    result = Set_Tensortype_And_Device()
    captured = capsys.readouterr()
    if result == "cpu":
      assert "hardware-accelerator" in captured.out, (
        "Expected CPU warning message to be printed"
      )
    else:
      # Non-cpu paths should not print that message
      assert "hardware-accelerator" not in captured.out

  # ------------------------------------------------------------------
  # Idempotency – calling multiple times yields the same result
  # ------------------------------------------------------------------
  def test_idempotent( self ) -> None:
    '''Calling the function twice must return the same device.'''
    first = Set_Tensortype_And_Device()
    second = Set_Tensortype_And_Device()
    assert first == second, f"First call returned '{ first }', second call returned '{ second }'"

  # ------------------------------------------------------------------
  # New tensors observe the defaults (integration check)
  # ------------------------------------------------------------------
  def test_new_tensor_uses_default_device_and_dtype( self ) -> None:
    '''A newly created empty tensor should be on the configured device/dtype.'''
    result = Set_Tensortype_And_Device()
    t = tor.zeros( 3 )
    assert str( t.device ) == result, (
      f"New tensor is on { t.device } but defaults were set to { result }"
    )
    if result.startswith( "cuda" ) or result == "cpu":
      assert t.dtype == tor.float64, (
        f"New tensor dtype is { t.dtype } but float64 expected on { result }"
      )
    elif result == "mps":
      assert t.dtype == tor.float32, (
        f"New tensor dtype is { t.dtype } but float32 expected on mps"
      )
