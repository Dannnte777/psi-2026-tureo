import numpy as np
import pytest

from psi_lab.senales import cuadrada, espectro, senoidal


def test_numero_de_muestras():
    _, x = senoidal(f=5, fs=100, duracion=2)
    assert x.size == 200

@pytest.mark.parametrize("f", [10.0, 50.0, 120.0])
def test_pico_en_la_frecuencia_correcta(f):
    _, x = senoidal(f=f, fs=1000, duracion=1)
    freqs, amp = espectro(x, fs=1000)
    assert freqs[np.argmax(amp)] == pytest.approx(f)

def test_amplitud_recuperada():
    x = senoidal(f=50, fs=1000, duracion=1, amplitud=3.0)[1]
    _freqs, amp = espectro(x, fs=1000)
    assert amp.max() == pytest.approx(3.0, rel=1e-6)


def test_onda_cuadrada_armonicos_impares():
    f_base = 50.0
    fs = 5000.0
    _, x = cuadrada(f=f_base, fs=fs, duracion=1.0)
    freqs, amp = espectro(x, fs)
    
    # Verificamos armónicos impares (1f, 3f, 5f) tienen picos significativos
    for k in [1, 3, 5]:
        freq_armonica = k * f_base
        idx = np.argmin(np.abs(freqs - freq_armonica))
        assert amp[idx] > 0.1  # Debe haber componente importante

    # Verificamos que en los armónicos pares (2f, 4f) la amplitud sea cercana a cero
    for k in [2, 4]:
        freq_armonica = k * f_base
        idx = np.argmin(np.abs(freqs - freq_armonica))
        assert amp[idx] < 0.1  # No debe haber picos relevantes en pares