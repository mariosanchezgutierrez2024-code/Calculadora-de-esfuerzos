"""Pruebas: pytest test_calculadora.py"""
import numpy as np
import pytest
from calculadora_esfuerzos import (calcular, tensor_desde_componentes as tc,
                                   validar_entradas, veredicto)


def test_traccion_uniaxial_falla_en_fluencia():
    r = calcular(tc(250, 0, 0, 0, 0, 0), True, 250, 0, 0)
    assert r["fs_tresca"] == pytest.approx(1.0)
    assert r["fs_vm"] == pytest.approx(1.0)
    assert veredicto(r["fs_vm"]) == "Falla"


def test_placa_A36_uniaxial_125():
    r = calcular(tc(125, 0, 0, 0, 0, 0), True, 250, 0, 0)
    assert r["fs_tresca"] == pytest.approx(2.0) and r["fs_vm"] == pytest.approx(2.0)


def test_cortante_puro():
    r = calcular(tc(0, 0, 0, 100, 0, 0), True, 250, 0, 0)
    assert r["princ_user"] == pytest.approx([100, 0, -100])
    assert r["fs_tresca"] == pytest.approx(250 / 200)
    assert r["fs_vm"] == pytest.approx(250 / (np.sqrt(3) * 100))
    assert r["tau_max"] == pytest.approx(100)


def test_hidrostatico_fs_infinito():
    r = calcular(tc(-50, -50, -50, 0, 0, 0), True, 250, 20, 30)
    assert np.isinf(r["fs_tresca"]) and np.isinf(r["fs_vm"])
    assert np.isinf(r["fs_mc"]) and np.isinf(r["fs_dp"])


def test_convencion_equivalente():
    a = calcular(tc(80, -30, 10, 20, 5, -15), True, 250, 20, 30)
    b = calcular(-tc(80, -30, 10, 20, 5, -15), False, 250, 20, 30)
    for k in ("fs_tresca", "fs_vm", "fs_mc", "fs_dp", "J2", "tau_max"):
        assert a[k] == pytest.approx(b[k])
    # mismo estado físico: cada convención lo lee con signo opuesto
    assert b["princ_user"] == pytest.approx(-a["princ_user"][::-1])


def test_mohr_coulomb_y_dp_en_falla_triaxial():
    c, phi, s3 = 20.0, 30.0, 100.0           # kPa, grados, compresión +
    N = np.tan(np.radians(60)) ** 2
    s1 = s3 * N + 2 * c * np.sqrt(N)
    r = calcular(np.diag([s1, s3, s3]), False, 250, c, phi)
    assert r["fs_mc"] == pytest.approx(1.0)
    assert r["fs_dp"] == pytest.approx(1.0)   # calibración compresión triaxial
    assert r["s1_falla_user"] == pytest.approx(s1)


def test_validaciones():
    asim = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1.0]])
    assert validar_entradas(asim, 250, 0, 30)
    assert validar_entradas(np.eye(3), 0, 0, 30)
    assert validar_entradas(np.eye(3), 250, 0, 90)
    assert not validar_entradas(np.eye(3), 250, 10, 0)


def test_traccion_neta_activa_advertencia():
    r = calcular(np.diag([10, 5, 1.0]), True, 250, 20, 30)
    assert r["solo_traccion"]


# ---- Casos del documento (esperado, tolerancia 2 decimales / FS 3) -------
def _cerca(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


M = lambda *f: np.array(f, float)
ACERO = {  # caso: (tensor, sigma_y, FS_Tresca, FS_vM, J2)
    "C1": (np.diag([250, 0, 0.]), 250, 1.000, 1.000, 20833.33),
    "C2": (tc(0, 0, 0, 100, 0, 0), 250, 1.250, 1.443, 10000.00),
    "C4": (tc(160, 0, 0, 70, 0, 0), 250, 1.176, 1.245, 13433.33),
    "C5": (M([220, 60, 0], [60, 140, 0], [0, 0, -80]), 350, 1.054, 1.213, 27733.33),
    "C6": (M([120, 40, -30], [40, -60, 25], [-30, 25, 80]), 300, 1.403, 1.577, 12058.33),
}
GEO = {  # caso: (tensor, tension+, c, phi, FS_MC, FS_DP, s1_falla(comp +))
    "C7": (np.diag([392.3, 120, 120]), False, 18, 28, 1.000, 1.000, 392.29),
    "C8": (np.diag([120, 30, 30.]), False, 45, 0, 1.000, 1.000, 120.00),
    "C9": (np.diag([-40, -25, -15.]), True, 12, 32, 1.980, 2.709, 92.12),
    "C10": (M([90, 25, 0], [25, 150, 0], [0, 0, 100]), False, 10, 30, 1.758, 2.223, 277.49),
    "C11": (M([200, 40, -30], [40, 150, 25], [-30, 25, 260]), False, 10, 30, 1.377, 1.948, 389.67),
}


def test_casos_acero_documento():
    for T, sy, ft, fv, j2 in ACERO.values():
        r = calcular(T, True, sy, 0, 0)
        _cerca(r["fs_tresca"], ft, 5e-4); _cerca(r["fs_vm"], fv, 5e-4)
        _cerca(r["J2"], j2, 5e-3)


def test_casos_geo_documento():
    for T, tp, c, phi, fmc, fdp, s1f in GEO.values():
        r = calcular(T, tp, 250, c, phi)
        _cerca(r["fs_mc"], fmc, 5e-4); _cerca(r["fs_dp"], fdp, 5e-4)
        _cerca(r["s1_falla_c"], s1f, 5e-3)


def test_c3_hidrostatico_y_c8_sigma_y_90():
    r = calcular(np.diag([-100, -100, -100.]), True, 250, 10, 30)
    assert np.isinf(r["fs_mc"]) and np.isinf(r["fs_dp"])
    _cerca(r["s1_falla_c"], 334.64, 5e-3)       # compresión + aunque la conv. sea tensión +
    r = calcular(np.diag([120, 30, 30.]), False, 90, 45, 0)
    _cerca(r["fs_tresca"], 1.0, 1e-9); _cerca(r["fs_vm"], 1.0, 1e-9)
    _cerca(r["alpha"], 0.0, 1e-12)