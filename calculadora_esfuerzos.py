"""
Calculadora de esfuerzos y criterios de falla (Ingeniería Civil)
================================================================
Ejecutar:  streamlit run calculadora_esfuerzos.py
Pruebas:   pytest test_calculadora.py

Convención INTERNA: tensión positiva. Si el usuario elige "compresión +",
el tensor de entrada se multiplica por -1 al entrar y los resultados se
devuelven a la convención elegida al salir (factor `sgn`).
La lógica numérica está separada de la interfaz para poder probarla.
"""
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-9  # tolerancia relativa


# --------------------------------------------------------------------------
# 1. Utilidades y validaciones
# --------------------------------------------------------------------------
def tensor_desde_componentes(sxx, syy, szz, txy, txz, tyz):
    """Arma el tensor simétrico 3x3 desde sus 6 componentes."""
    return np.array([[sxx, txy, txz],
                     [txy, syy, tyz],
                     [txz, tyz, szz]], dtype=float)


def validar_entradas(T, sigma_y, c, phi_deg):
    """Devuelve una lista de mensajes de error (vacía si todo es válido)."""
    errores = []
    if not np.all(np.isfinite(T)):
        errores.append("El tensor contiene valores no numéricos (NaN/inf).")
    elif not np.allclose(T, T.T, rtol=TOL, atol=TOL * max(1.0, np.abs(T).max())):
        errores.append("El tensor no es simétrico: se requiere σij = σji.")
    if not sigma_y > 0:
        errores.append("El límite de fluencia σy debe ser > 0.")
    if not (0 <= phi_deg < 90):
        errores.append("El ángulo de fricción φ debe cumplir 0 ≤ φ < 90°.")
    if c < 0:
        errores.append("La cohesión c no puede ser negativa.")
    return errores


def fs_div(num, den, escala=1.0):
    """FS = num/den sin dividir entre cero: den≈0 -> ∞ si num>0, 0 si num<=0."""
    if den <= TOL * max(1.0, escala):
        return np.inf if num > 0 else 0.0
    return num / den


def veredicto(fs):
    """Resiste si FS > 1 (FS = 1 equivale a F = 0, es decir, falla)."""
    return "Resiste" if fs > 1 + TOL else "Falla"


# --------------------------------------------------------------------------
# 2. Cálculo completo
# --------------------------------------------------------------------------
def calcular(T_user, tension_positiva, sigma_y, c, phi_deg):
    """
    T_user, sigma_y y c deben venir en la MISMA unidad de esfuerzo.
    Devuelve un diccionario con esfuerzos, invariantes y criterios de falla.
    """
    sgn = 1.0 if tension_positiva else -1.0   # T_interno = sgn * T_usuario
    T = sgn * T_user                          # interno: tensión (+)
    I = np.eye(3)

    # --- Esfuerzos principales -------------------------------------------
    w = np.sort(np.linalg.eigvalsh(T))[::-1]  # internos, t1 >= t2 >= t3
    t1, t2, t3 = w
    escala = max(1.0, np.abs(w).max())
    # Direcciones: eigh da vectores; se ordenan de mayor a menor en la
    # convención del usuario (σ1 >= σ2 >= σ3 tal como él los lee)
    wv, V = np.linalg.eigh(T)
    idx = np.argsort(sgn * wv)[::-1]
    princ_user = sgn * wv[idx]
    dirs = V[:, idx]                          # columnas = direcciones

    # --- Invariantes (en la convención del usuario) ----------------------
    I1 = np.trace(T_user)
    I2 = 0.5 * (I1 ** 2 - np.trace(T_user @ T_user))
    I3 = np.linalg.det(T_user)

    # --- Presión media, desviador, J2, J3 --------------------------------
    I1_int = sgn * I1
    p = -I1_int / 3.0                         # presión media (compresión +)
    S_user = T_user - (I1 / 3.0) * I          # desviador (conv. usuario)
    J2 = 0.5 * np.sum(S_user * S_user)        # invariante al signo
    J3 = np.linalg.det(S_user)
    sqJ2 = np.sqrt(J2)
    tau_oct = np.sqrt(2.0 * J2 / 3.0)
    tau_max = (t1 - t3) / 2.0

    # --- Tresca ----------------------------------------------------------
    dif = t1 - t3
    fs_tresca = fs_div(sigma_y, dif, escala)

    # --- Von Mises -------------------------------------------------------
    s_vm = np.sqrt(3.0 * J2)
    fs_vm = fs_div(sigma_y, s_vm, escala)

    # --- Mohr-Coulomb (trabaja con compresión positiva) ------------------
    phi = np.radians(phi_deg)
    s1c, s3c = -t3, -t1                       # σ1c >= σ3c (compresión +)
    N_phi = np.tan(np.pi / 4 + phi / 2) ** 2
    s1_falla_c = s3c * N_phi + 2 * c * np.sqrt(N_phi)
    theta = 45.0 + phi_deg / 2.0              # ángulo del plano de falla
    sig_m = (s1c + s3c) / 2.0
    R = (s1c - s3c) / 2.0
    fs_mc = fs_div(c * np.cos(phi) + sig_m * np.sin(phi), R, escala)
    # a la convención del usuario: v_user = -sgn * v_compresion
    s1_falla_user = -sgn * s1_falla_c

    # --- Drucker-Prager (calibrado a compresión triaxial) ----------------
    alpha = 2 * np.sin(phi) / (np.sqrt(3) * (3 - np.sin(phi)))
    k = 6 * c * np.cos(phi) / (np.sqrt(3) * (3 - np.sin(phi)))
    F_dp = alpha * I1_int + sqJ2 - k          # F<0 resiste, F=0 falla
    fs_dp = fs_div(k - alpha * I1_int, sqJ2, escala)

    return dict(
        sgn=sgn, T_user=T_user, princ_user=princ_user, dirs=dirs, internos=w,
        I1=I1, I2=I2, I3=I3, p=p, S_user=S_user, J2=J2, J3=J3,
        tau_oct=tau_oct, tau_max=tau_max, escala=escala,
        dif_tresca=dif, fs_tresca=fs_tresca,
        s_vm=s_vm, fs_vm=fs_vm,
        N_phi=N_phi, s1_falla_user=s1_falla_user, s1_falla_c=s1_falla_c, theta=theta,
        sig_m_user=-sgn * sig_m, R=R, fs_mc=fs_mc,
        alpha=alpha, k=k, F_dp=F_dp, fs_dp=fs_dp,
        # tracción neta: todos los principales > 0 (internos)
        solo_traccion=bool(t3 > TOL * escala),
        hay_traccion=bool(t1 > TOL * escala),
    )


# --------------------------------------------------------------------------
# 3. Gráficas
# --------------------------------------------------------------------------
def fig_mohr(r, c, phi_deg, unidad):
    """Tres círculos de Mohr + envolvente de Mohr-Coulomb (si c o φ > 0)."""
    sgn = r["sgn"]
    s = r["princ_user"]
    th = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    pares = [(0, 2, "σ1–σ3"), (0, 1, "σ1–σ2"), (1, 2, "σ2–σ3")]
    for (i, j, nom), col in zip(pares, ["tab:blue", "tab:green", "tab:orange"]):
        cen, rad = (s[i] + s[j]) / 2, abs(s[i] - s[j]) / 2
        ax.plot(cen + rad * np.cos(th), rad * np.sin(th), color=col, label=nom)
    ax.plot(s, np.zeros(3), "ko", ms=4)

    if c > 0 or phi_deg > 0:
        span = max(np.ptp(s), r["escala"] * 0.2, 1.0)
        xs = np.linspace(s.min() - span, s.max() + span, 400)
        sig_c = -sgn * xs                      # compresión + en la envolvente
        tau = c + sig_c * np.tan(np.radians(phi_deg))
        m = tau >= 0
        ax.plot(xs[m], tau[m], "r--", label="Envolvente Mohr-Coulomb")
        ax.plot(xs[m], -tau[m], "r--")
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    conv = "tensión +" if sgn > 0 else "compresión +"
    ax.set_xlabel(f"σ [{unidad}] ({conv})"); ax.set_ylabel(f"τ [{unidad}]")
    ax.set_title("Círculos de Mohr"); ax.legend(); ax.grid(alpha=0.3)
    return fig


def fig_plano(r, sigma_y, unidad):
    """Hexágono de Tresca y elipse de Von Mises (σ3 = 0) con el estado actual.
    Devuelve (fig, exacto): exacto=False si el estado no es plano."""
    sgn, w = r["sgn"], r["internos"]
    k = np.argmin(np.abs(w))                   # esfuerzo "fuera del plano"
    a, b = np.delete(w, k)
    exacto = abs(w[k]) <= 1e-6 * r["escala"]
    sy = sigma_y
    hexa = np.array([[sy, 0], [sy, sy], [0, sy], [-sy, 0], [-sy, -sy],
                     [0, -sy], [sy, 0]])
    t = np.linspace(0, 2 * np.pi, 400)
    A, B = np.sqrt(2) * sy * np.cos(t), np.sqrt(2 / 3) * sy * np.sin(t)
    e1, e2 = (A + B) / np.sqrt(2), (A - B) / np.sqrt(2)  # σ1²-σ1σ2+σ2²=σy²
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(hexa[:, 0], hexa[:, 1], "b-", label="Tresca")
    ax.plot(e1, e2, "r-", label="Von Mises")
    ax.plot(sgn * a, sgn * b, "k*", ms=14, label="Estado actual")
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"σa [{unidad}]"); ax.set_ylabel(f"σb [{unidad}]")
    ax.set_title("Esfuerzo plano (σ3 = 0)"); ax.legend(); ax.grid(alpha=0.3)
    return fig, exacto


# --------------------------------------------------------------------------
# 4. Interfaz Streamlit
# --------------------------------------------------------------------------
def fmt(v, d=4):
    return "∞" if np.isinf(v) else f"{v:.{d}g}"


def detalle_criterios(r, es_acero, sigma_y):
    """Lista de (criterio, [(magnitud, valor)], FS) solo del material elegido.
    Acero: Tresca y von Mises. Geomaterial: Mohr-Coulomb y Drucker-Prager."""
    f = fmt
    if es_acero:
        return [
            ("Tresca", [("σ1 − σ3", f(r["dif_tresca"])), ("σy", f(sigma_y)),
                        ("FS", f(r["fs_tresca"]))], r["fs_tresca"]),
            ("Von Mises", [("σvm = √(3·J2)", f(r["s_vm"])), ("σy", f(sigma_y)),
                           ("FS", f(r["fs_vm"]))], r["fs_vm"]),
        ]
    s1c, s3c = -r["internos"][2], -r["internos"][0]   # compresión +
    mc = [("Nφ", f(r["N_phi"])),
          ("σ3 actual (compresión +)", f(s3c)),
          ("σ1 actual (compresión +)", f(s1c)),
          ("σ1 de falla (compresión +)", f(r["s1_falla_c"])),
          ("Ángulo del plano de falla θ", f"{f(r['theta'])}°"),
          ("σm", f((s1c + s3c) / 2)), ("R", f(r["R"])),
          ("FS", f(r["fs_mc"]))]
    dp = [("α", f(r["alpha"])), ("k", f(r["k"])),
          ("I1 (tensión +)", f(r["sgn"] * r["I1"])),
          ("√J2", f(np.sqrt(r["J2"]))),
          ("F = α·I1 + √J2 − k", f(r["F_dp"])),
          ("FS", f(r["fs_dp"]))]
    return [("Mohr-Coulomb", mc, r["fs_mc"]), ("Drucker-Prager", dp, r["fs_dp"])]


def main():
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Esfuerzos y criterios de falla", layout="wide")
    st.title("Calculadora de esfuerzos y criterios de falla")

    # ---- Entradas -------------------------------------------------------
    with st.sidebar:
        st.header("Entradas")
        material = st.radio("Material", ["Acero", "Geomaterial (suelo/roca)"])
        es_acero = material == "Acero"
        conv = st.radio("Convención de signos",
                        ["Tensión positiva (+)", "Compresión positiva (+)"])
        unidad = st.radio("Unidad del tensor", ["MPa", "kPa"], horizontal=True)
        modo = st.radio("Ingreso del tensor", ["6 componentes", "Matriz 3×3"])
        if modo == "6 componentes":
            sxx = st.number_input(f"σxx [{unidad}]", value=150.0)
            syy = st.number_input(f"σyy [{unidad}]", value=0.0)
            szz = st.number_input(f"σzz [{unidad}]", value=0.0)
            txy = st.number_input(f"τxy [{unidad}]", value=0.0)
            txz = st.number_input(f"τxz [{unidad}]", value=0.0)
            tyz = st.number_input(f"τyz [{unidad}]", value=0.0)
            T_user = tensor_desde_componentes(sxx, syy, szz, txy, txz, tyz)
        else:
            T_user = np.zeros((3, 3))
            for i in range(3):
                cols = st.columns(3)
                for j in range(3):
                    T_user[i, j] = cols[j].number_input(
                        f"({'xyz'[i]}{'xyz'[j]})", value=150.0 if i == j == 0 else 0.0,
                        key=f"m{i}{j}")
        st.subheader("Resistencia")
        if es_acero:
            sy_val = st.number_input("σy acero", value=250.0)
            sy_u = st.selectbox("Unidad de σy", ["MPa", "kPa"])
            c_val, c_u, phi_deg = 0.0, "kPa", 0.0   # no se usan
        else:
            c_val = st.number_input("Cohesión c (c_u si φ = 0)", value=20.0)
            c_u = st.selectbox("Unidad de c", ["kPa", "MPa"])
            phi_deg = st.number_input("Ángulo de fricción φ [°]", value=30.0)
            sy_val, sy_u = 1.0, "MPa"                # no se usa (valor ficticio)

    tension_pos = conv.startswith("Tensión")
    # Unificar unidades: σy (MPa) y c (kPa) -> unidad del tensor
    a_kpa = {"kPa": 1.0, "MPa": 1000.0}
    sigma_y = sy_val * a_kpa[sy_u] / a_kpa[unidad]
    c = c_val * a_kpa[c_u] / a_kpa[unidad]

    # ---- Validaciones ---------------------------------------------------
    errores = validar_entradas(T_user, sigma_y, c, phi_deg)
    if errores:
        for e in errores:
            st.error(e)
        st.stop()

    r = calcular(T_user, tension_pos, sigma_y, c, phi_deg)

    # ---- Advertencias ---------------------------------------------------
    if not es_acero and r["solo_traccion"]:
        st.warning("Todos los esfuerzos principales son de tracción: Mohr-Coulomb "
                   "y Drucker-Prager no son representativos (se ignora el corte de "
                   "tracción del suelo).")
    elif not es_acero and r["hay_traccion"]:
        st.warning("Hay esfuerzos principales de tracción: Mohr-Coulomb y "
                   "Drucker-Prager suponen un material que casi no resiste tensión; "
                   "interprete estos resultados con cautela.")
    if r["J2"] <= TOL * r["escala"] ** 2:
        st.info("Estado hidrostático (J2 = 0): los FS tienden a infinito.")

    # ---- Tabla de criterios (solo los del material elegido) -------------
    st.subheader(f"Resultados — Material: {material} · Convención: {conv} · "
                 f"Unidades: {unidad}")
    detalle = detalle_criterios(r, es_acero, sigma_y)

    # Resumen: FS y veredicto por criterio
    st.table(pd.DataFrame([{
        "Criterio": n, "FS": fmt(fs),
        "Veredicto": ("✅ " if veredicto(fs) == "Resiste" else "❌ ") + veredicto(fs),
    } for n, _, fs in detalle]).set_index("Criterio"))

    # Detalle: todas las magnitudes intermedias de cada criterio
    for col, (n, filas, fs) in zip(st.columns(len(detalle)), detalle):
        with col:
            st.markdown(f"**{n}**")
            st.table(pd.DataFrame(filas, columns=["Magnitud", "Valor"])
                     .set_index("Magnitud"))

    fs_min = min(fs for _, _, fs in detalle)
    crit = " y ".join(n for n, _, _ in detalle)
    st.markdown(f"**Interpretación:** FS mínimo ({crit}) = **{fmt(fs_min)}** "
                f"→ {veredicto(fs_min).lower()}.")

    # ---- Detalle numérico -----------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Esfuerzos principales y direcciones**")
        d = pd.DataFrame(r["dirs"].T, columns=["n_x", "n_y", "n_z"],
                         index=["σ1", "σ2", "σ3"])
        d.insert(0, f"valor [{unidad}]", r["princ_user"])
        st.dataframe(d.round(5))
        st.markdown("**Tensor desviador s**")
        st.dataframe(pd.DataFrame(r["S_user"]).round(5))
    with c2:
        st.markdown("**Invariantes y esfuerzos derivados**")
        st.dataframe(pd.DataFrame({
            "Magnitud": ["I1", "I2", "I3", "p (compresión +)", "J2", "J3",
                         "τ_oct", "τ_max"],
            "Valor": [r["I1"], r["I2"], r["I3"], r["p"], r["J2"], r["J3"],
                      r["tau_oct"], r["tau_max"]]}).set_index("Magnitud"))

    # ---- Gráficas -------------------------------------------------------
    if es_acero:
        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(fig_mohr(r, 0.0, 0.0, unidad))   # sin envolvente
        with g2:
            fig, exacto = fig_plano(r, sigma_y, unidad)
            st.pyplot(fig)
            if not exacto:
                st.warning("El estado no es plano (ningún principal ≈ 0): el punto se "
                           "proyecta con los dos principales mayores en magnitud y "
                           "solo es exacto para σ3 = 0.")
    else:
        st.pyplot(fig_mohr(r, c, phi_deg, unidad))     # con envolvente M-C

if __name__ == "__main__":
    main()