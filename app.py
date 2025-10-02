# app.py
# -*- coding: utf-8 -*-
"""
Simulador MiVivienda / Techo Propio – Método Francés Vencido (meses de 30 días)
Compatible con Streamlit Cloud (https://streamlit.io/cloud)

Características clave:
- Autenticación (registro/inicio de sesión) con SQLite y contraseñas hasheadas (SHA-256)
- Moneda: PEN (S/.) o USD ($)
- Tasa: Efectiva Anual (TEA) o Nominal Anual (TNA) con capitalización configurable
- Plazos de gracia: Total (capitaliza intereses) o Parcial (se paga solo interés)
- Soporte Bono (p. ej., Techo Propio) que reduce el principal desembolsado
- Cronograma francés mensual con meses de 30 días (30/360) y pagos vencidos
- Indicadores: VAN, TIR mensual y TCEA (anual efectiva), monto total pagado, interés total
- Gestión de clientes y unidades inmobiliarias (alta, edición, guardado en SQLite)
- Exportación de cronograma a CSV y guardado del caso

Nota: Este ejemplo es educativo. Revise las normas de transparencia vigentes de la SBS/MEF
antes de usar en producción. Asegure credenciales y aplique controles adicionales según su entidad.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
from datetime import datetime, timedelta

# ----------------------------- Config general -----------------------------
st.set_page_config(page_title="MiVivienda – Simulador", page_icon="🏠", layout="wide")

# ----------------------------- Utilitarios DB -----------------------------
@st.cache_resource(show_spinner=False)
def get_conn():
    conn = sqlite3.connect("mivivienda.db", check_same_thread=False)
    return conn

@st.cache_resource(show_spinner=False)
def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # Usuarios
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        )
        """
    )
    # Clientes
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            full_name TEXT,
            phone TEXT,
            email TEXT,
            income_monthly REAL,
            dependents INTEGER,
            employment_type TEXT,
            notes TEXT,
            created_by TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    # Unidades inmobiliarias
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT,
            project TEXT,
            address TEXT,
            bedrooms INTEGER,
            area_m2 REAL,
            price REAL,
            currency TEXT,
            notes TEXT,
            created_by TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    # Casos de simulación guardados
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            client_id INTEGER,
            unit_id INTEGER,
            case_name TEXT,
            params_json TEXT,
            created_at TEXT,
            FOREIGN KEY (client_id) REFERENCES clients(id),
            FOREIGN KEY (unit_id) REFERENCES units(id)
        )
        """
    )
    conn.commit()
    return conn

conn = init_db()

# ----------------------------- Seguridad -----------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, password: str) -> bool:
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
                    (username, hash_password(password), datetime.utcnow().isoformat()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def check_login(username: str, password: str) -> bool:
    cur = get_conn().cursor()
    cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    if row:
        return row[0] == hash_password(password)
    return False

# Crear usuario demo si la tabla está vacía
cur = get_conn().cursor()
cur.execute("SELECT COUNT(*) FROM users")
if cur.fetchone()[0] == 0:
    create_user("admin", "admin")

# ----------------------------- Cálculos financieros -----------------------------
def nominal_to_effective_monthly(tna: float, cap_per_year: int) -> float:
    """Convierte TNA (decimal, p. ej. 0.12) con capitalización m a tasa efectiva mensual.
    Fórmula: i_m = (1 + tna/m)^(m/12) - 1
    """
    m = max(1, int(cap_per_year))
    return (1 + tna / m) ** (m / 12.0) - 1.0


def tea_to_monthly(tea: float) -> float:
    """Convierte TEA efectiva a tasa efectiva mensual: i_m = (1+TEA)^(1/12) - 1"""
    return (1 + tea) ** (1 / 12.0) - 1.0


def french_payment(principal: float, i_m: float, n: int) -> float:
    """Cuota fija del método francés con pagos vencidos. Maneja i_m = 0."""
    if n <= 0:
        return 0.0
    if i_m == 0:
        return principal / n
    return principal * (i_m * (1 + i_m) ** n) / ((1 + i_m) ** n - 1)


def build_schedule(
    principal: float,
    i_m: float,
    n_months: int,
    grace_total: int = 0,
    grace_partial: int = 0,
    start_date: datetime | None = None,
    fee_opening: float = 0.0,
    monthly_insurance: float = 0.0,
    monthly_admin_fee: float = 0.0,
    bono_monto: float = 0.0,
    ) -> pd.DataFrame:
    """Genera cronograma francés mensual (30 días) con gracia total/parcial y cargos.
    - grace_total: meses sin pago; intereses capitalizan (se suman al saldo)
    - grace_partial: meses pagando solo interés (no amortiza)
    - fee_opening: comisión/desembolso inicial (flujo en t0)
    - monthly_insurance y monthly_admin_fee: cargos fijos mensuales
    - bono_monto: bono aplicado en t0 que reduce el principal a financiar
    Devuelve DataFrame con columnas: Periodo, Fecha, Saldo Inicial, Interés, Amortización,
    Cuota (sin cargos), Seguros, Gastos Adm, Cuota Total, Saldo Final, Flujo Cliente
    (flujo desde la perspectiva del cliente; desembolso positivo en t0, pagos negativos).
    """
    # Ajuste por bono
    principal_neto = max(0.0, principal - bono_monto)

    # Fechas con meses de 30 días: usamos incremento de 30 días fijos
    if start_date is None:
        start_date = datetime.today()

    rows = []

    # Flujo en t0 (desembolso): cliente recibe el principal bruto y paga aperturas
    # Flujo cliente en t0: +principal - fee_opening - bono? (bono no lo paga el cliente)
    flujo_t0 = principal - fee_opening

    rows.append({
        "Periodo": 0,
        "Fecha": start_date.strftime("%Y-%m-%d"),
        "Saldo Inicial": 0.0,
        "Interés": 0.0,
        "Amortización": 0.0,
        "Cuota": 0.0,
        "Seguro": 0.0,
        "Gasto Adm": 0.0,
        "Cuota Total": 0.0,
        "Saldo Final": principal_neto,
        "Flujo Cliente": flujo_t0
    })

    saldo = principal_neto

    # Determinar cuota base (sin cargos) para la etapa de amortización real
    # Durante gracia total y parcial, el cálculo de cuota cambia
    # Construimos iterativamente
    date_i = start_date

    # Meses totales = gracia_total + gracia_parcial + n_months
    total_months = grace_total + grace_partial + n_months

    # Cuota fija aplica SOLO desde el mes (gracia_total + grace_partial + 1)
    # por n_months periodos.
    cuota_fija = french_payment(saldo, i_m, n_months) if n_months > 0 else 0.0

    for t in range(1, total_months + 1):
        date_i = date_i + timedelta(days=30)  # meses de 30 días
        interes = saldo * i_m

        if t <= grace_total:
            # Gracia total: no se paga; interés capitaliza
            amort = 0.0
            cuota = 0.0
            saldo_final = saldo + interes
            pago_cliente = -(monthly_insurance + monthly_admin_fee)  # paga solo cargos si aplican
        elif t <= grace_total + grace_partial:
            # Gracia parcial: paga solo interés (sin amortización)
            amort = 0.0
            cuota = interes
            saldo_final = saldo
            pago_cliente = -(cuota + monthly_insurance + monthly_admin_fee)
        else:
            # Etapa de amortización francesa
            cuota = cuota_fija
            amort = cuota - interes
            # Evitar residuales por redondeo en el último periodo
            if t == total_months:
                amort = saldo  # última amortización liquida saldo
                cuota = interes + amort
            saldo_final = saldo - amort
            pago_cliente = -(cuota + monthly_insurance + monthly_admin_fee)

        rows.append({
            "Periodo": t,
            "Fecha": date_i.strftime("%Y-%m-%d"),
            "Saldo Inicial": saldo,
            "Interés": interes,
            "Amortización": amort,
            "Cuota": cuota,
            "Seguro": monthly_insurance,
            "Gasto Adm": monthly_admin_fee,
            "Cuota Total": cuota + monthly_insurance + monthly_admin_fee,
            "Saldo Final": saldo_final,
            "Flujo Cliente": pago_cliente,
        })

        saldo = saldo_final

    df = pd.DataFrame(rows)
    return df


def npv(rate: float, cashflows: np.ndarray) -> float:
    """VAN con tasa por periodo (mensual). cashflows[0] es t0."""
    return float(np.sum(cashflows / (1 + rate) ** np.arange(len(cashflows))))


def irr(cashflows: np.ndarray, guess: float = 0.01, max_iter: int = 100, tol: float = 1e-7) -> float:
    """TIR (por periodo) mediante Newton-Raphson. Devuelve np.nan si no converge."""
    r = guess
    for _ in range(max_iter):
        # VAN y derivada
        times = np.arange(len(cashflows))
        denom = (1 + r) ** times
        f = np.sum(cashflows / denom)
        df = -np.sum(times * cashflows / ((1 + r) ** (times + 1)))
        if abs(df) < 1e-12:
            break
        r_new = r - f / df
        if abs(r_new - r) < tol:
            return float(r_new)
        r = r_new
    return np.nan

# ----------------------------- UI: Autenticación -----------------------------
if "auth" not in st.session_state:
    st.session_state.auth = {"logged": False, "user": None}

with st.sidebar:
    st.header("🔐 Acceso")
    if not st.session_state.auth["logged"]:
        login_tab, signup_tab = st.tabs(["Iniciar sesión", "Registrarse"])
        with login_tab:
            u = st.text_input("Usuario", key="login_user")
            p = st.text_input("Contraseña", type="password", key="login_pass")
            if st.button("Entrar", use_container_width=True):
                if check_login(u, p):
                    st.session_state.auth = {"logged": True, "user": u}
                    st.success(f"Bienvenido, {u}!")
                    st.experimental_rerun()
                else:
                    st.error("Credenciales inválidas")
        with signup_tab:
            u2 = st.text_input("Nuevo usuario", key="signup_user")
            p2 = st.text_input("Nueva contraseña", type="password", key="signup_pass")
            if st.button("Crear cuenta", use_container_width=True):
                if u2 and p2:
                    ok = create_user(u2, p2)
                    if ok:
                        st.success("Usuario creado. Ahora inicie sesión.")
                    else:
                        st.error("Usuario ya existe.")
                else:
                    st.warning("Complete usuario y contraseña")
    else:
        st.write(f"👤 Sesión: {st.session_state.auth['user']}")
        if st.button("Cerrar sesión", use_container_width=True):
            st.session_state.auth = {"logged": False, "user": None}
            st.experimental_rerun()

if not st.session_state.auth["logged"]:
    st.info("Inicie sesión para usar el simulador.")
    st.stop()

# ----------------------------- UI principal -----------------------------
st.title("🏠 MiVivienda / Techo Propio – Simulador método francés (30/360)")
st.caption("Empresa inmobiliaria – cálculo de cronograma, VAN/TIR/TCEA y gestión de clientes & unidades")

# Secciones
sec1, sec2, sec3 = st.tabs(["1) Cliente y Unidad", "2) Configurar Préstamo", "3) Resultados & Guardado"])

# ----------------------------- 1) Cliente y Unidad -----------------------------
with sec1:
    st.subheader("Datos del cliente")
    colc1, colc2, colc3 = st.columns(3)
    with colc1:
        doc_id = st.text_input("Documento (DNI/RUC/etc.)")
        full_name = st.text_input("Nombre completo")
        income_monthly = st.number_input("Ingreso mensual", min_value=0.0, step=100.0)
        dependents = st.number_input("Dependientes", min_value=0, step=1)
    with colc2:
        phone = st.text_input("Teléfono")
        email = st.text_input("Email")
        employment_type = st.selectbox("Tipo de empleo", ["Dependiente", "Independiente", "Mixto", "Otro"]) 
    with colc3:
        notes_client = st.text_area("Notas socioeconómicas")
        btn_save_client = st.button("💾 Guardar/Actualizar cliente")

    if btn_save_client:
        now = datetime.utcnow().isoformat()
        cur = get_conn().cursor()
        # Buscar si existe por doc_id
        cur.execute("SELECT id FROM clients WHERE doc_id=?", (doc_id,))
        row = cur.fetchone()
        if row:
            cur.execute(
                """
                UPDATE clients SET full_name=?, phone=?, email=?, income_monthly=?, dependents=?,
                employment_type=?, notes=?, updated_at=? WHERE id=?
                """,
                (full_name, phone, email, income_monthly, dependents, employment_type, notes_client, now, row[0])
            )
            get_conn().commit()
            st.success("Cliente actualizado")
        else:
            cur.execute(
                """
                INSERT INTO clients (doc_id, full_name, phone, email, income_monthly, dependents,
                employment_type, notes, created_by, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (doc_id, full_name, phone, email, income_monthly, dependents, employment_type,
                 notes_client, st.session_state.auth['user'], now, now)
            )
            get_conn().commit()
            st.success("Cliente creado")

    st.markdown("---")
    st.subheader("Datos de la unidad inmobiliaria")
    colu1, colu2, colu3 = st.columns(3)
    with colu1:
        code = st.text_input("Código interno de unidad")
        project = st.text_input("Proyecto")
        address = st.text_input("Dirección")
        bedrooms = st.number_input("Dormitorios", min_value=0, step=1)
    with colu2:
        area_m2 = st.number_input("Área (m²)", min_value=0.0, step=1.0)
        currency_unit = st.selectbox("Moneda precio", ["PEN", "USD"], index=0)
        price = st.number_input("Precio de venta", min_value=0.0, step=1000.0)
    with colu3:
        notes_unit = st.text_area("Notas de la unidad")
        btn_save_unit = st.button("💾 Guardar/Actualizar unidad")

    if btn_save_unit:
        now = datetime.utcnow().isoformat()
        cur = get_conn().cursor()
        cur.execute("SELECT id FROM units WHERE code=?", (code,))
        row = cur.fetchone()
        if row:
            cur.execute(
                """
                UPDATE units SET project=?, address=?, bedrooms=?, area_m2=?, price=?, currency=?,
                notes=?, updated_at=? WHERE id=?
                """,
                (project, address, bedrooms, area_m2, price, currency_unit, notes_unit, now, row[0])
            )
            get_conn().commit()
            st.success("Unidad actualizada")
        else:
            cur.execute(
                """
                INSERT INTO units (code, project, address, bedrooms, area_m2, price, currency, notes,
                created_by, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (code, project, address, bedrooms, area_m2, price, currency_unit, notes_unit,
                 st.session_state.auth['user'], now, now)
            )
            get_conn().commit()
            st.success("Unidad creada")

# ----------------------------- 2) Configurar Préstamo -----------------------------
with sec2:
    st.subheader("Configuración del préstamo")
    col1, col2, col3 = st.columns(3)
    with col1:
        currency = st.selectbox("Moneda del préstamo", ["PEN", "USD"], index=0)
        principal = st.number_input("Monto a financiar (principal bruto)", min_value=0.0, step=1000.0)
        bono = st.number_input("Bono (Techo Propio u otro)", min_value=0.0, step=500.0, help="Se aplica en t0, reduce el principal neto")
        term_months = st.number_input("Plazo (meses)", min_value=1, step=1)
    with col2:
        tasa_tipo = st.selectbox("Tipo de tasa", ["Efectiva (TEA)", "Nominal (TNA)"])
        tasa_anual = st.number_input("Tasa anual (%)", min_value=0.0, step=0.1, help="Ingrese TEA o TNA según selección") / 100.0
        cap_m = 12
        if tasa_tipo == "Nominal (TNA)":
            cap_m = st.number_input("Capitalización por año (m)", min_value=1, step=1, value=12)
        grace_total = st.number_input("Gracia total (meses)", min_value=0, step=1)
    with col3:
        grace_partial = st.number_input("Gracia parcial (meses)", min_value=0, step=1)
        fee_opening = st.number_input("Comisión de apertura (t0)", min_value=0.0, step=100.0)
        monthly_insurance = st.number_input("Seguro mensual", min_value=0.0, step=10.0)
        monthly_admin_fee = st.number_input("Gasto admin mensual", min_value=0.0, step=10.0)

    st.caption("Mes de 30 días: la fecha de cada periodo se incrementa en 30 días fijos (convención 30/360). Pagos vencidos.")

    # Tasa mensual efectiva
    if tasa_tipo == "Efectiva (TEA)":
        i_m = tea_to_monthly(tasa_anual)
    else:
        i_m = nominal_to_effective_monthly(tasa_anual, cap_m)

    st.info(f"Tasa efectiva mensual calculada: {i_m*100:.5f}%")

    if grace_total + grace_partial >= term_months:
        st.warning("La suma de meses de gracia no puede ser mayor o igual al plazo total. Ajuste los valores.")

    compute = st.button("📅 Generar cronograma")

    if compute:
        df = build_schedule(
            principal=principal,
            i_m=i_m,
            n_months=int(term_months - (grace_total + grace_partial)),
            grace_total=int(grace_total),
            grace_partial=int(grace_partial),
            start_date=datetime.today(),
            fee_opening=fee_opening,
            monthly_insurance=monthly_insurance,
            monthly_admin_fee=monthly_admin_fee,
            bono_monto=bono,
        )
        st.session_state["schedule_df"] = df
        st.session_state["schedule_cfg"] = {
            "currency": currency,
            "principal": principal,
            "bono": bono,
            "term_months": term_months,
            "tasa_tipo": tasa_tipo,
            "tasa_anual": tasa_anual,
            "cap_m": cap_m,
            "grace_total": grace_total,
            "grace_partial": grace_partial,
            "fee_opening": fee_opening,
            "monthly_insurance": monthly_insurance,
            "monthly_admin_fee": monthly_admin_fee,
            "i_m": i_m,
        }

        st.success("Cronograma generado. Revise la pestaña 3)")

# ----------------------------- 3) Resultados & Guardado -----------------------------
with sec3:
    st.subheader("Resultados")
    if "schedule_df" not in st.session_state:
        st.info("Primero configure y genere un cronograma en la pestaña 2.")
    else:
        df = st.session_state["schedule_df"].copy()
        cfg = st.session_state["schedule_cfg"]
        symbol = "S/." if cfg["currency"] == "PEN" else "$"

        # Indicadores
        cashflows = df["Flujo Cliente"].to_numpy()
        irr_m = irr(cashflows)
        tcea = (1 + irr_m) ** 12 - 1 if np.isfinite(irr_m) else np.nan

        st.metric("TIR mensual (TIRM)", f"{irr_m*100:.3f}%" if np.isfinite(irr_m) else "No converge")
        st.metric("TCEA (anual efectiva)", f"{tcea*100:.3f}%" if np.isfinite(tcea) else "-")

        colnpv1, colnpv2 = st.columns(2)
        with colnpv1:
            disc_rate_annual = st.number_input("Tasa de descuento anual para VAN (%)", min_value=0.0, step=0.1, value=cfg["tasa_anual"]*100) / 100.0
        with colnpv2:
            disc_m = tea_to_monthly(disc_rate_annual)
            st.write(f"Tasa de descuento mensual: {disc_m*100:.4f}%")
        van = npv(disc_m, cashflows)
        st.metric("VAN (moneda del préstamo)", f"{symbol} {van:,.2f}")

        total_pagado = -cashflows[1:].sum()
        interes_total = float(df["Interés"].sum())
        st.write( f"**Total pagado por el cliente (cuotas + cargos):** {symbol} {total_pagado:,.2f}  ")
        st.write( f"**Interés total (sin cargos):** {symbol} {interes_total:,.2f}")

        st.markdown("### Cronograma de pagos")
        st.dataframe(
            df.style.format({
                "Saldo Inicial": "{:,.2f}",
                "Interés": "{:,.2f}",
                "Amortización": "{:,.2f}",
                "Cuota": "{:,.2f}",
                "Seguro": "{:,.2f}",
                "Gasto Adm": "{:,.2f}",
                "Cuota Total": "{:,.2f}",
                "Saldo Final": "{:,.2f}",
                "Flujo Cliente": "{:,.2f}",
            })
            , use_container_width=True
        )

        # Descargar CSV
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Descargar cronograma CSV", csv, file_name="cronograma_mivivienda.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Guardar caso")
        # Seleccionar cliente y unidad ya creados
        cur = get_conn().cursor()
        cur.execute("SELECT id, doc_id, full_name FROM clients ORDER BY updated_at DESC NULLS LAST")
        clients_list = cur.fetchall()
        cur.execute("SELECT id, code, project FROM units ORDER BY updated_at DESC NULLS LAST")
        units_list = cur.fetchall()

        client_opt = st.selectbox("Cliente", options=[(None, "- Seleccione -")] + [(c[0], f"{c[1]} – {c[2]}") for c in clients_list], format_func=lambda x: x[1] if isinstance(x, tuple) else x)
        unit_opt = st.selectbox("Unidad", options=[(None, "- Seleccione -")] + [(u[0], f"{u[1]} – {u[2]}") for u in units_list], format_func=lambda x: x[1] if isinstance(x, tuple) else x)
        case_name = st.text_input("Nombre del caso (ej. ClienteX – Dpto 302 – Set/2025)")
        if st.button("💾 Guardar caso en base de datos"):
            if isinstance(client_opt, tuple):
                client_id = client_opt[0]
            else:
                client_id = None
            if isinstance(unit_opt, tuple):
                unit_id = unit_opt[0]
            else:
                unit_id = None
            if not client_id or not unit_id or not case_name:
                st.error("Seleccione cliente, unidad y asigne un nombre al caso.")
            else:
                params = {
                    **cfg,
                    "generated_at": datetime.utcnow().isoformat(),
                }
                cur = get_conn().cursor()
                cur.execute(
                    "INSERT INTO cases (user, client_id, unit_id, case_name, params_json, created_at) VALUES (?,?,?,?,?,?)",
                    (
                        st.session_state.auth['user'],
                        client_id,
                        unit_id,
                        case_name,
                        pd.Series(params).to_json(),
                        datetime.utcnow().isoformat(),
                    ),
                )
                get_conn().commit()
                st.success("Caso guardado")

        st.markdown("---")
        st.subheader("Cargar caso previo")
        cur = get_conn().cursor()
        cur.execute("SELECT id, case_name, created_at FROM cases ORDER BY id DESC")
        cases = cur.fetchall()
        opt_cases = [(c[0], f"#{c[0]} – {c[1]} – {c[2]}") for c in cases]
        case_sel = st.selectbox("Casos", options=[(None, "- Seleccione -")] + opt_cases, format_func=lambda x: x[1] if isinstance(x, tuple) else x)
        if st.button("📂 Cargar parámetros del caso"):
            if isinstance(case_sel, tuple) and case_sel[0]:
                cur.execute("SELECT params_json FROM cases WHERE id=?", (case_sel[0],))
                row = cur.fetchone()
                if row:
                    params = pd.read_json(row[0], typ='series')
                    st.session_state["schedule_cfg"] = dict(params)
                    df2 = build_schedule(
                        principal=params["principal"],
                        i_m=params["i_m"],
                        n_months=int(params["term_months"] - (params["grace_total"] + params["grace_partial"])) ,
                        grace_total=int(params["grace_total"]),
                        grace_partial=int(params["grace_partial"]),
                        start_date=datetime.today(),
                        fee_opening=params["fee_opening"],
                        monthly_insurance=params["monthly_insurance"],
                        monthly_admin_fee=params["monthly_admin_fee"],
                        bono_monto=params["bono"],
                    )
                    st.session_state["schedule_df"] = df2
                    st.success("Parámetros cargados y cronograma regenerado")
                else:
                    st.error("No se pudo leer el caso seleccionado")
            else:
                st.warning("Seleccione un caso válido")

st.markdown("""
---
**Transparencia – referencias técnicas**  
- Método francés: cuota fija con interés sobre saldo y amortización creciente.  
- Convención **30/360**: se asume cada mes con 30 días (adecuado para “francés vencido ordinario – meses de 30 días”).  
- **TCEA** aproximada como (1+TIRM)^12 - 1 bajo periodicidad mensual.  
- Gracia **total** (capitalización de intereses) y **parcial** (pago de interés) implementadas conforme práctica usual del mercado.  
- Ajuste final en el último periodo corrige residuales de redondeo.
""")
