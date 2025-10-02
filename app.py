# app.py
# -*- coding: utf-8 -*-
"""
Simulador MiVivienda / Techo Propio – Método Francés Vencido (meses de 30 días)
Compatible con Streamlit Community Cloud (https://streamlit.io/cloud)

Cambios implementados según feedback:
- Corregida carga de "caso previo" (selector muestra y carga el caso correcto por ID).
- En "Unidad inmobiliaria" ahora solo se gestionan **código** y **nombre** (proyecto).
- Ocultados mensajes/errores molestos al **iniciar/cerrar sesión** (flujo silencioso con `st.rerun`).
- Validaciones estrictas en **Clientes** (todos los campos obligatorios, teléfono 9 dígitos PE, email formato `xyz@xyz.com`).
- Caja para **seleccionar y editar** clientes ya registrados.
- En la carga de casos se muestra **#caso – nombre del cliente – vivienda (código/nombre)**.

Nota: Ejemplo educativo. Revise normas SBS y MiVivienda antes de producción.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import re
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
            doc_id TEXT UNIQUE,
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
    # Unidades inmobiliarias (solo código y nombre)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE,
            project TEXT,
            created_by TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    # Casos guardados
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
        cur = get_conn().cursor()
        cur.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
                    (username, hash_password(password), datetime.utcnow().isoformat()))
        get_conn().commit()
        return True
    except sqlite3.IntegrityError:
        return False


def check_login(username: str, password: str) -> bool:
    cur = get_conn().cursor()
    cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    return bool(row and row[0] == hash_password(password))

# Usuario demo si está vacío
cur = get_conn().cursor()
cur.execute("SELECT COUNT(*) FROM users")
if cur.fetchone()[0] == 0:
    create_user("admin", "admin")

# ----------------------------- Cálculos financieros -----------------------------
def nominal_to_effective_monthly(tna: float, cap_per_year: int) -> float:
    m = max(1, int(cap_per_year))
    return (1 + tna / m) ** (m / 12.0) - 1.0


def tea_to_monthly(tea: float) -> float:
    return (1 + tea) ** (1 / 12.0) - 1.0


def french_payment(principal: float, i_m: float, n: int) -> float:
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
    principal_neto = max(0.0, principal - bono_monto)
    if start_date is None:
        start_date = datetime.today()

    rows = []
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
    date_i = start_date
    total_months = grace_total + grace_partial + n_months
    cuota_fija = french_payment(saldo, i_m, n_months) if n_months > 0 else 0.0

    for t in range(1, total_months + 1):
        date_i = date_i + timedelta(days=30)  # 30/360
        interes = saldo * i_m

        if t <= grace_total:  # Gracia total: capitaliza interés
            amort = 0.0
            cuota = 0.0
            saldo_final = saldo + interes
            pago_cliente = -(monthly_insurance + monthly_admin_fee)
        elif t <= grace_total + grace_partial:  # Gracia parcial: paga solo interés
            amort = 0.0
            cuota = interes
            saldo_final = saldo
            pago_cliente = -(cuota + monthly_insurance + monthly_admin_fee)
        else:  # Amortización francesa
            cuota = cuota_fija
            amort = cuota - interes
            if t == total_months:  # limpiar residuo final
                amort = saldo
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

    return pd.DataFrame(rows)


def npv(rate: float, cashflows: np.ndarray) -> float:
    return float(np.sum(cashflows / (1 + rate) ** np.arange(len(cashflows))))


def irr(cashflows: np.ndarray, guess: float = 0.01, max_iter: int = 100, tol: float = 1e-7) -> float:
    r = guess
    for _ in range(max_iter):
        t = np.arange(len(cashflows))
        denom = (1 + r) ** t
        f = np.sum(cashflows / denom)
        df = -np.sum(t * cashflows / ((1 + r) ** (t + 1)))
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
            u = st.text_input("Usuario")
            p = st.text_input("Contraseña", type="password")
            if st.button("Entrar", use_container_width=True):
                if check_login(u, p):
                    st.session_state.auth = {"logged": True, "user": u}
                    st.rerun()  # sin mensajes
                else:
                    st.error("Credenciales inválidas")
        with signup_tab:
            u2 = st.text_input("Nuevo usuario")
            p2 = st.text_input("Nueva contraseña", type="password")
            if st.button("Crear cuenta", use_container_width=True):
                if u2 and p2:
                    ok = create_user(u2, p2)
                    if ok:
                        st.success("Usuario creado. Inicie sesión.")
                    else:
                        st.error("El usuario ya existe.")
                else:
                    st.warning("Complete usuario y contraseña")
    else:
        st.write(f"👤 {st.session_state.auth['user']}")
        if st.button("Cerrar sesión", use_container_width=True):
            st.session_state.clear()
            st.rerun()  # sin mensajes

if not st.session_state.auth.get("logged"):
    st.stop()

# ----------------------------- UI principal -----------------------------
st.title("🏠 MiVivienda / Techo Propio – Simulador método francés (30/360)")
st.caption("Empresa inmobiliaria – cálculo de cronograma, VAN/TIR/TCEA y gestión de clientes & unidades")

sec1, sec2, sec3 = st.tabs(["1) Cliente y Unidad", "2) Configurar Préstamo", "3) Resultados & Guardado"])

# ----------------------------- 1) Cliente y Unidad -----------------------------
with sec1:
    st.subheader("Datos del cliente")

    # Selector de cliente existente para edición
    cur = get_conn().cursor()
    cur.execute("SELECT id, doc_id, full_name FROM clients ORDER BY full_name ASC")
    clients_list = cur.fetchall()
    client_labels = ["➕ Nuevo cliente"] + [f"{c[1]} – {c[2]} (ID {c[0]})" for c in clients_list]
    client_choice = st.selectbox("Editar cliente", client_labels, index=0)

    # Inicialización de campos
    doc_id = ""
    full_name = ""
    income_monthly = 0.0
    dependents = 0
    phone = ""
    email = ""
    employment_type = "Dependiente"
    notes_client = ""
    editing_client_id = None

    if client_choice != "➕ Nuevo cliente":
        # Cargar datos del cliente elegido
        idx = client_labels.index(client_choice) - 1
        c = clients_list[idx]
        editing_client_id = c[0]
        cur.execute("SELECT doc_id, full_name, phone, email, income_monthly, dependents, employment_type, notes FROM clients WHERE id=?", (editing_client_id,))
        row = cur.fetchone()
        if row:
            doc_id, full_name, phone, email, income_monthly, dependents, employment_type, notes_client = row

    colc1, colc2, colc3 = st.columns(3)
    with colc1:
        doc_id = st.text_input("Documento (OBLIGATORIO)", value=doc_id)
        full_name = st.text_input("Nombre completo (OBLIGATORIO)", value=full_name)
        income_monthly = st.number_input("Ingreso mensual (OBLIGATORIO)", min_value=0.0, step=100.0, value=float(income_monthly))
        dependents = st.number_input("Dependientes (OBLIGATORIO)", min_value=0, step=1, value=int(dependents))
    with colc2:
        phone = st.text_input("Teléfono 9 dígitos (OBLIGATORIO)", value=phone, help="Ej.: 912345678")
        email = st.text_input("Email (OBLIGATORIO)", value=email, help="Formato: xyz@xyz.com")
        employment_type = st.selectbox("Tipo de empleo (OBLIGATORIO)", ["Dependiente", "Independiente", "Mixto", "Otro"], index=["Dependiente","Independiente","Mixto","Otro"].index(employment_type) if employment_type in ["Dependiente","Independiente","Mixto","Otro"] else 0)
    with colc3:
        notes_client = st.text_area("Notas socioeconómicas", value=notes_client)
        btn_save_client = st.button("💾 Guardar cliente")

    def valid_email(s: str) -> bool:
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s or ""))

    def valid_phone_pe(s: str) -> bool:
        return bool(re.fullmatch(r"\d{9}", (s or "").strip()))

    if btn_save_client:
        # Validaciones obligatorias
        missing = []
        if not doc_id: missing.append("Documento")
        if not full_name: missing.append("Nombre completo")
        if income_monthly <= 0: missing.append("Ingreso mensual > 0")
        if dependents < 0: missing.append("Dependientes")
        if not valid_phone_pe(phone): missing.append("Teléfono 9 dígitos")
        if not valid_email(email): missing.append("Email válido")
        if not employment_type: missing.append("Tipo de empleo")

        if missing:
            st.error("Complete correctamente: " + ", ".join(missing))
        else:
            now = datetime.utcnow().isoformat()
            cur = get_conn().cursor()
            if editing_client_id:  # actualizar
                cur.execute(
                    """
                    UPDATE clients SET doc_id=?, full_name=?, phone=?, email=?, income_monthly=?, dependents=?,
                    employment_type=?, notes=?, updated_at=? WHERE id=?
                    """,
                    (doc_id, full_name, phone, email, income_monthly, dependents, employment_type, notes_client, now, editing_client_id)
                )
                get_conn().commit()
                st.success("Cliente actualizado")
            else:  # crear
                try:
                    cur.execute(
                        """
                        INSERT INTO clients (doc_id, full_name, phone, email, income_monthly, dependents,
                        employment_type, notes, created_by, created_at, updated_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (doc_id, full_name, phone, email, income_monthly, dependents, employment_type, notes_client,
                         st.session_state.auth['user'], now, now)
                    )
                    get_conn().commit()
                    st.success("Cliente creado")
                except sqlite3.IntegrityError:
                    st.error("Ya existe un cliente con ese Documento")

    st.markdown("---")
    st.subheader("Unidad inmobiliaria (solo Código y Nombre)")

    cur.execute("SELECT id, code, project FROM units ORDER BY project ASC")
    units_list = cur.fetchall()
    unit_labels = ["➕ Nueva unidad"] + [f"{u[1]} – {u[2]} (ID {u[0]})" for u in units_list]
    unit_choice = st.selectbox("Editar unidad", unit_labels, index=0)

    code = ""
    project = ""
    editing_unit_id = None

    if unit_choice != "➕ Nueva unidad":
        idx = unit_labels.index(unit_choice) - 1
        u = units_list[idx]
        editing_unit_id = u[0]
        code = u[1] or ""
        project = u[2] or ""

    colu1, colu2 = st.columns(2)
    with colu1:
        code = st.text_input("Código (OBLIGATORIO)", value=code)
    with colu2:
        project = st.text_input("Nombre / Proyecto (OBLIGATORIO)", value=project)
    btn_save_unit = st.button("💾 Guardar unidad")

    if btn_save_unit:
        if not code or not project:
            st.error("Complete Código y Nombre/Proyecto")
        else:
            now = datetime.utcnow().isoformat()
            cur = get_conn().cursor()
            if editing_unit_id:
                try:
                    cur.execute("UPDATE units SET code=?, project=?, updated_at=? WHERE id=?", (code, project, now, editing_unit_id))
                    get_conn().commit()
                    st.success("Unidad actualizada")
                except sqlite3.IntegrityError:
                    st.error("Ya existe otra unidad con ese Código")
            else:
                try:
                    cur.execute("INSERT INTO units (code, project, created_by, created_at, updated_at) VALUES (?,?,?,?,?)",
                                (code, project, st.session_state.auth['user'], now, now))
                    get_conn().commit()
                    st.success("Unidad creada")
                except sqlite3.IntegrityError:
                    st.error("Ya existe una unidad con ese Código")

# ----------------------------- 2) Configurar Préstamo -----------------------------
with sec2:
    st.subheader("Configuración del préstamo")
    col1, col2, col3 = st.columns(3)
    with col1:
        currency = st.selectbox("Moneda", ["PEN", "USD"], index=0)
        principal = st.number_input("Monto a financiar (principal bruto)", min_value=0.0, step=1000.0)
        bono = st.number_input("Bono (Techo Propio u otro)", min_value=0.0, step=500.0)
        term_months = st.number_input("Plazo (meses)", min_value=1, step=1)
    with col2:
        tasa_tipo = st.selectbox("Tipo de tasa", ["Efectiva (TEA)", "Nominal (TNA)"])
        tasa_anual = st.number_input("Tasa anual (%)", min_value=0.0, step=0.1) / 100.0
        cap_m = 12
        if tasa_tipo == "Nominal (TNA)":
            cap_m = st.number_input("Capitalización por año (m)", min_value=1, step=1, value=12)
        grace_total = st.number_input("Gracia total (meses)", min_value=0, step=1)
    with col3:
        grace_partial = st.number_input("Gracia parcial (meses)", min_value=0, step=1)
        fee_opening = st.number_input("Comisión de apertura (t0)", min_value=0.0, step=100.0)
        monthly_insurance = st.number_input("Seguro mensual", min_value=0.0, step=10.0)
        monthly_admin_fee = st.number_input("Gasto admin mensual", min_value=0.0, step=10.0)

    if tasa_tipo == "Efectiva (TEA)":
        i_m = tea_to_monthly(tasa_anual)
    else:
        i_m = nominal_to_effective_monthly(tasa_anual, cap_m)

    st.caption(f"Tasa efectiva mensual: {i_m*100:.5f}% | Convención 30/360 | Pagos vencidos")

    if grace_total + grace_partial >= term_months:
        st.warning("La suma de gracia total y parcial no puede ser ≥ al plazo total.")

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

        cashflows = df["Flujo Cliente"].to_numpy()
        irr_m = irr(cashflows)
        tcea = (1 + irr_m) ** 12 - 1 if np.isfinite(irr_m) else np.nan

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("TIR mensual (TIRM)", f"{irr_m*100:.3f}%" if np.isfinite(irr_m) else "No converge")
        with c2:
            st.metric("TCEA (anual efectiva)", f"{tcea*100:.3f}%" if np.isfinite(tcea) else "-")
        with c3:
            total_pagado = -cashflows[1:].sum()
            st.metric("Total pagado", f"{symbol} {total_pagado:,.2f}")

        colnpv1, colnpv2 = st.columns(2)
        with colnpv1:
            disc_rate_annual = st.number_input("Tasa de descuento anual para VAN (%)", min_value=0.0, step=0.1, value=cfg["tasa_anual"]*100) / 100.0
        with colnpv2:
            disc_m = tea_to_monthly(disc_rate_annual)
            st.write(f"Tasa de descuento mensual: {disc_m*100:.4f}%")
        van = npv(disc_m, cashflows)
        st.metric("VAN", f"{symbol} {van:,.2f}")

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
            }), use_container_width=True
        )

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Descargar cronograma CSV", csv, file_name="cronograma_mivivienda.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Guardar caso")
        # Listas de clientes y unidades para guardar
        cur = get_conn().cursor()
        cur.execute("SELECT id, full_name FROM clients ORDER BY full_name ASC")
        clients_for_save = cur.fetchall()
        cur.execute("SELECT id, code, project FROM units ORDER BY project ASC")
        units_for_save = cur.fetchall()

        sel_client_label_to_id = {f"{c[1]} (ID {c[0]})": c[0] for c in clients_for_save}
        sel_unit_label_to_id = {f"{u[1]} / {u[2]} (ID {u[0]})": u[0] for u in units_for_save}

        client_label = st.selectbox("Cliente", options=["- Seleccione -"] + list(sel_client_label_to_id.keys()))
        unit_label = st.selectbox("Unidad", options=["- Seleccione -"] + list(sel_unit_label_to_id.keys()))
        case_name = st.text_input("Nombre del caso", value="")

        if st.button("💾 Guardar caso en base de datos"):
            if client_label == "- Seleccione -" or unit_label == "- Seleccione -" or not case_name:
                st.error("Seleccione cliente, unidad y defina un nombre para el caso")
            else:
                client_id = sel_client_label_to_id[client_label]
                unit_id = sel_unit_label_to_id[unit_label]
                params = {**cfg, "generated_at": datetime.utcnow().isoformat()}
                cur.execute("INSERT INTO cases (user, client_id, unit_id, case_name, params_json, created_at) VALUES (?,?,?,?,?,?)",
                            (st.session_state.auth['user'], client_id, unit_id, case_name, pd.Series(params).to_json(), datetime.utcnow().isoformat()))
                get_conn().commit()
                st.success("Caso guardado")

        st.markdown("---")
        st.subheader("Cargar caso previo")
        # Mostrar etiqueta: #ID – Cliente – Vivienda
        cur.execute(
            """
            SELECT cases.id, cases.case_name, clients.full_name, units.code, units.project
            FROM cases
            LEFT JOIN clients ON clients.id = cases.client_id
            LEFT JOIN units ON units.id = cases.unit_id
            ORDER BY cases.id DESC
            """
        )
        rows = cur.fetchall()
        label_to_caseid = {f"#{r[0]} – {r[2] or 'Cliente?'} – {r[3] or 'CODE?'} / {r[4] or 'PROYECTO?'} – {r[1]}": r[0] for r in rows}
        case_label = st.selectbox("Casos", options=["- Seleccione -"] + list(label_to_caseid.keys()))

        if st.button("📂 Cargar parámetros del caso"):
            if case_label == "- Seleccione -":
                st.warning("Seleccione un caso válido")
            else:
                case_id = label_to_caseid[case_label]
                cur.execute("SELECT params_json FROM cases WHERE id=?", (case_id,))
                row = cur.fetchone()
                if row and row[0]:
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
                    st.success(f"Caso #{case_id} cargado")
                else:
                    st.error("No se pudo leer el caso seleccionado")

st.markdown("""
---
**Transparencia – referencias técnicas**  
- Método francés: cuota fija (interés sobre saldo).  
- Convención **30/360** (meses de 30 días).  
- **TCEA** ≈ (1+TIRM)^12 - 1.  
- Gracia total capitaliza interés; gracia parcial paga interés únicamente.  
""")
