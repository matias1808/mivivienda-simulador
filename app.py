# app.py
# -*- coding: utf-8 -*-
import os
import re
import json
import hashlib
import sqlite3
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------
st.set_page_config(page_title="MiVivienda – Simulador", page_icon="🏠", layout="wide")

# --- Tipo de cambio fijo para normalizar ingresos vs moneda del caso ---
EXCHANGE_RATE = 3.75  # 1 USD = 3.75 PEN

# ---------------------------------------------------------------------
# Base de datos (en /tmp para que sea escribible en la nube)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_conn():
    db_path = None
    try:
        db_path = st.secrets.get("DB_PATH", None)
    except Exception:
        db_path = None
    db_path = os.environ.get("DB_PATH", db_path)
    if not db_path:
        db_path = os.path.join(tempfile.gettempdir(), "mivivienda.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
    except Exception:
        pass
    return conn

@st.cache_resource(show_spinner=False)
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clients(
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
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS units(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE,
            project TEXT,
            created_by TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cases(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            client_id INTEGER,
            unit_id INTEGER,
            case_name TEXT,
            params_json TEXT,
            created_at TEXT,
            FOREIGN KEY(client_id) REFERENCES clients(id),
            FOREIGN KEY(unit_id) REFERENCES units(id)
        )
    """)

    # Índices y migraciones
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_doc_id ON clients(doc_id)")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_units_code ON units(code)")

    # Asegurar columna project en units
    cur.execute("PRAGMA table_info(units)")
    ucols = [r[1] for r in cur.fetchall()]
    if "project" not in ucols:
        cur.execute("ALTER TABLE units ADD COLUMN project TEXT")

    # Asegurar columna income_currency en clients
    cur.execute("PRAGMA table_info(clients)")
    ccols = [r[1] for r in cur.fetchall()]
    if "income_currency" not in ccols:
        cur.execute("ALTER TABLE clients ADD COLUMN income_currency TEXT DEFAULT 'PEN'")

    conn.commit()
    return conn

conn = init_db()

# ---------------------------------------------------------------------
# Usuarios (demo simple)
# ---------------------------------------------------------------------
def hash_password(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()

def create_user(u: str, p: str) -> bool:
    try:
        cur = get_conn().cursor()
        cur.execute("INSERT INTO users(username, password_hash, created_at) VALUES(?,?,?)",
                    (u, hash_password(p), datetime.utcnow().isoformat()))
        get_conn().commit()
        return True
    except sqlite3.IntegrityError:
        return False

def check_login(u: str, p: str) -> bool:
    cur = get_conn().cursor()
    cur.execute("SELECT password_hash FROM users WHERE username=?", (u,))
    row = cur.fetchone()
    return bool(row and row[0] == hash_password(p))

cur = get_conn().cursor()
cur.execute("SELECT COUNT(*) FROM users")
if cur.fetchone()[0] == 0:
    create_user("admin", "admin")

# ---------------------------------------------------------------------
# Finanzas
# ---------------------------------------------------------------------
def nominal_to_effective_monthly(tna: float, cap_per_year: int) -> float:
    """TNA→TEA→TEM"""
    m = max(1, int(cap_per_year))
    tea = (1.0 + (tna / m)) ** m - 1.0
    tem = (1.0 + tea) ** (1.0 / 12.0) - 1.0
    return tem

def tea_to_monthly(tea: float) -> float:
    return (1.0 + tea) ** (1.0 / 12.0) - 1.0

def french_payment(P: float, i_m: float, n: int) -> float:
    if n <= 0:
        return 0.0
    if i_m == 0:
        return P / n
    return P * (i_m * (1.0 + i_m) ** n) / ((1.0 + i_m) ** n - 1.0)

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
        "Periodo": 0, "Fecha": start_date.strftime("%Y-%m-%d"),
        "Saldo Inicial": 0.0, "Interés": 0.0, "Amortización": 0.0,
        "Cuota": 0.0, "Seguro": 0.0, "Gasto Adm": 0.0, "Cuota Total": 0.0,
        "Saldo Final": principal_neto, "Flujo Cliente": flujo_t0
    })

    saldo = principal_neto
    date_i = start_date
    total_months = grace_total + grace_partial + n_months
    cuota_fija = french_payment(saldo, i_m, n_months) if n_months > 0 else 0.0

    for t in range(1, total_months + 1):
        date_i = date_i + timedelta(days=30)  # 30/360
        interes = saldo * i_m

        if t <= grace_total:
            amort = 0.0
            cuota = 0.0
            saldo_final = saldo + interes
            pago_cliente = -(monthly_insurance + monthly_admin_fee)
        elif t <= grace_total + grace_partial:
            amort = 0.0
            cuota = interes
            saldo_final = saldo
            pago_cliente = -(cuota + monthly_insurance + monthly_admin_fee)
        else:
            cuota = cuota_fija
            amort = cuota - interes
            if t == total_months:
                amort = saldo
                cuota = interes + amort
            saldo_final = saldo - amort
            pago_cliente = -(cuota + monthly_insurance + monthly_admin_fee)

        rows.append({
            "Periodo": t, "Fecha": date_i.strftime("%Y-%m-%d"),
            "Saldo Inicial": saldo, "Interés": interes, "Amortización": amort,
            "Cuota": cuota, "Seguro": monthly_insurance, "Gasto Adm": monthly_admin_fee,
            "Cuota Total": cuota + monthly_insurance + monthly_admin_fee,
            "Saldo Final": saldo_final, "Flujo Cliente": pago_cliente
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

def normalize_income_to_case_currency(income_amount: float, income_cur: str, case_cur: str, tc: float = EXCHANGE_RATE) -> float:
    income_cur = (income_cur or "PEN").upper()
    case_cur = (case_cur or "PEN").upper()
    if income_amount is None:
        return 0.0
    if income_cur == case_cur:
        return float(income_amount)
    if income_cur == "USD" and case_cur == "PEN":
        return float(income_amount) * tc
    if income_cur == "PEN" and case_cur == "USD":
        return float(income_amount) / tc
    return float(income_amount)

# ---------------------------------------------------------------------
# Autenticación (sidebar)
# ---------------------------------------------------------------------
if "auth" not in st.session_state:
    st.session_state.auth = {"logged": False, "user": None}

with st.sidebar:
    st.header("🔐 Acceso")
    if not st.session_state.auth["logged"]:
        tabs = st.tabs(["Iniciar sesión", "Registrarse"])
        with tabs[0]:
            u = st.text_input("Usuario")
            p = st.text_input("Contraseña", type="password")
            if st.button("Entrar"):
                if check_login(u, p):
                    st.session_state.auth = {"logged": True, "user": u}
                    st.rerun()
                else:
                    st.error("Credenciales inválidas")
        with tabs[1]:
            u2 = st.text_input("Nuevo usuario")
            p2 = st.text_input("Nueva contraseña", type="password")
            if st.button("Crear cuenta"):
                if u2 and p2:
                    ok = create_user(u2, p2)
                    if ok:
                        st.success("Usuario creado, ahora inicia sesión")
                    else:
                        st.error("El usuario ya existe")
                else:
                    st.warning("Complete usuario y contraseña")
    else:
        st.write(f"👤 {st.session_state.auth['user']}")
        if st.button("Cerrar sesión"):
            st.session_state.clear()
            st.rerun()

if not st.session_state.auth.get("logged"):
    st.stop()

# ---------------------------------------------------------------------
# UI principal
# ---------------------------------------------------------------------
st.title("🏠 MiVivienda / Techo Propio – Simulador método francés (30/360)")
st.caption("Empresa inmobiliaria – cálculo de cronograma, VAN/TIR/TCEA y gestión de clientes & unidades")

sec1, sec2, sec3, sec4 = st.tabs([
    "1) Cliente y Unidad",
    "2) Configurar Préstamo",
    "3) Vincular & Guardar caso",
    "4) Casos & KPIs",
])

# ---------------------------------------------------------------------
# 1) Cliente y Unidad (CRUD robusto)
# ---------------------------------------------------------------------
with sec1:
    st.subheader("Datos del cliente")

    cur = get_conn().cursor()
    cur.execute("SELECT id, doc_id, full_name FROM clients ORDER BY full_name ASC")
    clients_list = cur.fetchall()
    client_labels = ["➕ Nuevo cliente"] + [f"{c[1]} – {c[2]} (ID {c[0]})" for c in clients_list]
    client_choice = st.selectbox("Editar cliente", client_labels, index=0)

    # Estado del formulario
    doc_id = ""; full_name = ""; income_monthly = 0.0; dependents = 0
    phone = ""; email = ""; employment_type = "Dependiente"; notes_client = ""
    income_currency = "PEN"
    editing_client_id = None

    if client_choice != "➕ Nuevo cliente":
        idx = client_labels.index(client_choice) - 1
        c = clients_list[idx]; editing_client_id = c[0]
        cur.execute("""SELECT doc_id, full_name, phone, email, income_monthly, dependents,
                              employment_type, notes, income_currency
                       FROM clients WHERE id=?""", (editing_client_id,))
        row = cur.fetchone()
        if row:
            (doc_id, full_name, phone, email, income_monthly, dependents,
             employment_type, notes_client, income_currency) = row

    colc1, colc2, colc3 = st.columns(3)
    with colc1:
        doc_id = st.text_input("Documento (OBLIGATORIO)", value=doc_id)
        full_name = st.text_input("Nombre completo (OBLIGATORIO)", value=full_name)
        income_monthly = st.number_input("Ingreso mensual (OBLIGATORIO)", min_value=0.0, step=100.0,
                                         value=float(income_monthly))
        dependents = st.number_input("Dependientes (OBLIGATORIO)", min_value=0, step=1,
                                     value=int(dependents))
    with colc2:
        phone = st.text_input("Teléfono 9 dígitos (OBLIGATORIO)", value=phone)
        email = st.text_input("Email (OBLIGATORIO)", value=email)
        employment_type = st.selectbox("Tipo de empleo (OBLIGATORIO)",
            ["Dependiente", "Independiente", "Mixto", "Otro"],
            index=["Dependiente","Independiente","Mixto","Otro"].index(employment_type)
                  if employment_type in ["Dependiente","Independiente","Mixto","Otro"] else 0)
        income_currency = st.selectbox("Moneda del ingreso", ["PEN", "USD"],
                                       index=0 if income_currency not in ["PEN","USD"]
                                       else ["PEN","USD"].index(income_currency))
    with colc3:
        notes_client = st.text_area("Notas socioeconómicas", value=notes_client)
        c1, c2 = st.columns(2)
        with c1:
            btn_save_client = st.button("💾 Guardar cliente")
        with c2:
            btn_delete_client = st.button("🗑️ Borrar cliente", disabled=(editing_client_id is None))

    def valid_email(s: str) -> bool:
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s or ""))

    def valid_phone_pe(s: str) -> bool:
        return bool(re.fullmatch(r"\d{9}", (s or "").strip()))

    if btn_save_client:
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
            try:
                cur = get_conn().cursor()
                if editing_client_id:
                    cur.execute("""
                        UPDATE clients SET doc_id=?, full_name=?, phone=?, email=?, income_monthly=?, dependents=?,
                        employment_type=?, notes=?, income_currency=?, updated_at=? WHERE id=?""",
                        (doc_id, full_name, phone, email, income_monthly, dependents,
                         employment_type, notes_client, income_currency, now, editing_client_id))
                else:
                    cur.execute("""
                        INSERT INTO clients(doc_id, full_name, phone, email, income_monthly, dependents,
                        employment_type, notes, income_currency, created_by, created_at, updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (doc_id, full_name, phone, email, income_monthly, dependents,
                         employment_type, notes_client, income_currency,
                         st.session_state.auth["user"], now, now))
                get_conn().commit()
                st.success("Cliente guardado")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("Documento duplicado: ya existe un cliente con ese documento")
            except Exception as e:
                st.error("No se pudo guardar el cliente: " + str(e))

    # ---------- BORRAR CLIENTE (persistiendo ID en session_state) ----------
    if "pending_delete_client_id" not in st.session_state:
        st.session_state.pending_delete_client_id = None

    if btn_delete_client and editing_client_id:
        st.session_state.pending_delete_client_id = int(editing_client_id)
        st.rerun()

    if st.session_state.pending_delete_client_id is not None:
        _cid = st.session_state.pending_delete_client_id
        cur = get_conn().cursor()
        cur.execute("SELECT full_name, doc_id FROM clients WHERE id=?", (_cid,))
        info = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM cases WHERE client_id=?", (_cid,))
        cnt = cur.fetchone()[0]

        st.warning(f"¿Eliminar **cliente** ID={_cid} ({(info[0] or '').strip()} – {(info[1] or '').strip()})?")
        if cnt > 0:
            st.info(f"Este cliente tiene **{cnt}** caso(s) asociado(s).")
            confirm_chk = st.checkbox("Sí, eliminar también sus casos.", key="del_client_chk")
        else:
            confirm_chk = True

        col_dc1, col_dc2, _ = st.columns(3)
        with col_dc1:
            do_delete = st.button("✅ Confirmar borrado", key="del_client_confirm")
        with col_dc2:
            cancel_delete = st.button("✖️ Cancelar", key="del_client_cancel")

        if cancel_delete:
            st.session_state.pending_delete_client_id = None
            st.rerun()

        if do_delete and confirm_chk:
            try:
                if cnt > 0:
                    cur.execute("DELETE FROM cases WHERE client_id=?", (_cid,))
                cur.execute("DELETE FROM clients WHERE id=?", (_cid,))
                get_conn().commit()
                st.session_state.pending_delete_client_id = None
                st.success("Cliente eliminado correctamente.")
                st.rerun()
            except Exception as e:
                st.error("No se pudo borrar el cliente: " + str(e))

    # -------------------- Unidades --------------------
    st.markdown("---")
    st.subheader("Unidad inmobiliaria")

    cur.execute("SELECT id, code, project FROM units ORDER BY code ASC")
    units_list = cur.fetchall()
    unit_labels = ["➕ Nueva unidad"] + [f"{u[1]} – {u[2] or ''} (ID {u[0]})" for u in units_list]
    unit_choice = st.selectbox("Editar unidad (Código y Nombre)", unit_labels, index=0)

    code = ""; project = ""; editing_unit_id = None
    if unit_choice != "➕ Nueva unidad":
        idx = unit_labels.index(unit_choice) - 1
        u = units_list[idx]; editing_unit_id = u[0]
        code = u[1] or ""; project = u[2] or ""

    colu1, colu2 = st.columns(2)
    with colu1:
        code = st.text_input("Código (OBLIGATORIO)", value=code)
    with colu2:
        project = st.text_input("Nombre (OBLIGATORIO)", value=project)
    c1u, c2u = st.columns(2)
    with c1u:
        btn_save_unit = st.button("💾 Guardar unidad")
    with c2u:
        btn_delete_unit = st.button("🗑️ Borrar unidad", disabled=(editing_unit_id is None))

    if btn_save_unit:
        if not code or not project:
            st.error("Complete Código y Nombre")
        else:
            now = datetime.utcnow().isoformat()
            try:
                cur = get_conn().cursor()
                if editing_unit_id:
                    cur.execute("UPDATE units SET code=?, project=?, updated_at=? WHERE id=?",
                                (code, project, now, editing_unit_id))
                else:
                    cur.execute("INSERT INTO units(code, project, created_by, created_at, updated_at) "
                                "VALUES(?,?,?,?,?)",
                                (code, project, st.session_state.auth["user"], now, now))
                get_conn().commit()
                st.success("Unidad guardada")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("Código duplicado: ya existe una unidad con ese código")
            except Exception as e:
                st.error("No se pudo guardar la unidad: " + str(e))

    # ---------- BORRAR UNIDAD (persistiendo ID en session_state) ----------
    if "pending_delete_unit_id" not in st.session_state:
        st.session_state.pending_delete_unit_id = None

    if btn_delete_unit and editing_unit_id:
        st.session_state.pending_delete_unit_id = int(editing_unit_id)
        st.rerun()

    if st.session_state.pending_delete_unit_id is not None:
        _uid = st.session_state.pending_delete_unit_id
        cur = get_conn().cursor()
        cur.execute("SELECT code, project FROM units WHERE id=?", (_uid,))
        info = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM cases WHERE unit_id=?", (_uid,))
        cnt = cur.fetchone()[0]

        st.warning(f"¿Eliminar **unidad** ID={_uid} ({(info[0] or '').strip()} – {(info[1] or '').strip()})?")
        if cnt > 0:
            st.info(f"Esta unidad tiene **{cnt}** caso(s) asociado(s).")
            confirm_chk_u = st.checkbox("Sí, eliminar también sus casos.", key="del_unit_chk")
        else:
            confirm_chk_u = True

        col_du1, col_du2, _ = st.columns(3)
        with col_du1:
            do_delete_u = st.button("✅ Confirmar borrado", key="del_unit_confirm")
        with col_du2:
            cancel_delete_u = st.button("✖️ Cancelar", key="del_unit_cancel")

        if cancel_delete_u:
            st.session_state.pending_delete_unit_id = None
            st.rerun()

        if do_delete_u and confirm_chk_u:
            try:
                if cnt > 0:
                    cur.execute("DELETE FROM cases WHERE unit_id=?", (_uid,))
                cur.execute("DELETE FROM units WHERE id=?", (_uid,))
                get_conn().commit()
                st.session_state.pending_delete_unit_id = None
                st.success("Unidad eliminada correctamente.")
                st.rerun()
            except Exception as e:
                st.error("No se pudo borrar la unidad: " + str(e))

# ---------------------------------------------------------------------
# 2) Configurar Préstamo
# ---------------------------------------------------------------------
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

    st.caption(f"Tasa efectiva mensual (TEM): {i_m*100:.5f}% | Convención 30/360 | Pagos vencidos")

    if grace_partial > grace_total:
        st.error("La gracia parcial no puede ser mayor que la gracia total.")
    if grace_total + grace_partial >= term_months:
        st.warning("La suma de gracia total y parcial no puede ser ≥ al plazo total.")

    if st.button("📅 Generar cronograma"):
        if grace_partial > grace_total:
            st.error("Corrija los meses de gracia: parcial > total.")
        elif grace_total + grace_partial >= term_months:
            st.error("Ajuste los meses de gracia vs plazo total.")
        else:
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
                "currency": currency, "principal": principal, "bono": bono,
                "term_months": term_months, "tasa_tipo": tasa_tipo,
                "tasa_anual": tasa_anual, "cap_m": cap_m,
                "grace_total": grace_total, "grace_partial": grace_partial,
                "fee_opening": fee_opening, "monthly_insurance": monthly_insurance,
                "monthly_admin_fee": monthly_admin_fee, "i_m": i_m,
            }
            st.success("Cronograma generado. Ahora vincúlalo y guarda en la pestaña 3)")

# ---------------------------------------------------------------------
# 3) Vincular & Guardar caso
# ---------------------------------------------------------------------
with sec3:
    st.subheader("Vincular cliente + préstamo y guardar")

    if "schedule_df" not in st.session_state:
        st.info("Primero configure y genere un cronograma en la pestaña 2.")
    else:
        cfg = st.session_state.get("schedule_cfg", {})
        cur = get_conn().cursor()
        cur.execute("SELECT id, full_name FROM clients ORDER BY full_name ASC")
        clients_for_save = cur.fetchall()
        cur.execute("SELECT id, code, project FROM units ORDER BY code ASC")
        units_for_save = cur.fetchall()

        sel_client_label_to_id = {f"{c[1]} (ID {c[0]})": c[0] for c in clients_for_save}
        sel_unit_label_to_id = {f"{u[1]} – {u[2]} (ID {u[0]})": u[0] for u in units_for_save}

        client_label = st.selectbox("Cliente", ["- Seleccione -"] + list(sel_client_label_to_id.keys()))
        unit_label = st.selectbox("Unidad", ["- Seleccione -"] + list(sel_unit_label_to_id.keys()))
        case_name = st.text_input("Nombre del caso", value="")

        if st.button("💾 Guardar caso en base de datos"):
            if client_label == "- Seleccione -" or unit_label == "- Seleccione -" or not case_name:
                st.error("Seleccione cliente, unidad y defina un nombre para el caso")
            else:
                client_id = sel_client_label_to_id[client_label]
                unit_id = sel_unit_label_to_id[unit_label]
                params = {**cfg, "generated_at": datetime.utcnow().isoformat()}
                cur.execute(
                    "INSERT INTO cases(user, client_id, unit_id, case_name, params_json, created_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        st.session_state.auth["user"], client_id, unit_id,
                        case_name, pd.Series(params).to_json(), datetime.utcnow().isoformat()
                    ),
                )
                get_conn().commit()
                st.success("Caso guardado correctamente")

# ---------------------------------------------------------------------
# 4) Casos & KPIs (incluye borrar caso)
# ---------------------------------------------------------------------
with sec4:
    st.subheader("Casos guardados y KPIs del caso seleccionado")

    cur = get_conn().cursor()
    cur.execute("""
        SELECT cases.id, cases.case_name, clients.full_name, units.code, units.project
        FROM cases
        LEFT JOIN clients ON clients.id = cases.client_id
        LEFT JOIN units ON units.id = cases.unit_id
        ORDER BY cases.id DESC
    """)
    rows = cur.fetchall()

    if not rows:
        st.info("Aún no hay casos guardados.")
    else:
        label_to_caseid = {f"#{r[0]} – {r[2] or 'Cliente?'} – {r[3] or 'CODE?'} – {r[1]}": r[0] for r in rows}
        case_label = st.selectbox("Casos", options=list(label_to_caseid.keys()), key="kpi_case_selector")
        case_id = label_to_caseid[case_label]

        # Botón de BORRAR CASO (con rerun inmediato)
        del_col1, del_col2 = st.columns([1, 3])
        with del_col1:
            if st.button("🗑️ Borrar este caso", key="btn_delete_case"):
                try:
                    cur.execute("DELETE FROM cases WHERE id=?", (case_id,))
                    get_conn().commit()
                    st.success("Caso eliminado")
                    st.rerun()
                except Exception as e:
                    st.error("No se pudo borrar el caso: " + str(e))
                    st.stop()

        # Cargar params del caso + ingreso del cliente y moneda del ingreso
        cur.execute("""
            SELECT cases.case_name,
                   clients.full_name,
                   units.code, units.project,
                   cases.params_json,
                   clients.income_monthly, clients.income_currency
            FROM cases
            LEFT JOIN clients ON clients.id = cases.client_id
            LEFT JOIN units ON units.id = cases.unit_id
            WHERE cases.id=?
        """, (case_id,))
        row = cur.fetchone()

        if row and row[4]:
            case_name, client_name, code_u, proj_u, params_json, cli_income, cli_income_cur = row
            params = pd.Series(json.loads(params_json))   # ← sin FutureWarning

            df2 = build_schedule(
                principal=params["principal"],
                i_m=params["i_m"],
                n_months=int(params["term_months"] - (params["grace_total"] + params["grace_partial"])),
                grace_total=int(params["grace_total"]),
                grace_partial=int(params["grace_partial"]),
                start_date=datetime.today(),
                fee_opening=params["fee_opening"],
                monthly_insurance=params["monthly_insurance"],
                monthly_admin_fee=params["monthly_admin_fee"],
                bono_monto=params["bono"],
            )

            # KPIs por caso
            cashflows = df2["Flujo Cliente"].to_numpy()
            irr_m = irr(cashflows)
            tcea = (1 + irr_m) ** 12 - 1 if np.isfinite(irr_m) else np.nan
            symbol = "S/." if (params.get("currency", "PEN") == "PEN") else "$"

            principal_bruto = float(params.get("principal", 0.0))
            bono = float(params.get("bono", 0.0))
            principal_neto = max(0.0, principal_bruto - bono)
            g_total = int(params.get("grace_total", 0))
            g_parcial = int(params.get("grace_partial", 0))
            n_total = int(params.get("term_months", 0))
            n_amort = max(0, n_total - (g_total + g_parcial))
            i_m = float(params.get("i_m", float("nan")))
            case_currency = params.get("currency", "PEN")

            mask_amort = df2["Amortización"] > 0
            if mask_amort.any():
                primera = df2.loc[mask_amort].iloc[0]
                cuota_francesa = float(primera["Cuota"])
                cuota_inicial_total = float(primera["Cuota Total"])
                fecha_primera_cuota = str(primera["Fecha"])
            else:
                cuota_francesa = 0.0; cuota_inicial_total = 0.0; fecha_primera_cuota = "-"

            df_pos = df2[df2["Periodo"] > 0]
            interes_total = float(df_pos["Interés"].sum())
            amort_total   = float(df_pos["Amortización"].sum())
            seg_total     = float(df_pos["Seguro"].sum())
            gadm_total    = float(df_pos["Gasto Adm"].sum())
            costo_total_cliente = float(-df_pos["Flujo Cliente"].sum())

            # Normalizar ingreso mensual del cliente a la moneda del caso
            ingreso_norm = normalize_income_to_case_currency(cli_income or 0.0,
                                                             cli_income_cur or "PEN",
                                                             case_currency,
                                                             EXCHANGE_RATE)
            if ingreso_norm and ingreso_norm > 0:
                ratio_cuota_ingreso = (cuota_inicial_total / ingreso_norm) * 100.0
            else:
                ratio_cuota_ingreso = np.nan

            with st.expander("📄 Detalle del caso", expanded=True):
                st.write(f"**Caso**: #{case_id} – {case_name}")
                st.write(f"**Cliente**: {client_name or '-'}  |  **Unidad**: {code_u or '-'} – {proj_u or '-'}")

            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            with r1c1: st.metric("TIR mensual (TIRM)", f"{irr_m*100:.3f}%" if np.isfinite(irr_m) else "No converge")
            with r1c2: st.metric("TCEA (anual efectiva)", f"{tcea*100:.3f}%" if np.isfinite(tcea) else "-")
            with r1c3: st.metric("Total pagado (∑ pagos)", f"{symbol} {costo_total_cliente:,.2f}")
            with r1c4: st.metric("Monto financiado (neto)", f"{symbol} {principal_neto:,.2f}")

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            with r2c1: st.metric("Monto bruto", f"{symbol} {principal_bruto:,.2f}")
            with r2c2: st.metric("Bono", f"{symbol} {bono:,.2f}")
            with r2c3: st.metric("TEM (i_m)", f"{i_m*100:.4f}%" if np.isfinite(i_m) else "-")
            with r2c4: st.metric("Plazo amortización (meses)", f"{n_amort}")

            r3c1, r3c2, r3c3, r3c4 = st.columns(4)
            with r3c1: st.metric("Gracia total / parcial", f"{g_total} / {g_parcial}")
            with r3c2: st.metric("Cuota francesa (sin gastos)", f"{symbol} {cuota_francesa:,.2f}")
            with r3c3: st.metric("Cuota inicial total (con gastos)", f"{symbol} {cuota_inicial_total:,.2f}")
            with r3c4: st.metric("1ra fecha de cuota", fecha_primera_cuota)

            r4c1, r4c2, r4c3, r4c4 = st.columns(4)
            with r4c1: st.metric("Interés total", f"{symbol} {interes_total:,.2f}")
            with r4c2: st.metric("Amortización total", f"{symbol} {amort_total:,.2f}")
            with r4c3: st.metric("Seguros totales", f"{symbol} {seg_total:,.2f}")
            with r4c4: st.metric("Gastos Adm totales", f"{symbol} {gadm_total:,.2f}")

            # KPIs de ingreso normalizado y esfuerzo
            r5c1, r5c2, r5c3, r5c4 = st.columns(4)
            sym_case = "S/." if case_currency == "PEN" else "$"
            with r5c1: st.metric(f"Ingreso mensual ({case_currency})", f"{sym_case} {ingreso_norm:,.2f}")
            with r5c2: st.metric("Cuota/Ingreso (%)", f"{ratio_cuota_ingreso:.2f}%" if np.isfinite(ratio_cuota_ingreso) else "-")
            with r5c3: st.metric("TC usado", f"1 USD = {EXCHANGE_RATE:.2f} PEN")
            with r5c4: st.write("")

            st.markdown("### 📅 Cronograma del caso seleccionado")
            st.dataframe(
                df2.style.format({
                    "Saldo Inicial": "{:,.2f}", "Interés": "{:,.2f}", "Amortización": "{:,.2f}",
                    "Cuota": "{:,.2f}", "Seguro": "{:,.2f}", "Gasto Adm": "{:,.2f}",
                    "Cuota Total": "{:,.2f}", "Saldo Final": "{:,.2f}", "Flujo Cliente": "{:,.2f}",
                }),
                width='stretch'
            )
            csv2 = df2.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Descargar cronograma (CSV)", csv2,
                               file_name=f"cronograma_caso_{case_id}.csv", mime="text/csv")
        else:
            st.error("No se pudo leer el caso seleccionado")
