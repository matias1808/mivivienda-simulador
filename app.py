# app.py
# -*- coding: utf-8 -*-
import os
import re
import json
import hashlib
import sqlite3
import tempfile
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------
st.set_page_config(page_title="MiVivienda – Simulador", page_icon="🏠", layout="wide")

EXCHANGE_RATE = 3.75  # 1 USD = 3.75 PEN (para normalizar ingresos)

# --- Entidades financieras (TEA anual) ---
FINANCIAL_ENTITIES_TEA = {
    "BBVA": 0.1368,
    "Banco de Comercio": 0.1250,
    "Banco Pichincha": 0.1500,
    "Scotiabank": 0.1240,
    "Interbank": 0.1260,
    "Banco GNB": 0.1300,
    "Caja Metropolitana de Lima": 0.1300,
    "Caja Tacna": 0.1403,
    "Caja Trujillo": 0.1340,
    "Financiera Confianza": 0.1230,
    "BanBif": 0.1300,
}

# Demo: clave de acceso a "Base de datos" (NO se muestra en UI ni bot)
DB_VIEW_KEY = "12345"

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
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def _cols(cur: sqlite3.Cursor, table: str) -> list[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


@st.cache_resource(show_spinner=False)
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # users
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        )
    """)

    # clients
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

    # migración: income_currency
    cur.execute("PRAGMA table_info(clients)")
    ccols = [r[1] for r in cur.fetchall()]
    if "income_currency" not in ccols:
        cur.execute("ALTER TABLE clients ADD COLUMN income_currency TEXT DEFAULT 'PEN'")

    # cases (sin unidad inmobiliaria)
    if not _table_exists(cur, "cases"):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cases(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT,
                client_id INTEGER,
                case_name TEXT,
                params_json TEXT,
                created_at TEXT,
                FOREIGN KEY(client_id) REFERENCES clients(id)
            )
        """)

    # índices
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_doc_id ON clients(doc_id)")

    # LEGACY COMPAT: si tu BD vieja tiene cases.unit_id (FK), aseguramos tabla units para no romper FK.
    try:
        if "unit_id" in _cols(cur, "cases") and not _table_exists(cur, "units"):
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
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_units_code ON units(code)")
    except Exception:
        pass

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
        cur.execute(
            "INSERT INTO users(username, password_hash, created_at) VALUES(?,?,?)",
            (u, hash_password(p), datetime.utcnow().isoformat()),
        )
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
def tea_to_monthly(tea: float) -> float:
    return (1.0 + float(tea)) ** (1.0 / 12.0) - 1.0


def french_payment(P: float, i_m: float, n: int) -> float:
    P = float(P)
    i_m = float(i_m)
    n = int(n)
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
    start_date: Optional[datetime] = None,
    fee_opening: float = 0.0,
    monthly_insurance: float = 0.0,   # desgravamen (monto fijo mensual)
    monthly_admin_fee: float = 0.0,   # gasto adm (monto fijo mensual)
    bono_monto: float = 0.0,
) -> pd.DataFrame:
    principal = float(principal)
    i_m = float(i_m)
    n_months = int(n_months)
    grace_total = int(grace_total)
    grace_partial = int(grace_partial)
    fee_opening = float(fee_opening)
    monthly_insurance = float(monthly_insurance)
    monthly_admin_fee = float(monthly_admin_fee)
    bono_monto = float(bono_monto)

    principal_neto = max(0.0, principal - bono_monto)
    if start_date is None:
        start_date = datetime.today()

    rows = []

    # Cliente recibe el préstamo, pero se descuenta comisión en t0
    flujo_t0_cliente = principal - fee_opening
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
        "Flujo Cliente": flujo_t0_cliente
    })

    saldo = principal_neto
    date_i = start_date
    total_months = grace_total + grace_partial + n_months
    cuota_fija = french_payment(saldo, i_m, n_months) if n_months > 0 else 0.0

    for t in range(1, total_months + 1):
        date_i = date_i + timedelta(days=30)  # 30/360
        interes = saldo * i_m

        if t <= grace_total:
            # gracia total: no paga cuota (interés se capitaliza)
            amort = 0.0
            cuota = 0.0
            saldo_final = saldo + interes
            pago_cliente = -(monthly_insurance + monthly_admin_fee)
        elif t <= grace_total + grace_partial:
            # gracia parcial: paga solo interés
            amort = 0.0
            cuota = interes
            saldo_final = saldo
            pago_cliente = -(cuota + monthly_insurance + monthly_admin_fee)
        else:
            # amortización francesa
            cuota = cuota_fija
            amort = cuota - interes
            if t == total_months:
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
            "Flujo Cliente": pago_cliente
        })
        saldo = saldo_final

    return pd.DataFrame(rows)


def irr(cashflows: np.ndarray, guess: float = 0.01, max_iter: int = 160, tol: float = 1e-10) -> float:
    r = float(guess)
    cashflows = np.array(cashflows, dtype=float)
    for _ in range(int(max_iter)):
        t = np.arange(len(cashflows), dtype=float)
        denom = (1.0 + r) ** t
        f = np.sum(cashflows / denom)
        df = -np.sum(t * cashflows / ((1.0 + r) ** (t + 1.0)))
        if abs(df) < 1e-14:
            break
        r_new = r - f / df
        if abs(r_new - r) < tol:
            return float(r_new)
        r = float(r_new)
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


def cases_has_unit_id() -> bool:
    try:
        cur = get_conn().cursor()
        return "unit_id" in _cols(cur, "cases")
    except Exception:
        return False


def insert_case(user: str, client_id: int, case_name: str, params_json: str):
    cur = get_conn().cursor()
    now = datetime.utcnow().isoformat()

    if cases_has_unit_id():
        # BD legacy: insert con unit_id NULL
        cur.execute(
            "INSERT INTO cases(user, client_id, unit_id, case_name, params_json, created_at) VALUES(?,?,?,?,?,?)",
            (user, client_id, None, case_name, params_json, now),
        )
    else:
        cur.execute(
            "INSERT INTO cases(user, client_id, case_name, params_json, created_at) VALUES(?,?,?,?,?)",
            (user, client_id, case_name, params_json, now),
        )
    get_conn().commit()

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

# Contexto del bot (solo si hay caso seleccionado)
if "helpbot_ctx" not in st.session_state:
    st.session_state["helpbot_ctx"] = None

# Gate BD
if "db_unlocked" not in st.session_state:
    st.session_state["db_unlocked"] = False

# ---------------------------------------------------------------------
# UI principal
# ---------------------------------------------------------------------
st.title("🏠 MiVivienda / Techo Propio – Simulador método francés (30/360)")
st.caption("Cálculo de cronograma, TCEA/TREA y gestión de clientes")

sec1, sec2, sec3, sec4, sec5 = st.tabs([
    "Cliente",
    "Configurar Préstamo",
    "Guardar caso",
    "Casos & KPIs",
    "📦 Base de datos",
])

# ---------------------------------------------------------------------
# 1) Cliente (CRUD)
# ---------------------------------------------------------------------
with sec1:
    st.subheader("Datos del cliente")

    cur = get_conn().cursor()
    cur.execute("SELECT id, doc_id, full_name FROM clients ORDER BY full_name ASC")
    clients_list = cur.fetchall()
    client_labels = ["➕ Nuevo cliente"] + [f"{c[1]} – {c[2]} (ID {c[0]})" for c in clients_list]
    client_choice = st.selectbox("Editar cliente", client_labels, index=0)

    doc_id = ""; full_name = ""; income_monthly = 0.0; dependents = 0
    phone = ""; email = ""; employment_type = "Dependiente"; notes_client = ""
    income_currency = "PEN"; editing_client_id = None

    if client_choice != "➕ Nuevo cliente":
        idx = client_labels.index(client_choice) - 1
        c = clients_list[idx]
        editing_client_id = c[0]
        cur.execute("""
            SELECT doc_id, full_name, phone, email, income_monthly, dependents,
                   employment_type, notes, income_currency
            FROM clients WHERE id=?
        """, (editing_client_id,))
        row = cur.fetchone()
        if row:
            (doc_id, full_name, phone, email, income_monthly, dependents,
             employment_type, notes_client, income_currency) = row

    colc1, colc2, colc3 = st.columns(3)
    with colc1:
        doc_id = st.text_input("Documento (OBLIGATORIO)", value=doc_id)
        full_name = st.text_input("Nombre completo (OBLIGATORIO)", value=full_name)
        income_monthly = st.number_input("Ingreso mensual (OBLIGATORIO)", min_value=0.0, step=100.0, value=float(income_monthly))
        dependents = st.number_input("Dependientes (OBLIGATORIO)", min_value=0, step=1, value=int(dependents))
    with colc2:
        phone = st.text_input("Teléfono 9 dígitos (OBLIGATORIO)", value=phone)
        email = st.text_input("Email (OBLIGATORIO)", value=email)
        employment_type = st.selectbox(
            "Tipo de empleo (OBLIGATORIO)",
            ["Dependiente", "Independiente", "Mixto", "Otro"],
            index=["Dependiente","Independiente","Mixto","Otro"].index(employment_type)
            if employment_type in ["Dependiente","Independiente","Mixto","Otro"] else 0
        )
        income_currency = st.selectbox(
            "Moneda del ingreso",
            ["PEN", "USD"],
            index=0 if income_currency not in ["PEN","USD"] else ["PEN","USD"].index(income_currency)
        )
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

    # -------- BORRAR CLIENTE (confirmación) --------
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

        st.warning(f"¿Eliminar cliente ID={_cid} ({(info[0] or '').strip()} – {(info[1] or '').strip()})?")
        if cnt > 0:
            st.info(f"Este cliente tiene {cnt} caso(s) asociado(s).")
            confirm_chk = st.checkbox("Sí, eliminar también sus casos.", key="del_client_chk")
        else:
            confirm_chk = True

        cdc1, cdc2, _ = st.columns(3)
        with cdc1:
            do_delete = st.button("✅ Confirmar borrado", key="del_client_confirm")
        with cdc2:
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
                st.success("Cliente eliminado")
                st.rerun()
            except Exception as e:
                st.error("No se pudo borrar el cliente: " + str(e))

# ---------------------------------------------------------------------
# 2) Configurar Préstamo
# ---------------------------------------------------------------------
with sec2:
    st.subheader("Configuración del préstamo (con cuota inicial + entidad financiera)")

    col1, col2, col3 = st.columns(3)

    with col1:
        currency = st.selectbox("Moneda", ["PEN", "USD"], index=0)
        valor_inmueble = st.number_input("Valor del inmueble", min_value=0.0, step=1000.0)
        cuota_inicial = st.number_input("Cuota inicial", min_value=0.0, step=1000.0)
        bono = st.number_input("Bono (Techo Propio u otro)", min_value=0.0, step=500.0)
        term_months = st.number_input("Plazo (meses)", min_value=1, step=1)

        monto_prestamo = max(0.0, valor_inmueble - cuota_inicial - bono)
        sym_case = "S/." if currency == "PEN" else "$"
        st.metric("Monto a financiar (PRÉSTAMO)", f"{sym_case} {monto_prestamo:,.2f}")

    with col2:
        entidad = st.selectbox("Entidad financiera", list(FINANCIAL_ENTITIES_TEA.keys()), index=0)
        tasa_anual = float(FINANCIAL_ENTITIES_TEA[entidad])
        st.number_input("TEA anual (%)", value=float(tasa_anual * 100.0), disabled=True, format="%.2f")
        grace_total = st.number_input("Gracia total (meses)", min_value=0, step=1)

    with col3:
        grace_partial = st.number_input("Gracia parcial (meses)", min_value=0, step=1)
        fee_opening = st.number_input("Comisión de apertura (t0)", min_value=0.0, step=100.0)
        seguro_desgravamen = st.number_input("Seguro de desgravamen (mensual)", min_value=0.0, step=10.0)
        gasto_administrativo = st.number_input("Gasto administrativo (mensual)", min_value=0.0, step=10.0)

    i_m = tea_to_monthly(tasa_anual)
    st.caption(
        f"Entidad: {entidad} | TEA: {tasa_anual*100:.2f}% | TEM: {i_m*100:.5f}% | Convención 30/360 | Pagos vencidos"
    )

    if grace_partial > grace_total:
        st.error("La gracia parcial no puede ser mayor que la gracia total.")
    if grace_total + grace_partial >= term_months:
        st.warning("La suma de gracia total y parcial no puede ser ≥ al plazo total.")

    if st.button("📅 Generar cronograma"):
        ok = True

        if valor_inmueble <= 0:
            st.error("Ingrese un Valor del inmueble > 0.")
            ok = False
        if cuota_inicial + bono > valor_inmueble:
            st.error("Cuota inicial + Bono no puede ser mayor que el Valor del inmueble.")
            ok = False
        if monto_prestamo <= 0:
            st.error("El monto a financiar queda en 0. Ajusta cuota inicial/bono o valor del inmueble.")
            ok = False
        if grace_partial > grace_total:
            st.error("Corrija los meses de gracia: parcial > total.")
            ok = False
        if grace_total + grace_partial >= term_months:
            st.error("Ajuste los meses de gracia vs plazo total.")
            ok = False

        if ok:
            df = build_schedule(
                principal=monto_prestamo,
                i_m=i_m,
                n_months=int(term_months - (grace_total + grace_partial)),
                grace_total=int(grace_total),
                grace_partial=int(grace_partial),
                start_date=datetime.today(),
                fee_opening=fee_opening,
                monthly_insurance=seguro_desgravamen,
                monthly_admin_fee=gasto_administrativo,
                bono_monto=0.0,
            )

            st.session_state["schedule_df"] = df
            st.session_state["schedule_cfg"] = {
                "currency": currency,
                "entidad": entidad,
                "tasa_anual": tasa_anual,
                "i_m": i_m,
                "valor_inmueble": valor_inmueble,
                "cuota_inicial": cuota_inicial,
                "bono": bono,
                "principal": monto_prestamo,
                "term_months": term_months,
                "grace_total": grace_total,
                "grace_partial": grace_partial,
                "fee_opening": fee_opening,
                "monthly_insurance": seguro_desgravamen,
                "monthly_admin_fee": gasto_administrativo,
                "principal_mode": "NET",
                "generated_at": datetime.utcnow().isoformat(),
            }
            st.success("Cronograma generado. Guarda el caso en la pestaña 3).")

# ---------------------------------------------------------------------
# 3) Guardar caso
# ---------------------------------------------------------------------
with sec3:
    st.subheader("Guardar caso (cliente + préstamo)")

    if "schedule_df" not in st.session_state:
        st.info("Primero genere un cronograma en la pestaña 2.")
    else:
        cfg = st.session_state.get("schedule_cfg", {})
        cur = get_conn().cursor()
        cur.execute("SELECT id, full_name FROM clients ORDER BY full_name ASC")
        clients_for_save = cur.fetchall()

        sel_client_label_to_id = {f"{c[1]} (ID {c[0]})": c[0] for c in clients_for_save}
        client_label = st.selectbox("Cliente", ["- Seleccione -"] + list(sel_client_label_to_id.keys()))
        case_name = st.text_input("Nombre del caso", value="")

        if st.button("💾 Guardar caso en base de datos"):
            if client_label == "- Seleccione -" or not case_name:
                st.error("Seleccione cliente y defina un nombre para el caso")
            else:
                client_id = int(sel_client_label_to_id[client_label])
                params_json = pd.Series(cfg).to_json()
                try:
                    insert_case(st.session_state.auth["user"], client_id, case_name, params_json)
                    st.success("Caso guardado correctamente")
                except Exception as e:
                    st.error("No se pudo guardar el caso: " + str(e))

# ---------------------------------------------------------------------
# 4) Casos & KPIs
# ---------------------------------------------------------------------
with sec4:
    st.subheader("Casos guardados y KPIs del caso seleccionado")

    cur = get_conn().cursor()
    cur.execute("""
        SELECT cases.id, cases.case_name, clients.full_name
        FROM cases
        LEFT JOIN clients ON clients.id = cases.client_id
        ORDER BY cases.id DESC
    """)
    rows = cur.fetchall()

    if not rows:
        st.info("Aún no hay casos guardados.")
        st.session_state["helpbot_ctx"] = None
    else:
        label_to_caseid = {f"#{r[0]} – {r[2] or 'Cliente?'} – {r[1]}": r[0] for r in rows}
        case_label = st.selectbox("Casos", options=list(label_to_caseid.keys()), key="kpi_case_selector")
        case_id = label_to_caseid[case_label]

        del_col1, _ = st.columns([1, 3])
        with del_col1:
            if st.button("🗑️ Borrar este caso", key="btn_delete_case"):
                try:
                    cur.execute("DELETE FROM cases WHERE id=?", (case_id,))
                    get_conn().commit()
                    st.success("Caso eliminado")
                    st.session_state["helpbot_ctx"] = None
                    st.rerun()
                except Exception as e:
                    st.error("No se pudo borrar el caso: " + str(e))

        cur.execute("""
            SELECT cases.case_name,
                   clients.full_name,
                   cases.params_json,
                   clients.income_monthly, clients.income_currency
            FROM cases
            LEFT JOIN clients ON clients.id = cases.client_id
            WHERE cases.id=?
        """, (case_id,))
        row = cur.fetchone()

        if not (row and row[2]):
            st.error("No se pudo leer el caso seleccionado")
            st.session_state["helpbot_ctx"] = None
        else:
            case_name, client_name, params_json, cli_income, cli_income_cur = row
            params = pd.Series(json.loads(params_json))

            principal_mode = str(params.get("principal_mode", "LEGACY"))
            bono_for_schedule = 0.0 if principal_mode == "NET" else float(params.get("bono", 0.0))

            df2 = build_schedule(
                principal=float(params.get("principal", 0.0)),
                i_m=float(params.get("i_m", 0.0)),
                n_months=int(float(params.get("term_months", 0)) - (float(params.get("grace_total", 0)) + float(params.get("grace_partial", 0)))),
                grace_total=int(float(params.get("grace_total", 0))),
                grace_partial=int(float(params.get("grace_partial", 0))),
                start_date=datetime.today(),
                fee_opening=float(params.get("fee_opening", 0.0)),
                monthly_insurance=float(params.get("monthly_insurance", 0.0)),
                monthly_admin_fee=float(params.get("monthly_admin_fee", 0.0)),
                bono_monto=bono_for_schedule,
            )

            case_currency = str(params.get("currency", "PEN"))
            symbol = "S/." if case_currency == "PEN" else "$"

            # Cliente: TCEA
            cashflows_cli = df2["Flujo Cliente"].to_numpy()
            irr_m_cli = irr(cashflows_cli)
            tcea = (1 + irr_m_cli) ** 12 - 1 if np.isfinite(irr_m_cli) else np.nan

            principal = float(params.get("principal", 0.0))
            fee_opening = float(params.get("fee_opening", 0.0))
            i_m = float(params.get("i_m", float("nan")))
            df_pos = df2[df2["Periodo"] > 0]

            # Banco: TREA (crédito) solo Cuota (sin cargos)
            cf_bank_credit = [-principal + fee_opening]
            for _, r in df_pos.iterrows():
                cf_bank_credit.append(float(r["Cuota"]))
            irr_m_bank_credit = irr(np.array(cf_bank_credit, dtype=float))
            trea_credit = (1 + irr_m_bank_credit) ** 12 - 1 if np.isfinite(irr_m_bank_credit) else np.nan

            # Banco: TREA (total cobros) Cuota Total
            cf_bank_total = [-principal + fee_opening]
            for _, r in df_pos.iterrows():
                cf_bank_total.append(float(r["Cuota Total"]))
            irr_m_bank_total = irr(np.array(cf_bank_total, dtype=float))
            trea_total = (1 + irr_m_bank_total) ** 12 - 1 if np.isfinite(irr_m_bank_total) else np.nan

            g_total = int(float(params.get("grace_total", 0)))
            g_parcial = int(float(params.get("grace_partial", 0)))
            n_total = int(float(params.get("term_months", 0)))
            n_amort = max(0, n_total - (g_total + g_parcial))

            mask_amort = df2["Amortización"] > 0
            if mask_amort.any():
                primera = df2.loc[mask_amort].iloc[0]
                cuota_francesa = float(primera["Cuota"])
                cuota_inicial_total = float(primera["Cuota Total"])
                fecha_primera_cuota = str(primera["Fecha"])
            else:
                cuota_francesa = 0.0
                cuota_inicial_total = 0.0
                fecha_primera_cuota = "-"

            interes_total = float(df_pos["Interés"].sum())
            amort_total = float(df_pos["Amortización"].sum())
            seg_total = float(df_pos["Seguro"].sum())
            gadm_total = float(df_pos["Gasto Adm"].sum())
            costo_total_cliente = float(-df_pos["Flujo Cliente"].sum())

            ingreso_norm = normalize_income_to_case_currency(cli_income or 0.0, cli_income_cur or "PEN", case_currency, EXCHANGE_RATE)
            ratio_cuota_ingreso = (cuota_inicial_total / ingreso_norm) * 100.0 if ingreso_norm > 0 else np.nan

            entidad = str(params.get("entidad", "-"))
            tea_case = float(params.get("tasa_anual", np.nan))
            valor_inmueble = float(params.get("valor_inmueble", np.nan))
            cuota_ini = float(params.get("cuota_inicial", np.nan))
            bono = float(params.get("bono", 0.0))

            # Contexto para Helpy
            st.session_state["helpbot_ctx"] = {
                "case_id": int(case_id),
                "case_name": str(case_name),
                "client_name": str(client_name or ""),
                "currency": str(case_currency),
                "symbol": str(symbol),
                "entidad": str(entidad),
                "tea": float(tea_case) if np.isfinite(tea_case) else None,
                "tem": float(i_m) if np.isfinite(i_m) else None,
                "principal": float(principal),
                "fee_opening": float(fee_opening),
                "monthly_insurance": float(params.get("monthly_insurance", 0.0)),
                "monthly_admin_fee": float(params.get("monthly_admin_fee", 0.0)),
                "cuota": float(cuota_francesa),
                "cuota_total": float(cuota_inicial_total),
                "tcea": float(tcea) if np.isfinite(tcea) else None,
                "trea_credit": float(trea_credit) if np.isfinite(trea_credit) else None,
                "trea_total": float(trea_total) if np.isfinite(trea_total) else None,
                "interes_total": float(interes_total),
                "seg_total": float(seg_total),
                "adm_total": float(gadm_total),
                "total_pagado": float(costo_total_cliente),
                "plazo_total": int(n_total),
                "gracia_total": int(g_total),
                "gracia_parcial": int(g_parcial),
            }

            with st.expander("📄 Detalle del caso", expanded=True):
                st.write(f"**Caso**: #{case_id} – {case_name}")
                st.write(f"**Cliente**: {client_name or '-'}")
                if entidad != "-":
                    if np.isfinite(tea_case):
                        st.write(f"**Entidad**: {entidad} | **TEA**: {tea_case*100:.2f}%")
                    else:
                        st.write(f"**Entidad**: {entidad}")
                if np.isfinite(valor_inmueble) and np.isfinite(cuota_ini):
                    st.write(
                        f"**Valor inmueble**: {symbol} {valor_inmueble:,.2f} | "
                        f"**Cuota inicial**: {symbol} {cuota_ini:,.2f} | "
                        f"**Bono**: {symbol} {bono:,.2f}"
                    )

            r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
            with r1c1: st.metric("TIR mensual (TIRM)", f"{irr_m_cli*100:.3f}%" if np.isfinite(irr_m_cli) else "No converge")
            with r1c2: st.metric("TCEA (cliente)", f"{tcea*100:.3f}%" if np.isfinite(tcea) else "-")
            with r1c3: st.metric("TREA (crédito)", f"{trea_credit*100:.3f}%" if np.isfinite(trea_credit) else "-")
            with r1c4: st.metric("Total pagado (∑ pagos)", f"{symbol} {costo_total_cliente:,.2f}")
            with r1c5: st.metric("Monto préstamo", f"{symbol} {principal:,.2f}")

            st.caption(f"TREA Total Cobros (cuota+seguro+adm): {trea_total*100:.3f}%" if np.isfinite(trea_total) else "TREA Total Cobros: -")

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            with r2c1: st.metric("TEM (i_m)", f"{i_m*100:.4f}%" if np.isfinite(i_m) else "-")
            with r2c2: st.metric("Plazo amortización (meses)", f"{n_amort}")
            with r2c3: st.metric("Gracia total / parcial", f"{g_total} / {g_parcial}")
            with r2c4: st.metric("1ra fecha de cuota", fecha_primera_cuota)

            r3c1, r3c2, r3c3, r3c4 = st.columns(4)
            with r3c1: st.metric("Cuota francesa (sin gastos)", f"{symbol} {cuota_francesa:,.2f}")
            with r3c2: st.metric("Cuota inicial total (con gastos)", f"{symbol} {cuota_inicial_total:,.2f}")
            with r3c3: st.metric("Cuota/Ingreso (%)", f"{ratio_cuota_ingreso:.2f}%" if np.isfinite(ratio_cuota_ingreso) else "-")
            with r3c4: st.metric("TC usado", f"1 USD = {EXCHANGE_RATE:.2f} PEN")

            r4c1, r4c2, r4c3, r4c4 = st.columns(4)
            with r4c1: st.metric("Interés total", f"{symbol} {interes_total:,.2f}")
            with r4c2: st.metric("Amortización total", f"{symbol} {amort_total:,.2f}")
            with r4c3: st.metric("Seguros totales", f"{symbol} {seg_total:,.2f}")
            with r4c4: st.metric("Gastos Adm totales", f"{symbol} {gadm_total:,.2f}")

            st.markdown("### 📅 Cronograma del caso seleccionado")
            st.dataframe(
                df2.style.format({
                    "Saldo Inicial": "{:,.2f}",
                    "Interés": "{:,.2f}",
                    "Amortización": "{:,.2f}",
                    "Cuota": "{:,.2f}",
                    "Seguro": "{:,.2f}",
                    "Gasto Adm": "{:,.2f}",
                    "Cuota Total": "{:,.2f}",
                    "Saldo Final": "{:,.2f}",
                    "Flujo Cliente": "{:,.2f}",
                }),
                use_container_width=True
            )

            csv2 = df2.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Descargar cronograma (CSV)",
                csv2,
                file_name=f"cronograma_caso_{case_id}.csv",
                mime="text/csv"
            )

# ---------------------------------------------------------------------
# 5) 📦 Base de datos (viewer con clave, SIN pistas)
# ---------------------------------------------------------------------
with sec5:
    st.subheader("📦 Base de datos")

    topA, topB = st.columns([3, 1])
    with topB:
        if st.session_state["db_unlocked"] and st.button("🔒 Bloquear"):
            st.session_state["db_unlocked"] = False
            st.rerun()

    if not st.session_state["db_unlocked"]:
        key = st.text_input("Clave de acceso", type="password", key="db_key_input")
        if key:
            if key == DB_VIEW_KEY:
                st.session_state["db_unlocked"] = True
                st.success("Acceso concedido.")
                st.rerun()
            else:
                st.error("Acceso denegado.")
    else:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [r[0] for r in cur.fetchall()]

        if not tables:
            st.warning("No hay tablas en la base de datos.")
        else:
            colA, colB = st.columns([2, 1])
            with colA:
                default_idx = tables.index("clients") if "clients" in tables else 0
                table = st.selectbox("Tabla", tables, index=default_idx)
            with colB:
                limit = st.number_input("Máx. filas", min_value=10, max_value=5000, value=500, step=50)

            try:
                df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY 1 DESC LIMIT {int(limit)}", conn)
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    f"⬇️ Descargar {table}.csv",
                    df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{table}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"No se pudo leer la tabla {table}: {e}")

# ---------------------------------------------------------------------
# 🤖 Helpy flotante + globito emergente cada 10s (blanco)
# ---------------------------------------------------------------------
ctx = st.session_state.get("helpbot_ctx", None)
ctx_json = json.dumps(ctx, ensure_ascii=False) if ctx else "null"

HELPBOT_INJECT_TEMPLATE = r"""
<script>
(function () {
  try {
    const parentWin = window.parent;
    const doc = parentWin && parentWin.document;
    if (!doc) throw new Error("No parent document");

    const CTX = __CTX__;

    const old = doc.getElementById("mv-helpbot-root");
    if (old) old.remove();
    const styleOld = doc.getElementById("mv-helpbot-style");
    if (styleOld) styleOld.remove();

    const style = doc.createElement("style");
    style.id = "mv-helpbot-style";
    style.textContent = `
      #mv-helpbot-root * { box-sizing: border-box; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }

      /* FAB subido para no tapar “Manage app” */
      #mv-helpbot-fab {
        position: fixed; right: 18px; bottom: 84px; z-index: 2147483000;
        width: 56px; height: 56px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,.14);
        cursor: pointer;
        background: rgba(30,30,30,.55);
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0,0,0,.25);
        display: grid; place-items: center;
        user-select: none;
      }
      #mv-helpbot-fab:hover { transform: translateY(-1px); }

      /* Nudge (globito) - BLANCO */
      #mv-helpbot-nudge {
        position: fixed;
        right: 84px;
        bottom: 98px;
        z-index: 2147483000;
        max-width: 260px;
        background: rgba(255,255,255,.96);
        border: 1px solid rgba(0,0,0,.10);
        color: rgba(0,0,0,.88);
        padding: 10px 12px;
        border-radius: 14px;
        box-shadow: 0 12px 35px rgba(0,0,0,.28);
        backdrop-filter: blur(6px);
        font-size: 12.5px;
        line-height: 1.25;
        opacity: 0;
        transform: translateY(6px);
        transition: opacity .22s ease, transform .22s ease;
        display: none;
        pointer-events: none;
      }
      #mv-helpbot-nudge.show {
        opacity: 1;
        transform: translateY(0);
      }

      /* Panel también subido */
      #mv-helpbot-panel {
        position: fixed; right: 18px; bottom: 152px; z-index: 2147483000;
        width: 360px; max-width: calc(100vw - 36px);
        height: 500px; max-height: calc(100vh - 120px);
        border-radius: 16px;
        overflow: hidden;
        background: rgba(20,20,20,.82);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 16px 45px rgba(0,0,0,.35);
        display: none;
        flex-direction: column;
      }

      #mv-helpbot-header {
        padding: 12px 12px;
        display:flex; align-items:center; justify-content:space-between;
        background: rgba(255,255,255,.06);
        border-bottom: 1px solid rgba(255,255,255,.10);
        color: rgba(255,255,255,.92);
        flex: 0 0 auto;
      }
      #mv-helpbot-title { font-weight: 700; font-size: 14px; display:flex; gap:8px; align-items:center; }
      #mv-helpbot-badge { font-size: 11px; padding: 2px 8px; border-radius: 999px; background: rgba(255,255,255,.10); }

      #mv-helpbot-close {
        border:none; background: transparent; color: rgba(255,255,255,.7);
        cursor:pointer; font-size: 16px; line-height: 1;
      }

      #mv-helpbot-chips {
        padding: 10px 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        border-bottom: 1px solid rgba(255,255,255,.10);
        background: rgba(255,255,255,.04);
        flex: 0 0 auto;
      }
      .mv-chip {
        border: 1px solid rgba(255,255,255,.12);
        background: rgba(255,255,255,.08);
        color: rgba(255,255,255,.92);
        padding: 7px 10px;
        border-radius: 999px;
        cursor: pointer;
        font-size: 12px;
        user-select: none;
      }
      .mv-chip:hover { background: rgba(255,255,255,.12); }

      #mv-helpbot-body {
        padding: 12px;
        overflow-y: auto;
        color: rgba(255,255,255,.90);
        flex: 1 1 auto;
      }
      .mv-msg { margin: 0 0 10px 0; display:flex; gap:10px; align-items:flex-start; }
      .mv-bubble {
        padding: 10px 12px; border-radius: 14px;
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.10);
        font-size: 13px; line-height: 1.35;
        white-space: pre-wrap;
      }
      .mv-me .mv-bubble { background: rgba(79,70,229,.22); border-color: rgba(79,70,229,.35); }
      .mv-avatar {
        width: 26px; height: 26px; border-radius: 8px;
        display:grid; place-items:center;
        background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.10);
        flex: 0 0 26px;
        font-size: 14px;
      }

      .mv-context {
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(16,185,129,.10);
        border: 1px solid rgba(16,185,129,.20);
        margin: 0 0 10px 0;
        font-size: 12px;
        color: rgba(255,255,255,.92);
      }

      #mv-helpbot-footer {
        padding: 10px 10px;
        background: rgba(255,255,255,.06);
        border-top: 1px solid rgba(255,255,255,.10);
        display:flex; gap:8px; align-items:center;
        flex: 0 0 auto;
      }
      #mv-helpbot-input {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,.14);
        background: rgba(0,0,0,.20);
        color: rgba(255,255,255,.92);
        padding: 10px 12px;
        outline: none;
        font-size: 13px;
      }
      #mv-helpbot-input::placeholder { color: rgba(255,255,255,.45); }
      #mv-helpbot-send {
        border:none; cursor:pointer;
        border-radius: 12px;
        padding: 10px 12px;
        background: rgba(255,255,255,.12);
        color: rgba(255,255,255,.92);
      }
      #mv-helpbot-send:hover { background: rgba(255,255,255,.16); }
    `;
    doc.head.appendChild(style);

    const root = doc.createElement("div");
    root.id = "mv-helpbot-root";
    root.innerHTML = `
      <div id="mv-helpbot-fab" aria-label="Abrir Helpy">🤖</div>
      <div id="mv-helpbot-nudge"></div>

      <div id="mv-helpbot-panel" role="dialog" aria-label="Helpy">
        <div id="mv-helpbot-header">
          <div id="mv-helpbot-title">🤖 Helpy <span id="mv-helpbot-badge">Ayuda</span></div>
          <button id="mv-helpbot-close" aria-label="Cerrar">✕</button>
        </div>

        <div id="mv-helpbot-chips"></div>
        <div id="mv-helpbot-body"></div>

        <div id="mv-helpbot-footer">
          <input id="mv-helpbot-input" placeholder="Escribe tu pregunta (ej: ¿cómo guardo un caso?)" />
          <button id="mv-helpbot-send">Enviar</button>
        </div>
      </div>
    `;
    doc.body.appendChild(root);

    const STORAGE_KEY = "mivivienda_helpy_msgs_v1";

    const fab = doc.getElementById("mv-helpbot-fab");
    const nudge = doc.getElementById("mv-helpbot-nudge");
    const panel = doc.getElementById("mv-helpbot-panel");
    const closeBtn = doc.getElementById("mv-helpbot-close");
    const chips = doc.getElementById("mv-helpbot-chips");
    const body = doc.getElementById("mv-helpbot-body");
    const input = doc.getElementById("mv-helpbot-input");
    const send = doc.getElementById("mv-helpbot-send");

    const QUICK = [
      { label: "👤 Crear cliente", q: "¿Cómo creo un cliente?" },
      { label: "📅 Cronograma", q: "¿Cómo genero el cronograma?" },
      { label: "💾 Guardar caso", q: "¿Cómo guardo un caso?" },
      { label: "📈 TCEA vs TREA", q: "Explícame por qué TCEA y TREA salen diferentes" },
      { label: "⬇️ Descargar CSV", q: "¿Cómo descargo el CSV?" }
    ];

    const NUDGES = [
      "👋 ¿Necesitas ayuda? Soy Helpy 🤖",
      "Tip: genera el cronograma y luego guarda el caso 💾",
      "¿No entiendes TCEA/TREA? Pregúntame y te lo explico fácil 🙂",
      "¿Quieres exportar? Tienes botón para descargar CSV ⬇️",
    ];

    let nudgeIdx = 0;
    let nudgeTimer = null;

    function money(sym, x){
      const n = Number(x);
      if (!isFinite(n)) return `${sym} -`;
      return `${sym} ` + n.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
    }
    function pct(x){
      const n = Number(x);
      if (!isFinite(n)) return "-";
      return (n*100).toFixed(3) + "%";
    }

    function showNudge(){
      if (panel.style.display === "flex") return;
      const msg = NUDGES[nudgeIdx % NUDGES.length];
      nudgeIdx++;

      nudge.textContent = msg;
      nudge.style.display = "block";
      void nudge.offsetHeight;
      nudge.classList.add("show");

      setTimeout(() => {
        nudge.classList.remove("show");
      }, 2200);

      setTimeout(() => {
        if (!nudge.classList.contains("show")) nudge.style.display = "none";
      }, 2600);
    }

    function startNudges(){
      if (nudgeTimer) return;
      setTimeout(showNudge, 2000);
      nudgeTimer = setInterval(showNudge, 10000); // cada 10s
    }

    function stopNudges(){
      if (!nudgeTimer) return;
      clearInterval(nudgeTimer);
      nudgeTimer = null;
      nudge.classList.remove("show");
      nudge.style.display = "none";
    }

    function addMsg(role, text){
      const wrap = doc.createElement("div");
      wrap.className = "mv-msg " + (role === "me" ? "mv-me" : "mv-bot");
      const av = doc.createElement("div");
      av.className = "mv-avatar";
      av.textContent = (role === "me" ? "🧑" : "🤖");
      const bub = doc.createElement("div");
      bub.className = "mv-bubble";
      bub.textContent = text || "";
      wrap.appendChild(av);
      wrap.appendChild(bub);
      body.appendChild(wrap);
      body.scrollTop = body.scrollHeight;
      persist();
    }

    function renderChips(){
      chips.innerHTML = "";
      QUICK.forEach(item => {
        const b = doc.createElement("div");
        b.className = "mv-chip";
        b.textContent = item.label;
        b.addEventListener("click", () => {
          addMsg("me", item.q);
          addMsg("bot", respond(item.q));
        });
        chips.appendChild(b);
      });
    }

    function addContextCard(){
      if (!CTX) return;
      const div = doc.createElement("div");
      div.className = "mv-context";
      const sym = CTX.symbol || "";
      const tcea = CTX.tcea != null ? pct(CTX.tcea) : "-";
      const trea = CTX.trea_credit != null ? pct(CTX.trea_credit) : "-";
      div.textContent =
        `Caso activo #${CTX.case_id}: ${CTX.case_name} | Préstamo: ${money(sym, CTX.principal)} | TCEA: ${tcea} | TREA(crédito): ${trea}`;
      body.appendChild(div);
    }

    function persist(){
      const msgs = [];
      body.querySelectorAll(".mv-msg").forEach(m => {
        const role = m.classList.contains("mv-me") ? "me" : "bot";
        const text = (m.querySelector(".mv-bubble")?.innerText) || "";
        msgs.push({role, text});
      });
      try { parentWin.localStorage.setItem(STORAGE_KEY, JSON.stringify(msgs)); } catch(e){}
    }

    function load(){
      body.innerHTML = "";
      addContextCard();

      let msgs = null;
      try { msgs = JSON.parse(parentWin.localStorage.getItem(STORAGE_KEY)); } catch(e){}
      if (!Array.isArray(msgs) || msgs.length === 0){
        addMsg("bot",
`Hola 👋 Soy Helpy 🤖
Puedo responder preguntas sobre cómo usar la plataforma.`);
      } else {
        msgs.forEach(m => addMsg(m.role, m.text));
      }
    }

    function explainTceaTreaWithNumbers(){
      if (!CTX) {
        return "Aún no tengo números de un caso. Ve a 'Casos & KPIs', selecciona un caso y vuelve aquí 🙂";
      }
      const sym = CTX.symbol || "";
      const p = CTX.principal;
      const fee = CTX.fee_opening;
      const segM = CTX.monthly_insurance;
      const admM = CTX.monthly_admin_fee;
      const cuota = CTX.cuota;
      const cuotaTot = CTX.cuota_total;

      const tcea = CTX.tcea != null ? pct(CTX.tcea) : "-";
      const treaC = CTX.trea_credit != null ? pct(CTX.trea_credit) : "-";
      const treaT = CTX.trea_total != null ? pct(CTX.trea_total) : "-";

      return (
`No son lo mismo porque miran flujos distintos:

1) TCEA (cliente) = “cuánto te cuesta a ti”.
- En t0 tú recibes: préstamo - comisión apertura
- Luego tú pagas: Cuota Total = cuota + desgravamen + gasto adm
  Ejemplo: ${money(sym, cuota)} + ${money(sym, segM)} + ${money(sym, admM)} = ${money(sym, cuotaTot)}
➡️ Por eso sale TCEA = ${tcea}

2) TREA (entidad) = “cuánto gana la entidad”.
- TREA (crédito): solo ‘Cuota’ del préstamo → ${treaC}
- TREA (total cobros): si metes también cargos → ${treaT}`
      );
    }

    function respond(qRaw){
      const q = (qRaw || "").toLowerCase().trim();
      if (!q) return "Escribe tu pregunta y te ayudo 🙂";

      if (q.includes("tcea") && q.includes("trea") && (q.includes("difer") || q.includes("por que") || q.includes("por qué") || q.includes("explica"))) {
        return explainTceaTreaWithNumbers();
      }
      if (q.includes("tcea") || q.includes("trea") || q.includes("tir") || q.includes("tirm")) {
        return "TCEA = costo anual efectivo para el cliente. TREA = rentabilidad anual efectiva para la entidad. Si quieres, dime: “Explícame por qué TCEA y TREA salen diferentes”.";
      }

      if (q.includes("login") || q.includes("iniciar") || q.includes("sesion") || q.includes("sesión") || q.includes("contraseña") || q.includes("usuario")) {
        return "Para entrar: usa el panel izquierdo (Acceso). Si no tienes cuenta: 'Registrarse'. Demo: admin/admin.";
      }

      if (q.includes("cliente") || q.includes("dni") || q.includes("documento") || q.includes("correo") || q.includes("email") || q.includes("telefono") || q.includes("teléfono")) {
        return "Clientes: pestaña 'Cliente'. Completa Documento, Nombre, Ingreso, Dependientes, Teléfono (9 dígitos) y Email. Luego 'Guardar cliente'. Para editar: usa el selector de arriba.";
      }

      if (q.includes("cronograma") || q.includes("cuota") || q.includes("prestamo") || q.includes("préstamo") || q.includes("gracia") || q.includes("tea") || q.includes("tem") || q.includes("entidad") || q.includes("cuota inicial")) {
        return "Cronograma: pestaña 'Configurar Préstamo'. Pon valor del inmueble, cuota inicial, bono y plazo. Elige entidad (TEA ya está). Define gracia y gastos y presiona 'Generar cronograma'.";
      }

      if (q.includes("guardar caso") || (q.includes("guardar") && q.includes("caso"))) {
        return "Guardar caso: pestaña 'Guardar caso'. Primero debes generar el cronograma. Luego eliges cliente, pones nombre del caso y clic en 'Guardar caso en base de datos'.";
      }

      if (q.includes("borrar") || q.includes("eliminar")) {
        return "Borrar: en 'Casos & KPIs' puedes borrar un caso. En 'Cliente' puedes borrar un cliente (pide confirmación y si tiene casos, puede borrar también esos casos).";
      }

      if (q.includes("csv") || q.includes("descargar") || q.includes("exportar")) {
        return "Descargar: en 'Casos & KPIs' baja al cronograma y usa 'Descargar cronograma (CSV)'.";
      }

      if (q.includes("base de datos") || q.includes("bd") || q.includes("sqlite") || q.includes("tabla")) {
        return "En la pestaña correspondiente puedes acceder si tienes la clave de acceso asignada por el administrador.";
      }

      return "Te ayudo con: clientes, cronograma, guardar casos, TCEA/TREA y exportación CSV. Escribe tu duda.";
    }

    function sendMsg(){
      const q = input.value || "";
      input.value = "";
      addMsg("me", q);
      addMsg("bot", respond(q));
    }

    startNudges();

    fab.addEventListener("click", () => {
      const isOpen = panel.style.display === "flex";
      panel.style.display = isOpen ? "none" : "flex";
      if (!isOpen){
        stopNudges();
        renderChips();
        load();
        setTimeout(() => input.focus(), 50);
      } else {
        startNudges();
      }
    });

    closeBtn.addEventListener("click", () => {
      panel.style.display = "none";
      startNudges();
    });

    send.addEventListener("click", sendMsg);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMsg(); });

  } catch (e) {
    console.warn("Helpy inject failed:", e);
  }
})();
</script>
"""

HELPBOT_INJECT = HELPBOT_INJECT_TEMPLATE.replace("__CTX__", ctx_json)
components.html(HELPBOT_INJECT, height=1)
