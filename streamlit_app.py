# streamlit_app.py
# ============================================================
# App Streamlit – Passos Mágicos (Datathon Fase 5)
# Gera índices: IAN, IDA, IEG, IAA, IPS, IPP, IPV (informado ou estimado)
# Opcional: modelo preditivo de risco (probabilidade)
# ============================================================

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ----------------------------
# Config / UI helpers
# ----------------------------

APP_TITLE = "Passos Mágicos – Calculadora de Índices (IAN/IDA/IEG/IAA/IPS/IPP/IPV)"
APP_ICON = "📊"

DEFAULT_DATA_PATHS = [
    "Base Tratada.xlsx",
    "./Base Tratada.xlsx",
    "./data/Base Tratada.xlsx",
    "./BASES/Base Tratada.xlsx",
]

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _fmt_score(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.2f}"


# ----------------------------
# Índices – regras encontradas na base
# ----------------------------

def calcular_defasagem(fase_atual: float, fase_ideal: float) -> float:
    """Defasagem = fase_atual - fase_ideal (negativo => abaixo do ideal)."""
    return _safe_float(fase_atual) - _safe_float(fase_ideal)


def calcular_IAN(defasagem: float) -> float:
    """
    Regra observada na base:
    - defasagem == 0 ou positiva => IAN = 10
    - defasagem -1 ou -2 => IAN = 5
    - defasagem <= -3 => IAN = 2.5
    """
    d = _safe_float(defasagem)
    if np.isnan(d):
        return np.nan
    if d >= 0:
        return 10.0
    if d in (-1, -2):
        return 5.0
    if d <= -3:
        return 2.5
    # fallback (caso apareça -0.5 etc.)
    if d > -1:
        return 10.0
    if d > -3:
        return 5.0
    return 2.5


def calcular_IDA(nota_mat: float, nota_por: float, nota_ing: float) -> float:
    """Regra observada na base: IDA = média(Mat, Por, Ing)."""
    m = _safe_float(nota_mat)
    p = _safe_float(nota_por)
    i = _safe_float(nota_ing)
    vals = [v for v in (m, p, i) if not np.isnan(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


# ----------------------------
# Data loading / models
# ----------------------------

@st.cache_data(show_spinner=False)
def load_base_tratada(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Base Tratada", engine="openpyxl")
    return df


def find_data_path() -> Optional[str]:
    for p in DEFAULT_DATA_PATHS:
        if os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner=False)
def train_ipv_regressor(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    IPV não é trivial de pedir ao usuário leigo.
    Então, treinamos um modelo regressivo (Ridge) que estima IPV a partir
    dos demais indicadores e notas, usando os registros históricos.
    """
    needed = ["IPV", "IEG", "IDA", "IAA", "IPS", "IPP", "IAN", "Atingiu PV"]
    available = [c for c in needed if c in df.columns]
    df2 = df[available].copy()

    for c in ["IPV", "IEG", "IDA", "IAA", "IPS", "IPP", "IAN", "Atingiu PV"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    df2 = df2.dropna(subset=["IPV"])
    if len(df2) < 200:
        model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("ridge", Ridge(alpha=1.0)),
        ])
        X = df2.drop(columns=["IPV"])
        y = (df2["IPV"] * 10.0).values
        model.fit(X, y)
        return model, {"r2_train": np.nan, "n_train": len(df2)}

    X = df2.drop(columns=["IPV"])
    y = (df2["IPV"] * 10.0).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("ridge", Ridge(alpha=2.0, random_state=42)),
    ])
    model.fit(X_train, y_train)

    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    meta = {
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(X.columns),
    }
    return model, meta


@st.cache_resource(show_spinner=False)
def train_risk_classifier(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Modelo opcional de risco: segue a ideia do notebook:
    cria alvo 'em_risco' e treina classificador.
    """
    dfm = df.copy()

    dfm["em_risco"] = 0

    if "Defasagem" in dfm.columns:
        d = pd.to_numeric(dfm["Defasagem"], errors="coerce")
        dfm["em_risco"] = np.where(d.notna() & (d < 0), 1, dfm["em_risco"])

    if ("Fase" in dfm.columns) and ("Fase Ideal" in dfm.columns):
        f = pd.to_numeric(dfm["Fase"], errors="coerce")
        fi = pd.to_numeric(dfm["Fase Ideal"], errors="coerce")
        dfm["em_risco"] = np.where((f.notna() & fi.notna() & (f < fi)), 1, dfm["em_risco"])

    if "INDE atual" in dfm.columns:
        inde = pd.to_numeric(dfm["INDE atual"], errors="coerce")
        p25 = inde.quantile(0.25)
        dfm["em_risco"] = np.where((inde.notna() & (inde < p25)), 1, dfm["em_risco"])

    num_candidates = ["IEG", "IDA", "IPS", "IPP", "IAA", "IAN", "Mat", "Por", "Ing", "Idade", "Ano ingresso"]
    cat_candidates = ["Gênero", "Instituição de ensino", "Pedra atual", "Fase"]

    num_cols = [c for c in num_candidates if c in dfm.columns]
    cat_cols = [c for c in cat_candidates if c in dfm.columns]

    df_model = dfm[num_cols + cat_cols + ["em_risco"]].copy()

    for c in num_cols:
        df_model[c] = pd.to_numeric(df_model[c], errors="coerce")

    df_model = df_model.dropna(subset=["em_risco"])
    df_model = df_model.dropna(thresh=int(len(num_cols + cat_cols) * 0.5))

    X = df_model.drop(columns=["em_risco"])
    y = df_model["em_risco"].astype(int)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=2000, n_jobs=None)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    model.fit(X_train, y_train)

    meta = {"auc_test": np.nan, "n_train": len(X_train), "n_test": len(X_test)}
    if y_test.nunique() > 1:
        proba = model.predict_proba(X_test)[:, 1]
        meta["auc_test"] = float(roc_auc_score(y_test, proba))

    meta["num_cols"] = num_cols
    meta["cat_cols"] = cat_cols
    return model, meta


# ----------------------------
# DTO (registro do aluno)
# ----------------------------

@dataclass
class RegistroAluno:
    nome: str
    idade: float
    genero: str
    instituicao: str
    ano_ingresso: float
    fase_atual: float
    fase_ideal: float

    nota_mat: float
    nota_por: float
    nota_ing: float

    IEG: float
    IAA: float
    IPS: float
    IPP: float

    atingiu_pv: int  # 0/1

    # calculados
    Defasagem: float
    IAN: float
    IDA: float
    IPV: float  # 0-10 (informado ou estimado)


# ----------------------------
# UI
# ----------------------------

st.title(f"{APP_ICON} {APP_TITLE}")
st.caption(
    "Preencha os campos abaixo para gerar o relatório individual com os índices. "
    "O app foi desenhado para ser usado por pessoas leigas: basta seguir as etapas."
)

data_path = find_data_path()
if data_path is None:
    st.error(
        "Não encontrei o arquivo **Base Tratada.xlsx** no repositório.

"
        "Coloque o arquivo na raiz do projeto (mesma pasta do `streamlit_app.py`) "
        "ou em `./data/`."
    )
    st.stop()

df_base = load_base_tratada(data_path)

with st.expander("📌 Diagnóstico rápido da base (somente leitura)", expanded=False):
    st.write(f"Arquivo carregado: `{data_path}`")
    st.write("Dimensão da base:", df_base.shape)
    st.write("Colunas disponíveis (amostra):", list(df_base.columns)[:25])

# Treinar modelos
ipv_model, ipv_meta = train_ipv_regressor(df_base)
risk_model, risk_meta = train_risk_classifier(df_base)

# Sidebar
st.sidebar.header("⚙️ Configurações")
modo_ipv = st.sidebar.radio(
    "Como calcular o IPV?",
    options=[
        "Estimar automaticamente (recomendado)",
        "Informar manualmente"
    ],
    index=0
)

mostrar_risco = st.sidebar.checkbox("Também calcular probabilidade de risco (modelo)", value=True)

with st.sidebar.expander("Qualidade dos modelos", expanded=False):
    st.write("**Estimador de IPV (Ridge)**")
    st.json(ipv_meta)
    st.write("**Classificador de risco (LogReg)**")
    st.json(risk_meta)

# Histórico
if "historico" not in st.session_state:
    st.session_state["historico"] = []

st.divider()
st.subheader("🧭 Etapa 1 — Identificação e fase")

col1, col2, col3 = st.columns(3)

with col1:
    nome = st.text_input("Nome (ou apelido) do aluno", value="", placeholder="Ex.: Aluno 001")
    idade = st.number_input("Idade (anos)", min_value=5, max_value=30, value=12, step=1)
    genero = st.selectbox("Gênero", options=["Feminino", "Masculino", "Outro/Prefiro não informar", "Desconhecido"])

with col2:
    instituicao = st.text_input("Instituição de ensino", value="", placeholder="Ex.: Escola Municipal X")
    ano_ingresso = st.number_input("Ano de ingresso", min_value=1990, max_value=2035, value=2024, step=1)
    fase_atual = st.number_input("Fase atual (número)", min_value=0.0, max_value=20.0, value=6.0, step=1.0)

with col3:
    fase_ideal = st.number_input("Fase ideal (número)", min_value=0.0, max_value=20.0, value=7.0, step=1.0)
    atingiu_pv = st.selectbox("Atingiu o Ponto de Virada (PV)?", options=["Não", "Sim"], index=0)
    atingiu_pv_int = 1 if atingiu_pv == "Sim" else 0

defasagem = calcular_defasagem(fase_atual, fase_ideal)
ian = calcular_IAN(defasagem)

st.info(f"**Defasagem calculada** = Fase atual − Fase ideal = **{defasagem:.0f}**  |  **IAN** = **{ian:.2f}**")

st.subheader("🧮 Etapa 2 — Notas (IDA)")

c1, c2, c3 = st.columns(3)
with c1:
    nota_mat = st.number_input("Matemática (0 a 10)", min_value=0.0, max_value=10.0, value=6.0, step=0.5)
with c2:
    nota_por = st.number_input("Português (0 a 10)", min_value=0.0, max_value=10.0, value=6.0, step=0.5)
with c3:
    nota_ing = st.number_input("Inglês (0 a 10)", min_value=0.0, max_value=10.0, value=6.0, step=0.5)

ida = calcular_IDA(nota_mat, nota_por, nota_ing)
st.success(f"**IDA (média das notas)** = **{ida:.2f}**")

st.subheader("🧠 Etapa 3 — Questionários (IEG / IAA / IPS / IPP)")

st.caption("Use as barras (0 a 10). A ideia é ser intuitivo: 0 = muito ruim / 10 = excelente.")

q1, q2 = st.columns(2)
with q1:
    ieg = st.slider("IEG – Engajamento nas atividades", 0.0, 10.0, 7.0, 0.1)
    iaa = st.slider("IAA – Autoavaliação", 0.0, 10.0, 8.0, 0.1)
with q2:
    ips = st.slider("IPS – Aspectos psicossociais", 0.0, 10.0, 7.0, 0.1)
    ipp = st.slider("IPP – Aspectos psicopedagógicos", 0.0, 10.0, 7.5, 0.1)

st.subheader("🔁 Etapa 4 — IPV (Ponto de Virada)")

if modo_ipv == "Informar manualmente":
    ipv_final = float(st.slider("IPV – Ponto de Virada (0 a 10)", 0.0, 10.0, 7.0, 0.01))
    st.warning("Você escolheu informar o IPV manualmente.")
else:
    X_ipv = pd.DataFrame([{
        "IEG": ieg,
        "IDA": ida,
        "IAA": iaa,
        "IPS": ips,
        "IPP": ipp,
        "IAN": ian,
        "Atingiu PV": atingiu_pv_int,
    }])
    if "features" in ipv_meta:
        X_ipv = X_ipv[ipv_meta["features"]]
    ipv_estimado = float(ipv_model.predict(X_ipv)[0])
    ipv_final = _clamp(ipv_estimado, 0.0, 10.0)

    st.write(
        f"**IPV estimado automaticamente**: **{ipv_final:.2f}** "
        f"(modelo treinado na base histórica; qualidade aproximada indicada na sidebar)"
    )

st.divider()
st.subheader("📄 Relatório final do aluno")

registro = RegistroAluno(
    nome=nome.strip() if nome.strip() else "Aluno (sem nome)",
    idade=float(idade),
    genero=str(genero),
    instituicao=instituicao.strip() if instituicao.strip() else "Não informado",
    ano_ingresso=float(ano_ingresso),
    fase_atual=float(fase_atual),
    fase_ideal=float(fase_ideal),
    nota_mat=float(nota_mat),
    nota_por=float(nota_por),
    nota_ing=float(nota_ing),
    IEG=float(ieg),
    IAA=float(iaa),
    IPS=float(ips),
    IPP=float(ipp),
    atingiu_pv=int(atingiu_pv_int),
    Defasagem=float(defasagem),
    IAN=float(ian),
    IDA=float(ida),
    IPV=float(ipv_final),
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("IAN", _fmt_score(registro.IAN), help="Adequação do nível (derivada da defasagem)")
m2.metric("IDA", _fmt_score(registro.IDA), help="Desempenho acadêmico (média de Mat/Por/Ing)")
m3.metric("IEG", _fmt_score(registro.IEG), help="Engajamento nas atividades")
m4.metric("IPV", _fmt_score(registro.IPV), help="Ponto de virada (manual ou estimado)")

m5, m6, m7 = st.columns(3)
m5.metric("IAA", _fmt_score(registro.IAA))
m6.metric("IPS", _fmt_score(registro.IPS))
m7.metric("IPP", _fmt_score(registro.IPP))

st.caption("Visão rápida (0 a 10):")
df_plot = pd.DataFrame([{
    "IAN": registro.IAN,
    "IDA": registro.IDA,
    "IEG": registro.IEG,
    "IAA": registro.IAA,
    "IPS": registro.IPS,
    "IPP": registro.IPP,
    "IPV": registro.IPV,
}]).T.rename(columns={0: "score"})
st.bar_chart(df_plot)

if mostrar_risco:
    st.subheader("🧯 Probabilidade de risco (opcional)")
    row: Dict[str, Any] = {}

    for c in risk_meta.get("num_cols", []):
        if c == "IEG":
            row[c] = registro.IEG
        elif c == "IDA":
            row[c] = registro.IDA
        elif c == "IPS":
            row[c] = registro.IPS
        elif c == "IPP":
            row[c] = registro.IPP
        elif c == "IAA":
            row[c] = registro.IAA
        elif c == "IAN":
            row[c] = registro.IAN
        elif c == "Mat":
            row[c] = registro.nota_mat
        elif c == "Por":
            row[c] = registro.nota_por
        elif c == "Ing":
            row[c] = registro.nota_ing
        elif c == "Idade":
            row[c] = registro.idade
        elif c == "Ano ingresso":
            row[c] = registro.ano_ingresso
        else:
            row[c] = np.nan

    for c in risk_meta.get("cat_cols", []):
        if c == "Gênero":
            row[c] = registro.genero
        elif c == "Instituição de ensino":
            row[c] = registro.instituicao
        elif c == "Fase":
            row[c] = str(int(registro.fase_atual)) if float(registro.fase_atual).is_integer() else str(registro.fase_atual)
        elif c == "Pedra atual":
            row[c] = "Desconhecido"
        else:
            row[c] = "Desconhecido"

    X_risk = pd.DataFrame([row])
    try:
        proba = float(risk_model.predict_proba(X_risk)[0, 1])
        st.metric("Probabilidade estimada de risco", f"{proba*100:.1f}%")
        if proba >= 0.70:
            st.error("Risco alto (>=70%). Sugere atenção prioritária.")
        elif proba >= 0.40:
            st.warning("Risco moderado (40–69%). Sugere acompanhamento.")
        else:
            st.success("Risco baixo (<40%). Manter monitoramento.")
    except Exception as e:
        st.warning("Não foi possível calcular o risco com o modelo atual.")
        st.exception(e)

st.subheader("💾 Salvar / Exportar")

cA, cB = st.columns([1, 2])
with cA:
    if st.button("➕ Adicionar ao histórico desta sessão"):
        st.session_state["historico"].append(asdict(registro))
        st.success("Registro adicionado ao histórico da sessão.")

with cB:
    st.caption("Você pode baixar o relatório em JSON/CSV para juntar vários alunos.")
    registro_json = asdict(registro)
    st.download_button(
        "⬇️ Baixar relatório (JSON)",
        data=pd.Series(registro_json).to_json(orient="index", force_ascii=False),
        file_name=f"relatorio_{registro.nome.replace(' ', '_')}.json",
        mime="application/json",
    )

if len(st.session_state["historico"]) > 0:
    st.subheader("📚 Histórico (sessão atual)")
    df_hist = pd.DataFrame(st.session_state["historico"])
    st.dataframe(df_hist, use_container_width=True)

    st.download_button(
        "⬇️ Baixar histórico (CSV)",
        data=df_hist.to_csv(index=False).encode("utf-8"),
        file_name="historico_indices.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Notas técnicas: IDA e IAN foram reproduzidos por regras observadas diretamente na base. "
    "IPV estimado é um ajuste por modelo treinado no histórico para facilitar uso por pessoas leigas."
)
