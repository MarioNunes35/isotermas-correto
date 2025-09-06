import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import zscore

st.set_page_config(page_title="Isotermas de Adsorção - Análise Robusta", layout="wide")
st.title("Isotermas de Adsorção com Tratamento de Outliers")
st.write("Aplicativo aprimorado com **ajuste ponderado apenas por σ(qe)** e **detecção de outliers**.")

def langmuir(C, qmax, KL):
    return (qmax * KL * C) / (1.0 + KL * C)

def freundlich(C, KF, n):
    n = np.maximum(n, 1e-8)
    return KF * np.power(C, 1.0/n)

def temkin(C, B, A):
    return B * np.log(np.maximum(A*C, 1e-12))

def sips(C, qmax, Ks, n):
    """Modelo de Sips: qe = qmax * (Ks*C)^n / (1 + (Ks*C)^n)"""
    x = np.power(np.maximum(Ks*C, 1e-12), np.maximum(n, 1e-8))
    return qmax * x / (1.0 + x)

def redlich_peterson(C, KR, a, g):
    g = np.clip(g, 1e-8, 1.0)
    return (KR * C) / (1.0 + a * np.power(C, g))

def hill_langmuir(C, qmax, K, n):
    """Modelo de Hill-Langmuir: pode ter comportamento sigmoidal com pico"""
    n = np.maximum(n, 0.1)
    Cn = np.power(C, n)
    return (qmax * Cn) / (K + Cn)

def competitive_langmuir(C, qmax1, K1, qmax2, K2):
    """Langmuir competitivo: dois sítios com afinidades diferentes"""
    return (qmax1 * K1 * C) / (1 + K1 * C) + (qmax2 * K2 * C) / (1 + K2 * C)

def toth_modified(C, qmax, KT, n, alpha):
    """Tóth modificado com termo de inibição"""
    alpha = np.clip(alpha, 0.01, 2.0)
    denominator = (1 + (KT * C)**n)**(1/n) + alpha * C
    return (qmax * KT * C) / denominator

def khan_model(C, qmax, K, beta):
    """Modelo de Khan: adsorção com inibição em altas concentrações"""
    beta = np.maximum(beta, 0.01)
    return (qmax * K * C) / (1 + K * C + beta * C**2)

def dual_site_langmuir(C, qmax1, K1, qmax2, K2, alpha):
    """Langmuir de dois sítios com interação"""
    alpha = np.clip(alpha, -0.5, 0.5)
    site1 = (qmax1 * K1 * C) / (1 + K1 * C)
    site2 = (qmax2 * K2 * C) / (1 + K2 * C)
    interaction = alpha * site1 * site2 / qmax1 if qmax1 > 0 else 0
    return site1 + site2 + interaction

MODEL_SPECS = {
    "Langmuir": {
        "func": langmuir, 
        "p0": lambda C,q: [np.nanmax(q)*1.2 if np.all(np.isfinite(q)) else 1.0, 1.0/max(np.nanmean(C),1e-6)],
        "bounds": ([0,0],[200, np.inf]),  # Increased from 120 to 200 mg/g
        "params": ["qmax (mg/g)","KL (L/mg)"],
        "equation": "qe = (qmax × KL × Ce) / (1 + KL × Ce)",
        "category": "Clássico"
    },
    "Freundlich": {
        "func": freundlich, 
        "p0": lambda C,q: [max(np.nanmin(q),1e-3), 2.0],
        "bounds": ([0,0.1],[np.inf,np.inf]),  # n >= 0.1 (mais flexível que n >= 1)
        "params": ["KF ((mg/g)(L/mg)^(1/n))","n (-)"],
        "equation": "qe = KF × Ce^(1/n)",
        "category": "Clássico"
    },
    "Temkin": {
        "func": temkin, 
        "p0": lambda C,q: [(np.nanmax(q)-np.nanmin(q))/max(np.log(np.nanmax(C))-np.log(np.nanmin(C)),1e-6),
                           np.exp(np.nanmin(q)/max((np.nanmax(q)-np.nanmin(q)),1e-6))/max(np.nanmin(C),1e-6)],
        "bounds": ([0,0],[np.inf,np.inf]), 
        "params": ["B (mg/g)","A (L/mg)"],
        "equation": "qe = B × ln(A × Ce)",
        "category": "Clássico"
    },
    "Sips": {
        "func": sips, 
        "p0": lambda C,q: [np.nanmax(q)*1.2, 1.0/max(np.nanmean(C),1e-6), 1.0],
        "bounds": ([0,0,0.1],[200, np.inf, 5.0]),  # Increased from 120 to 200 mg/g
        "params": ["qmax (mg/g)","Ks (L/mg)","n (-)"],
        "equation": "qe = qmax × (Ks×Ce)^n / (1 + (Ks×Ce)^n)",
        "category": "Clássico"
    },
    "Redlich-Peterson": {
        "func": redlich_peterson, 
        "p0": lambda C,q: [max(np.nanmean(q)/max(np.nanmean(C),1e-6),1e-3),
                           1.0/max(np.nanmax(C),1e-6), 0.8],
        "bounds": ([0,0,0.1],[np.inf,np.inf,1.0]), 
        "params": ["KR (L/g)","a (L/mg)^g","g (-)"],
        "equation": "qe = (KR × Ce) / (1 + a × Ce^g)",
        "category": "Clássico"
    },
    # MODELOS AVANÇADOS PARA COMPORTAMENTO NÃO-MONOTÔNICO
    "Hill-Langmuir": {
        "func": hill_langmuir,
        "p0": lambda C,q: [np.nanmax(q)*0.95, np.nanmean(C), 1.5],
        "bounds": ([0,0,0.1],[110, 300, 5.0]),  # qmax ≤ 110 mg/g
        "params": ["qmax (mg/g)","K (mg/L)^n","n (-)"],
        "equation": "qe = qmax × Ce^n / (K + Ce^n)",
        "category": "Não-monotônico"
    },
    "Langmuir Competitivo": {
        "func": competitive_langmuir,
        "p0": lambda C,q: [np.nanmax(q)*0.5, 1.0/max(np.nanmean(C),1e-6), 
                          np.nanmax(q)*0.5, 0.1/max(np.nanmean(C),1e-6)],
        "bounds": ([0,0,0,0],[55, 1.0, 55, 1.0]),  # 55+55 = 110 mg/g total
        "params": ["qmax1 (mg/g)","K1 (L/mg)","qmax2 (mg/g)","K2 (L/mg)"],
        "equation": "qe = (qmax1×K1×Ce)/(1+K1×Ce) + (qmax2×K2×Ce)/(1+K2×Ce)",
        "category": "Não-monotônico"
    },
    "Tóth Modificado": {
        "func": toth_modified,
        "p0": lambda C,q: [np.nanmax(q)*0.95, 1.0/max(np.nanmean(C),1e-6), 1.0, 0.01],
        "bounds": ([0,0,0.1,0.001],[110, 1.0, 3.0, 0.1]),  # qmax ≤ 110 mg/g
        "params": ["qmax (mg/g)","KT (L/mg)","n (-)","α (inibição)"],
        "equation": "qe = qmax×KT×Ce / [(1+(KT×Ce)^n)^(1/n) + α×Ce]",
        "category": "Não-monotônico"
    },
    "Khan (Inibição)": {
        "func": khan_model,
        "p0": lambda C,q: [np.nanmax(q)*0.95, 1.0/max(np.nanmean(C),1e-6), 1e-6],
        "bounds": ([0,0,1e-8],[110, 1.0, 1e-3]),  # qmax ≤ 110 mg/g
        "params": ["qmax (mg/g)","K (L/mg)","β (inibição)"],
        "equation": "qe = (qmax×K×Ce) / (1 + K×Ce + β×Ce²)",
        "category": "Não-monotônico"
    },
    "Langmuir Duplo": {
        "func": dual_site_langmuir,
        "p0": lambda C,q: [np.nanmax(q)*0.5, 1.0/max(np.nanmean(C),1e-6),
                          np.nanmax(q)*0.5, 0.5/max(np.nanmean(C),1e-6), 0.1],
        "bounds": ([0,0,0,0,-0.5],[55, 1.0, 55, 1.0, 0.5]),  # 55+55 = 110 mg/g total
        "params": ["qmax1 (mg/g)","K1 (L/mg)","qmax2 (mg/g)","K2 (L/mg)","α (interação)"],
        "equation": "qe = Site1 + Site2 + α×(Site1×Site2/qmax1)",
        "category": "Não-monotônico"
    }
}

def detect_columns(df):
    cols = {c.lower(): c for c in df.columns}
    def find(cands):
        for cand in cands:
            for key, orig in cols.items():
                if cand in key.replace(" ",""):
                    return orig
        return None
    m = {}
    m["Ce"] = find(["ce(mg/l)","ce","c_e","ce_mg/l","ce(mg l)","ce(mg·l^-1)","ce(mg.l-1)"])
    m["qe"] = find(["qe(mg/g)","qe","q_e","qe_mg/g","qe(mg g)","qe(mg·g^-1)","qe(mg.g-1)"])
    m["sigma_Ce"] = find(["±ce","sdce","sigmace","s_ce","stdce","sd_ce","sigma_ce"])
    m["sigma_qe"] = find(["±qe","sdqe","sigmaqe","s_qe","stdqe","sd_qe","sigma_qe"])
    return m

def weighted_r2(y, yhat, w):
    ybar = np.sum(w*y)/np.sum(w)
    ss_res = np.sum(w*(y-yhat)**2)
    ss_tot = np.sum(w*(y-ybar)**2)
    return (1.0 - ss_res/ss_tot) if ss_tot>0 else np.nan, ss_res

def aic_bic(ss_res_w, n, k):
    aic = n*np.log(ss_res_w/max(n,1)) + 2*k
    bic = n*np.log(ss_res_w/max(n,1)) + k*np.log(max(n,1))
    return aic, bic

def adjust_bounds_to_data(model_name, bounds, Ce_data, qe_data, bounds_mode="Livres (sem limitação)"):
    """Ajusta bounds dinamicamente baseado nos dados experimentais e modo selecionado"""
    if bounds_mode == "Livres (sem limitação)":
        # Bounds totalmente livres - apenas limites técnicos mínimos necessários
        lower, upper = bounds
        return bounds  # Retorna bounds originais sem modificação
    
    qe_max = np.nanmax(qe_data) if len(qe_data) > 0 else 110
    Ce_max = np.nanmax(Ce_data) if len(Ce_data) > 0 else 200
    
    lower, upper = bounds
    upper = list(upper)
    
    if bounds_mode == "Conservadores (baseados nos dados)":
        # Bounds baseados nos dados experimentais - MAIS PERMISSIVOS
        if model_name in ["Langmuir", "Sips", "Hill-Langmuir", "Tóth Modificado", "Khan (Inibição)"]:
            # qmax limitado a 200% dos dados experimentais (era 150%)
            qmax_limit = qe_max * 2.0  # INCREASED FROM 1.5 to 2.0
            # Ensure minimum bound is not below experimental maximum + 10%
            qmax_minimum = qe_max * 1.1
            qmax_limit = max(qmax_limit, qmax_minimum)
            
            if len(upper) > 0 and upper[0] != np.inf:
                upper[0] = max(min(upper[0], qmax_limit), qmax_minimum)  # FIXED: ensure not below experimental max
            elif len(upper) > 0:
                upper[0] = qmax_limit
                
        elif model_name in ["Langmuir Competitivo", "Langmuir Duplo"]:
            # Cada sítio limitado a ~100% dos dados experimentais (era 75%)
            site_limit = qe_max * 1.0  # INCREASED FROM 0.75 to 1.0
            site_minimum = qe_max * 0.6  # Minimum 60% of experimental max per site
            site_limit = max(site_limit, site_minimum)
            
            if len(upper) > 0:
                upper[0] = max(min(upper[0], site_limit) if upper[0] != np.inf else site_limit, site_minimum)
            if len(upper) >= 3:
                upper[2] = max(min(upper[2], site_limit) if upper[2] != np.inf else site_limit, site_minimum)
                
    elif bounds_mode == "Ultra-restritivos (≤115 mg/g)":
        # Bounds ultra-restritivos originais - KEEP SAME BUT ADD MINIMUM CHECK
        if model_name in ["Langmuir", "Sips"]:
            qmax_limit = min(qe_max * 1.2, 120)  # INCREASED FROM 1.05 to 1.2
            qmax_minimum = qe_max * 1.05  # Minimum 5% above experimental
            upper[0] = max(qmax_limit, qmax_minimum)
        elif model_name in ["Hill-Langmuir", "Tóth Modificado", "Khan (Inibição)"]:
            qmax_limit = min(qe_max * 1.15, 115)  # INCREASED FROM 1.02 to 1.15
            qmax_minimum = qe_max * 1.05
            upper[0] = max(qmax_limit, qmax_minimum)
        elif model_name in ["Langmuir Competitivo", "Langmuir Duplo"]:
            site_limit = min(qe_max * 0.6, 57)  # INCREASED FROM 0.52 to 0.6
            site_minimum = qe_max * 0.45
            site_limit = max(site_limit, site_minimum)
            upper[0] = site_limit
            if len(upper) >= 3:
                upper[2] = site_limit
    
    return (lower, tuple(upper))

def validate_parameters(model_name, params, qe_max, is_nonmonotonic=False, bounds_mode="Livres (sem limitação)"):
    """Valida se os parâmetros estão fisicamente realistas"""
    warnings = []
    
    if bounds_mode == "Livres (sem limitação)":
        # Validação muito branda para bounds livres
        if model_name in ["Langmuir", "Sips", "Hill-Langmuir"]:
            qmax = params[0]
            if qmax > qe_max * 3:  # Apenas avisar se muito exagerado
                warnings.append(f"ℹ️ qmax ({qmax:.1f}) é 3× maior que qe_experimental ({qe_max:.1f})")
        elif model_name in ["Langmuir Competitivo", "Langmuir Duplo"]:
            qmax_total = params[0] + params[2]
            if qmax_total > qe_max * 3:
                warnings.append(f"ℹ️ qmax_total ({qmax_total:.1f}) é 3× maior que qe_experimental")
        elif model_name in ["Tóth Modificado", "Khan (Inibição)"]:
            qmax = params[0]
            if qmax > qe_max * 3:
                warnings.append(f"ℹ️ qmax ({qmax:.1f}) é muito alto - verifique se é fisicamente realista")
        
        # Avisos sobre adequação do modelo (independente dos bounds)
        if is_nonmonotonic:
            if model_name in ["Freundlich", "Temkin"]:
                warnings.append(f"⚠️ {model_name} pode ser inadequado para dados não-monotônicos")
        
        return warnings
    
    # Validações mais rigorosas para bounds conservadores/restritivos
    if model_name in ["Langmuir", "Sips", "Hill-Langmuir"]:
        qmax = params[0]
        if bounds_mode == "Ultra-restritivos (≤115 mg/g)":
            if qmax > qe_max * 1.1:
                warnings.append(f"🚨 qmax ({qmax:.1f}) > qe_experimental × 1.1 ({qe_max*1.1:.1f})")
            if qmax > 120 and model_name in ["Langmuir", "Sips"]:
                warnings.append(f"🚨 qmax ({qmax:.1f}) > 120 mg/g (limite físico)")
            if qmax > 115 and model_name == "Hill-Langmuir":
                warnings.append(f"🚨 qmax ({qmax:.1f}) > 115 mg/g (limite para não-monotônico)")
        else:  # Conservadores
            if qmax > qe_max * 2:
                warnings.append(f"⚠️ qmax ({qmax:.1f}) > 2× qe_experimental ({qe_max*2:.1f})")
    
    elif model_name in ["Langmuir Competitivo", "Langmuir Duplo"]:
        qmax_total = params[0] + params[2]
        threshold = qe_max * 1.1 if bounds_mode == "Ultra-restritivos (≤115 mg/g)" else qe_max * 2
        if qmax_total > threshold:
            warnings.append(f"⚠️ qmax_total ({qmax_total:.1f}) > {threshold:.1f} mg/g")
    
    elif model_name in ["Tóth Modificado", "Khan (Inibição)"]:
        qmax = params[0]
        if bounds_mode == "Ultra-restritivos (≤115 mg/g)":
            if qmax > qe_max * 1.05:
                warnings.append(f"🚨 qmax ({qmax:.1f}) > qe_experimental × 1.05 ({qe_max*1.05:.1f})")
        else:  # Conservadores
            if qmax > qe_max * 2:
                warnings.append(f"⚠️ qmax ({qmax:.1f}) > 2× qe_experimental ({qe_max*2:.1f})")
    
    # Avisos específicos para modelos inadequados em dados não-monotônicos
    if is_nonmonotonic:
        if model_name in ["Freundlich", "Temkin"]:
            warnings.append(f"⚠️ Modelo {model_name} é INADEQUADO para dados não-monotônicos")
        elif model_name in ["Langmuir", "Sips", "Redlich-Peterson"]:
            warnings.append(f"⚠️ Modelo {model_name} assume comportamento monotônico")
    
    return warnings

def detect_outliers(Ce, qe, method="zscore", threshold=2.0):
    """Detecta outliers usando diferentes métodos"""
    if method == "zscore":
        z_scores = np.abs(zscore(qe))
        outliers = z_scores > threshold
    elif method == "iqr":
        Q1 = np.percentile(qe, 25)
        Q3 = np.percentile(qe, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (qe < lower_bound) | (qe > upper_bound)
    elif method == "monotonic":
        # Detecta pontos que quebram a monotonicidade esperada
        outliers = np.zeros(len(qe), dtype=bool)
        for i in range(len(qe)-1):
            if qe[i+1] < qe[i] * 0.8:  # Redução > 20%
                outliers[i+1] = True
    return outliers

st.sidebar.header("📊 Entrada de dados")
uploaded = st.sidebar.file_uploader("Carregue um arquivo (.xlsx ou .csv)", type=["xlsx","csv"])

def detect_nonmonotonic_behavior(Ce, qe):
    """Detecta se os dados apresentam comportamento não-monotônico"""
    if len(Ce) < 3:
        return False, "Poucos pontos para análise"
    
    # Ordenar por Ce para análise
    sorted_indices = np.argsort(Ce)
    Ce_sorted = Ce[sorted_indices]
    qe_sorted = qe[sorted_indices]
    
    # Encontrar o ponto de qe máximo
    max_idx = np.argmax(qe_sorted)
    qe_max = qe_sorted[max_idx]
    Ce_at_max = Ce_sorted[max_idx]
    
    # Verificar se há pontos após o máximo com qe significativamente menor
    points_after_max = qe_sorted[max_idx+1:] if max_idx < len(qe_sorted)-1 else []
    
    if len(points_after_max) == 0:
        return False, f"Sem pontos após qe_max ({qe_max:.1f} mg/g)"
    
    # Critério: redução > 5% do qe_max em pontos subsequentes
    min_after_max = np.min(points_after_max)
    reduction_pct = (qe_max - min_after_max) / qe_max * 100
    
    is_nonmonotonic = reduction_pct > 5.0
    
    if is_nonmonotonic:
        message = f"✅ Comportamento não-monotônico detectado!\n"
        message += f"   qe_max: {qe_max:.1f} mg/g (Ce = {Ce_at_max:.1f} mg/L)\n"
        message += f"   Redução: {reduction_pct:.1f}% após o pico\n"
        message += f"   → Recomendados: Modelos avançados (Hill, Tóth, Khan)"
    else:
        message = f"❌ Comportamento aparentemente monotônico (redução: {reduction_pct:.1f}%)\n"
        message += f"   → Adequados: Modelos clássicos (Langmuir, Freundlich)"
    
    return is_nonmonotonic, message

st.sidebar.subheader("🎯 Seleção de Modelos")

# Organizar modelos por categoria
classic_models = [name for name, spec in MODEL_SPECS.items() if spec.get("category") == "Clássico"]
advanced_models = [name for name, spec in MODEL_SPECS.items() if spec.get("category") == "Não-monotônico"]

st.sidebar.markdown("**📚 Modelos Clássicos (Monotônicos)**")
chosen_models = []
for m in classic_models:
    if st.sidebar.checkbox(m, value=True, key=f"classic_{m}"):
        chosen_models.append(m)

st.sidebar.markdown("**🔬 Modelos Avançados (Não-monotônicos)**")
st.sidebar.markdown("*Adequados para sistemas com redução em altas concentrações*")
for m in advanced_models:
    if st.sidebar.checkbox(m, value=False, key=f"advanced_{m}"):
        chosen_models.append(m)

st.sidebar.subheader("🔧 Tratamento de Dados")

# Opção de bounds
bounds_mode = st.sidebar.selectbox(
    "Tipo de bounds para parâmetros", 
    ["Livres (sem limitação)", "Conservadores (baseados nos dados)", "Ultra-restritivos (≤115 mg/g)"],
    index=0
)

if bounds_mode == "Livres (sem limitação)":
    st.sidebar.info("🆓 **Bounds livres**: Permite qmax ilimitado - adequado para explorar diferentes sistemas")
elif bounds_mode == "Conservadores (baseados nos dados)":
    st.sidebar.info("🎯 **Bounds conservadores**: qmax limitado a ~200% dos dados experimentais (ATUALIZADO)")
else:
    st.sidebar.warning("🔒 **Bounds ultra-restritivos**: qmax ≤ 115 mg/g - apenas para sistemas específicos")

# Sugerir método baseado nos modelos selecionados
has_nonmonotonic = any(MODEL_SPECS[m].get("category") == "Não-monotônico" for m in chosen_models)
if has_nonmonotonic:
    st.sidebar.info("💡 **Modelos não-monotônicos selecionados!** Considere usar 'Z-Score' ou 'IQR' ao invés de 'Monotonicidade'.")

outlier_method = st.sidebar.selectbox(
    "Método de detecção de outliers", 
    ["Nenhum", "Z-Score", "IQR", "Monotonicidade"], 
    index=0 if has_nonmonotonic else 3
)

# Definir outlier_action em todos os casos
outlier_action = "Nenhum"
z_threshold = 2.0

if outlier_method != "Nenhum":
    outlier_action = st.sidebar.selectbox(
        "Ação para outliers", 
        ["Marcar apenas", "Reduzir peso (10%)", "Excluir do ajuste"], 
        index=1
    )
    
    if outlier_method == "Z-Score":
        z_threshold = st.sidebar.slider("Limiar Z-Score", 1.0, 3.0, 2.0, 0.1)

st.sidebar.subheader("📈 Configurações de Plot")
x_grid_factor = st.sidebar.slider("Extensão do grid de Ce", 0.5, 2.0, 1.2, 0.05)
show_comparison = st.sidebar.checkbox("Mostrar comparação com/sem outliers", True)

if uploaded is not None:
    df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
    st.subheader("📋 Pré-visualização dos dados")
    st.dataframe(df.head())

    mapping = detect_columns(df)
    st.markdown("### 🎯 Mapeamento de colunas")
    col1, col2 = st.columns(2)
    with col1:
        Ce_col = st.selectbox("Coluna de Ce (mg/L)", df.columns, 
                             index=(df.columns.get_loc(mapping["Ce"]) if mapping["Ce"] in df.columns else 0))
        qe_col = st.selectbox("Coluna de qe (mg/g)", df.columns, 
                             index=(df.columns.get_loc(mapping["qe"]) if mapping["qe"] in df.columns else (1 if len(df.columns)>1 else 0)))
    with col2:
        sigma_Ce_col = st.selectbox("Coluna de σCe (mg/L) - Apenas para visualização", [None]+list(df.columns),
                                   index=(( [None]+list(df.columns) ).index(mapping["sigma_Ce"]) if mapping["sigma_Ce"] in ([None]+list(df.columns)) else 0))
        sigma_qe_col = st.selectbox("Coluna de σqe (mg/g) - Para ponderação", [None]+list(df.columns),
                                   index=(( [None]+list(df.columns) ).index(mapping["sigma_qe"]) if mapping["sigma_qe"] in ([None]+list(df.columns)) else 0))

    Ce = df[Ce_col].astype(float).to_numpy()
    qe = df[qe_col].astype(float).to_numpy()
    
    # 🔍 DETECÇÃO AUTOMÁTICA DE COMPORTAMENTO NÃO-MONOTÔNICO
    is_nonmonotonic, behavior_msg = detect_nonmonotonic_behavior(Ce, qe)
    
    if is_nonmonotonic:
        st.success("🎯 **COMPORTAMENTO NÃO-MONOTÔNICO DETECTADO!**")
        st.info(behavior_msg)
        st.warning("⚠️ **RECOMENDAÇÃO**: Desmarque modelos clássicos inadequados (Freundlich, Temkin) e foque nos **modelos avançados**")
    else:
        st.info("📈 **Comportamento aparentemente monotônico detectado**")
        st.info(behavior_msg)
    
    if sigma_qe_col is None:
        st.warning("⚠️ σ(qe) não informado – ajuste **não ponderado**. Recomenda-se σ(qe).")
        sigma_qe = np.ones_like(qe)
    else:
        sigma_qe = df[sigma_qe_col].astype(float).to_numpy()
        sigma_qe = np.where(sigma_qe<=0, np.nanmedian(sigma_qe[sigma_qe>0]) if np.any(sigma_qe>0) else 1.0, sigma_qe)
    
    sigma_Ce = np.zeros_like(Ce) if sigma_Ce_col is None else df[sigma_Ce_col].astype(float).to_numpy()
    
    # Detectar outliers
    outliers = np.zeros(len(qe), dtype=bool)  # Sempre inicializar como False
    if outlier_method == "Z-Score":
        outliers = detect_outliers(Ce, qe, "zscore", z_threshold)
    elif outlier_method == "IQR":
        outliers = detect_outliers(Ce, qe, "iqr")
    elif outlier_method == "Monotonicidade":
        outliers = detect_outliers(Ce, qe, "monotonic")
    
    if outlier_method != "Nenhum" and np.any(outliers):
        st.warning(f"🚨 **{np.sum(outliers)} outlier(s) detectado(s)** nos índices: {np.where(outliers)[0]}")
        outlier_df = pd.DataFrame({
            'Índice': np.where(outliers)[0],
            'Ce (mg/L)': Ce[outliers],
            'qe (mg/g)': qe[outliers],
            'σqe (mg/g)': sigma_qe[outliers] if sigma_qe_col else "N/A"
        })
        st.dataframe(outlier_df)
    
    # Preparar dados para ajuste
    if outlier_method != "Nenhum" and outlier_action == "Excluir do ajuste" and np.any(outliers):
        mask = ~outliers
        Ce_fit, qe_fit, sigma_qe_fit = Ce[mask], qe[mask], sigma_qe[mask]
        st.info(f"✂️ Ajuste realizado com {len(Ce_fit)} pontos (excluídos {np.sum(outliers)} outliers)")
    else:
        Ce_fit, qe_fit, sigma_qe_fit = Ce.copy(), qe.copy(), sigma_qe.copy()
        if outlier_method != "Nenhum" and outlier_action == "Reduzir peso (10%)" and np.any(outliers):
            sigma_qe_fit[outliers] *= np.sqrt(10)  # Reduz peso em 10x
            st.info("⚖️ Peso dos outliers reduzido para 10%")
    
    w_fit = 1.0/(sigma_qe_fit**2)
    
    # Ajustar modelos
    results = {}
    results_full = {}  # Para comparação com todos os pontos
    
    if bounds_mode != "Livres (sem limitação)":
        st.markdown("### 🔧 **Aplicando Bounds Baseados nos Dados**")
        bounds_info = []
    
    for name in chosen_models:
        spec = MODEL_SPECS[name]
        func = spec["func"]
        
        # Ajuste com dados filtrados
        try:
            p0 = spec["p0"](Ce_fit, qe_fit)
            bounds = spec["bounds"]
            
            # Ajustar bounds dinamicamente baseado nos dados e modo selecionado
            original_bounds = bounds
            bounds = adjust_bounds_to_data(name, bounds, Ce_fit, qe_fit, bounds_mode)
            
            # Capturar informações sobre limites aplicados apenas se não for livre
            if bounds_mode != "Livres (sem limitação)":
                if name in ["Langmuir", "Sips"]:
                    qmax_limit = bounds[1][0] if bounds[1][0] != np.inf else "∞"
                    bounds_info.append(f"🔧 **{name}**: qmax ≤ {qmax_limit:.1f}" if isinstance(qmax_limit, (int, float)) else f"🔧 **{name}**: qmax ≤ {qmax_limit}")
                elif name in ["Hill-Langmuir", "Tóth Modificado", "Khan (Inibição)"]:
                    qmax_limit = bounds[1][0] if bounds[1][0] != np.inf else "∞"
                    bounds_info.append(f"🔧 **{name}**: qmax ≤ {qmax_limit:.1f}" if isinstance(qmax_limit, (int, float)) else f"🔧 **{name}**: qmax ≤ {qmax_limit}")
                elif name in ["Langmuir Competitivo", "Langmuir Duplo"]:
                    site_limit = bounds[1][0] if bounds[1][0] != np.inf else "∞"
                    bounds_info.append(f"🔧 **{name}**: cada sítio ≤ {site_limit:.1f}" if isinstance(site_limit, (int, float)) else f"🔧 **{name}**: cada sítio ≤ {site_limit}")
            
            popt, pcov = curve_fit(func, Ce_fit, qe_fit, sigma=sigma_qe_fit, 
                                 absolute_sigma=True, p0=p0, bounds=bounds, maxfev=500000)
            perr = np.sqrt(np.diag(pcov))
            qhat = func(Ce_fit, *popt)
            R2w, ss_res_w = weighted_r2(qe_fit, qhat, w_fit)
            k, n = len(popt), len(qe_fit)
            aic, bic = aic_bic(ss_res_w, n, k)
            results[name] = {"popt": popt, "perr": perr, "R2w": R2w, "aic": aic, "bic": bic, "func": func}
        except Exception as e:
            if bounds_mode != "Livres (sem limitação)":
                bounds_info.append(f"❌ **{name}**: Falha no ajuste - {str(e)[:50]}...")
            results[name] = {"error": str(e)}
        
        # Ajuste com todos os dados (para comparação)
        if show_comparison and outlier_method != "Nenhum" and np.any(outliers):
            try:
                w_full = 1.0/(sigma_qe**2)
                p0_full = spec["p0"](Ce, qe)
                bounds_full = adjust_bounds_to_data(name, spec["bounds"], Ce, qe, bounds_mode)
                popt_full, pcov_full = curve_fit(func, Ce, qe, sigma=sigma_qe, 
                                               absolute_sigma=True, p0=p0_full, bounds=bounds_full, maxfev=500000)
                qhat_full = func(Ce, *popt_full)
                R2w_full, _ = weighted_r2(qe, qhat_full, w_full)
                results_full[name] = {"popt": popt_full, "R2w": R2w_full, "func": func}
            except:
                results_full[name] = {"error": "Falha"}
    
    # Mostrar bounds aplicados apenas se não for livre
    if bounds_mode != "Livres (sem limitação)" and bounds_info:
        st.info(f"**Limites aplicados automaticamente ({bounds_mode}):**")
        for info in bounds_info:
            st.write(info)

    # Exibir resultados
    st.markdown("### 📊 Resultados dos Ajustes")
    
    # Separar por categoria e validar parâmetros
    classic_results = []
    advanced_results = []
    qe_max_exp = np.nanmax(qe_fit)
    
    for name in chosen_models:
        res = results.get(name, {})
        category = MODEL_SPECS[name].get("category", "Clássico")
        
        if "error" in res:
            row = {"Modelo": name, "Status": f"❌ {res['error'][:50]}..."}
        else:
            spec = MODEL_SPECS[name]
            row = {"Modelo": name, "R² ponderado": f"{res['R2w']:.4f}", 
                   "AIC": f"{res['aic']:.2f}", "BIC": f"{res['bic']:.2f}"}
            
            # Validar parâmetros
            param_warnings = validate_parameters(name, res["popt"], qe_max_exp, is_nonmonotonic, bounds_mode)
            if param_warnings:
                row["⚠️ Avisos"] = " | ".join(param_warnings)
            
            # Adicionar comparação se disponível
            if name in results_full and "error" not in results_full[name]:
                row["R² (todos pontos)"] = f"{results_full[name]['R2w']:.4f}"
                delta_r2 = res['R2w'] - results_full[name]['R2w']
                row["ΔR²"] = f"{delta_r2:+.4f}"
            
            for pname, val, err in zip(spec["params"], res["popt"], res["perr"]):
                row[pname] = f"{val:.4g} ± {err:.2g}"
        
        if category == "Clássico":
            classic_results.append(row)
        else:
            advanced_results.append(row)
    
    # Exibir tabelas separadas
    if classic_results:
        st.markdown("#### 📚 **Modelos Clássicos**")
        classic_df = pd.DataFrame(classic_results).set_index("Modelo")
        st.dataframe(classic_df, use_container_width=True)
    
    if advanced_results:
        st.markdown("#### 🔬 **Modelos Avançados (Não-monotônicos)**")
        advanced_df = pd.DataFrame(advanced_results).set_index("Modelo")
        st.dataframe(advanced_df, use_container_width=True)
        
        # Destacar o melhor modelo não-monotônico
        if len(advanced_results) > 0:
            best_advanced = max([r for r in advanced_results if "R² ponderado" in r], 
                              key=lambda x: float(x["R² ponderado"]), default=None)
            if best_advanced:
                st.success(f"🏆 **Melhor modelo para comportamento não-monotônico**: {best_advanced['Modelo']} (R² = {best_advanced['R² ponderado']})")
        
        # Alertas sobre parâmetros problemáticos
        problematic_models = [r for r in advanced_results if "⚠️ Avisos" in r]
        if problematic_models:
            st.markdown("#### 🚨 **Alertas de Validação de Parâmetros**")
            for model in problematic_models:
                st.warning(f"**{model['Modelo']}**: {model['⚠️ Avisos']}")
    
    # Tabela combinada para download
    all_results = classic_results + advanced_results
    result_df = pd.DataFrame(all_results).set_index("Modelo")
    st.dataframe(result_df)
    
    # Download dos parâmetros
    if 'result_df' in locals():
        csv_data = result_df.to_csv().encode("utf-8")
        st.download_button("📥 Baixar parâmetros (.csv)", data=csv_data, 
                          file_name="isotermas_parametros.csv", mime="text/csv")

    # Gerar curvas para plotagem
    Ce_min_pos = np.nanmin(Ce[Ce>0]) if np.any(Ce>0) else np.nanmin(Ce)
    Ce_grid = np.linspace(max(1e-9, 0.8*Ce_min_pos), np.nanmax(Ce)*float(x_grid_factor), 200)
    
    pred_dict = {"Ce_grid": Ce_grid}
    for name in chosen_models:
        res = results.get(name, {})
        if "error" in res: continue
        pred_dict[f"{name}_filtrado"] = res["func"](Ce_grid, *res["popt"])
        
        if (outlier_method != "Nenhum" and name in results_full and 
            "error" not in results_full[name]):
            pred_dict[f"{name}_todos"] = results_full[name]["func"](Ce_grid, *results_full[name]["popt"])
    
    pred_df = pd.DataFrame(pred_dict)

    # Gráfico principal
    st.markdown("### 📈 Curvas ajustadas")
    
    fig, ax = plt.subplots(figsize=(10,7), dpi=120)
    
    # Plotar dados experimentais
    colors = ['blue', 'red', 'green', 'purple', 'brown']
    
    # Pontos normais
    normal_mask = ~outliers if outlier_method != "Nenhum" and np.any(outliers) else np.ones(len(Ce), dtype=bool)
    if np.any(normal_mask):
        ax.errorbar(Ce[normal_mask], qe[normal_mask], 
                   xerr=sigma_Ce[normal_mask] if sigma_Ce_col else None,
                   yerr=sigma_qe[normal_mask], 
                   fmt='o', capsize=4, color='blue', alpha=0.8,
                   label='Dados experimentais')
    
    # Outliers
    if outlier_method != "Nenhum" and np.any(outliers):
        ax.errorbar(Ce[outliers], qe[outliers], 
                   xerr=sigma_Ce[outliers] if sigma_Ce_col else None,
                   yerr=sigma_qe[outliers], 
                   fmt='s', capsize=4, color='red', alpha=0.8,
                   label=f'Outliers ({outlier_method})')
    
    # Plotar curvas ajustadas
    for i, name in enumerate(chosen_models):
        res = results.get(name, {})
        if "error" in res: continue
        
        color = colors[i % len(colors)]
        
        # Curva com dados filtrados
        ax.plot(Ce_grid, res["func"](Ce_grid, *res["popt"]), 
               color=color, linewidth=2.5, linestyle='-',
               label=f'{name} (R²={res["R2w"]:.3f})')
        
        # Curva com todos os dados (pontilhada)
        if (show_comparison and outlier_method != "Nenhum" and 
            name in results_full and "error" not in results_full[name]):
            ax.plot(Ce_grid, results_full[name]["func"](Ce_grid, *results_full[name]["popt"]), 
                   color=color, linewidth=1.5, linestyle='--', alpha=0.7,
                   label=f'{name} - todos pontos (R²={results_full[name]["R2w"]:.3f})')
    
    ax.set_xlabel("Ce (mg/L)", fontsize=12)
    ax.set_ylabel("qe (mg/g)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Título dinâmico baseado no tratamento
    if outlier_method == "Nenhum":
        title = "Isotermas ajustadas - Sem tratamento de outliers"
    else:
        title = f"Isotermas ajustadas - {outlier_method} - {outlier_action}"
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Download da figura
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", bbox_inches="tight", dpi=200)
    st.download_button("📥 Baixar figura (.png)", data=png_buf.getvalue(), 
                      file_name="isotermas_analise.png", mime="image/png")
    
    # Download do pacote completo
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="dados_originais", index=False)
        result_df.reset_index().to_excel(writer, sheet_name="parametros", index=False)
        pred_df.to_excel(writer, sheet_name="curvas_preditas", index=False)
        if outlier_method != "Nenhum" and np.any(outliers):
            outlier_df.to_excel(writer, sheet_name="outliers_detectados", index=False)
    
    st.download_button("📦 Baixar pacote completo (.xlsx)", data=excel_buf.getvalue(),
                      file_name="isotermas_analise_completa.xlsx",
                      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("📁 Carregue um arquivo para começar. Layout sugerido: **Ce, qe, sigma_Ce, sigma_qe**.")
    
    # Informações sobre bounds
    st.markdown("### ⚙️ **Configurações de Bounds Disponíveis**")
    st.markdown("""
    Este aplicativo oferece três modos de bounds para parâmetros:
    
    - **🆓 Livres (sem limitação)**: Ideal para explorar sistemas desconhecidos ou com alta capacidade
    - **🎯 Conservadores**: Baseados nos dados experimentais - qmax limitado a 200% dos dados (ATUALIZADO)
    - **🔒 Ultra-restritivos**: Para sistemas onde se conhece o limite físico (~115 mg/g)
    
    **💡 Recomendação**: Para sistemas como MFC/8B@CO que atingem ~55 mg/g, use bounds **Conservadores**.
    """)
    
    # Dados de exemplo
    demo = pd.DataFrame({
        "Ce": [18.10, 47.10, 73.85, 99.10, 177.85],
        "qe": [35.50, 62.15, 85.90, 112.85, 90.20],
        "sigma_Ce": [1.60, 0.90, 3.75, 5.80, 12.65],
        "sigma_qe": [2.70, 4.15, 1.30, 5.25, 11.80]
    })
    
    st.markdown("### 📊 **Exemplo de dados**")
    st.dataframe(demo, use_container_width=True)
    
    st.download_button("📥 Baixar planilha modelo (.csv)", 
                      data=demo.to_csv(index=False).encode("utf-8"),
                      file_name="isotermas_modelo.csv", mime="text/csv")
