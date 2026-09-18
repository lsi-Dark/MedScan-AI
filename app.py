import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import plotly.graph_objects as go
import time
import os

st.set_page_config(
    page_title="MedScan AI | المنظومة الطبية الذكية",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================= التصميم =============================
st.markdown("""
    <style>
    .main { direction: rtl; }

    #MainMenu, footer, header {visibility: hidden !important;}
    .viewerBadge_container__1QSob, .stDeployButton {display: none !important;}

    .stApp {
        background: linear-gradient(-45deg, #0b1220, #0f172a, #0c2a4a, #1e1b4b, #072033, #0f172a) !important;
        background-size: 400% 400% !important;
        animation: gradientShift 20s ease infinite !important;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    [data-testid="stSidebar"], 
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #070c14 0%, #0c1626 50%, #0a101d 100%) !important;
        color: #f1f5f9 !important;
    }

    [data-testid="stSidebar"] {
        border-left: 1px solid rgba(56, 189, 248, 0.2) !important;
    }

    .sidebar-card {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 14px;
        direction: rtl;
        text-align: right;
    }
    .sidebar-card h3 {
        color: #38bdf8 !important;
        font-size: 1.1rem;
        margin: 0 0 6px 0;
    }
    .sidebar-card p {
        color: #cbd5e1 !important;
        font-size: 0.88rem;
        line-height: 1.6;
        margin: 0;
    }

    [data-testid="stSidebarCollapseButton"] button {
        color: #38bdf8 !important;
    }

    .float-particle {
        position: fixed;
        z-index: 0;
        pointer-events: none;
        filter: blur(0.4px);
        animation-name: floatDrift;
        animation-timing-function: ease-in-out;
        animation-iteration-count: infinite;
        opacity: 0.18;
    }
    @keyframes floatDrift {
        0% { transform: translateY(0px) translateX(0px) rotate(0deg); }
        25% { transform: translateY(-25px) translateX(10px) rotate(6deg); }
        50% { transform: translateY(-45px) translateX(-8px) rotate(-4deg); }
        75% { transform: translateY(-15px) translateX(14px) rotate(3deg); }
        100% { transform: translateY(0px) translateX(0px) rotate(0deg); }
    }

    .block-container { position: relative; z-index: 1; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 12px; }

    .hero {
        background: linear-gradient(135deg, rgba(56,189,248,0.18) 0%, rgba(15,23,42,0.9) 60%);
        border: 1px solid #1e293b;
        border-radius: 18px;
        padding: 34px 30px;
        margin-bottom: 24px;
        text-align: right;
        direction: rtl;
    }
    .hero h1 { color: #60a5fa; margin: 0 0 8px 0; font-size: 2.1rem; }
    .hero p { color: #cbd5e1; font-size: 1.02rem; margin: 0; }
    .hero .badges { margin-top: 14px; }
    .hero .badge {
        display: inline-block;
        background-color: #0284c7;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin-left: 8px;
        margin-bottom: 6px;
    }

    .feature-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: right;
        direction: rtl;
        height: 100%;
    }
    .feature-card h4 { color: #38bdf8; margin: 0 0 8px 0; }
    .feature-card p { color: #94a3b8; font-size: 0.92rem; margin: 0; line-height: 1.6; }

    .step-box {
        background-color: #0f172a;
        border-right: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        text-align: right;
        direction: rtl;
    }
    .step-box b { color: #f8fafc; }
    .step-box span { color: #94a3b8; font-size: 0.9rem; }

    .dev-footer {
        margin-top: 40px;
        padding: 22px;
        border-radius: 16px;
        background: linear-gradient(90deg, rgba(2,132,199,0.15) 0%, rgba(15,23,42,0.9) 100%);
        border: 1px solid #1e293b;
        text-align: center;
        direction: rtl;
    }
    .dev-footer h3 { color: #38bdf8; margin: 4px 0; }
    .dev-footer p { color: #94a3b8; margin: 0; font-size: 0.9rem; }
    .dev-footer .tag {
        background-color: #0284c7;
        color: white;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 12px;
    }

    .button-selector-box {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 24px;
        direction: rtl;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# ============================= تهيئة النماذج والبيانات =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

brain_classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
brain_info = {
    'glioma_tumor': {
        'ar': 'ورم دبقي (Glioma)',
        'color': '#ef4444',
        'desc': 'ورم ينشأ داخل الخلايا الدبقية الداعمة للدماغ، ويحتاج تقييماً سريعاً مع طبيب الأعصاب.'
    },
    'meningioma_tumor': {
        'ar': 'ورم سحائي (Meningioma)',
        'color': '#f59e0b',
        'desc': 'ورم ينمو على الأغشية المحيطة بالدماغ والنخاع الشوكي، وغالباً ما يكون حميداً وبطيء النمو.'
    },
    'no_tumor': {
        'ar': 'سليم - لا يوجد ورم (Healthy Scan)',
        'color': '#10b981',
        'desc': 'لم تظهر الأشعة وجود أي كتل غير طبيعية. الأنسجة تبدو بحالة سليمة.'
    },
    'pituitary_tumor': {
        'ar': 'ورم الغدة النخامية (Pituitary)',
        'color': '#8b5cf6',
        'desc': 'ورم يقع في قاعدة الدماغ حول الغدة النخامية المسؤولة عن تنظيم الهرمونات.'
    }
}

lung_classes = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'normal',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]
lung_info = {
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': {
        'ar': 'ورم غدي نسيجي (Adenocarcinoma)',
        'color': '#ef4444',
        'desc': 'النوع الأكثر شيوعاً لسرطان الرئة، يبدأ في الخلايا المفرزة للمخاط في الأطراف الخارجية للرئة.'
    },
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': {
        'ar': 'سرطان الخلايا الكبيرة (Large Cell Carcinoma)',
        'color': '#f97316',
        'desc': 'ينمو وينتشر بسرعة في أي جزء من أجزاء نسيج الرئة، ويتطلب تدخلاً طبياً عاجلاً.'
    },
    'normal': {
        'ar': 'سليم - رئة طبيعية (Healthy Lung)',
        'color': '#10b981',
        'desc': 'صورة المفراس المقطعية تُظهر أنسجة رئة طبيعية تماماً وخالية من العُقيدات أو الأورام.'
    },
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': {
        'ar': 'سرطان الخلايا الحرشفية (Squamous Cell Carcinoma)',
        'color': '#ec4899',
        'desc': 'يبدأ عادةً في الممرات الهوائية الرئيسية في مركز الرئتين، ويرتبط غالباً بالتدخين.'
    }
}

@st.cache_resource
def load_brain_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    # مطابقة رأس النموذج مع معمارية التدريب الفعلية
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(brain_classes))
    )
    
    model_path = "brain_tumor_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"الملف {model_path} غير موجود في المسار الحالي.")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_lung_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(lung_classes))
    model_path = "lung_cancer_model.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"الملف {model_path} غير موجود في المسار الحالي.")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model

# ============================= أزرار التبديل المباشرة (Buttons) =============================
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "brain"

st.markdown('<div class="button-selector-box">', unsafe_allow_html=True)
st.markdown("<h4 style='color:#38bdf8; margin:0 0 12px 0;'>🎛️ اختر نمط الفحص الطبي المطلوب:</h4>", unsafe_allow_html=True)

btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    is_active_brain = (st.session_state.selected_mode == "brain")
    if st.button(
        "🧠 أورام الدماغ (Brain MRI)",
        key="btn_brain",
        use_container_width=True,
        type="primary" if is_active_brain else "secondary"
    ):
        st.session_state.selected_mode = "brain"
        st.rerun()

with btn_col2:
    is_active_lung = (st.session_state.selected_mode == "lung")
    if st.button(
        "🫁 سرطان الرئة (Chest CT)",
        key="btn_lung",
        use_container_width=True,
        type="primary" if is_active_lung else "secondary"
    ):
        st.session_state.selected_mode = "lung"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

is_brain = (st.session_state.selected_mode == "brain")

# ============================= القائمة الجانبية (Sidebar) =============================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 16px;">
            <div style="font-size: 48px; line-height: 1;">🩺</div>
            <h2 style="color: #60a5fa; margin: 6px 0 0 0; font-size: 1.4rem;">MedScan AI</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="sidebar-card">
            <h3>⚙️ مواصفات النموذج الفعّال</h3>
            <p style="margin-bottom: 4px;">• العضو المختار: <b style="color:#38bdf8;">{'الدماغ (Brain)' if is_brain else 'الرئة (Lung)'}</b></p>
            <p style="margin-bottom: 4px;">• المعمارية: <b style="color:#38bdf8;">{'ResNet-18 Deep CNN' if is_brain else 'ResNet-50 Deep CNN'}</b></p>
            <p style="margin-bottom: 4px;">• المعالجة: <b style="color:#38bdf8;">PyTorch Engine</b></p>
            <p>• الفحص: <b style="color:#38bdf8;">تصنيف متعدد (4 Classes)</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="sidebar-card">
            <h3>📬 للتواصل والطلب</h3>
            <p style="margin-bottom: 10px;">للحصول على السورس كود أو التراخيص:</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.link_button("🟢 تواصل عبر WhatsApp", "https://wa.me/9647712063322", use_container_width=True)
    st.link_button("📸 حساب Instagram", "https://instagram.com/6.mr.2", use_container_width=True)
    
    st.write("---")
    st.caption("💡 **تنويه:** هذا النظام أداة بحثية مساعدة ولا يغني عن تقرير الطبيب المختص.")

# ============================= تفعيل النموذج المختار =============================
if is_brain:
    try:
        active_model = load_brain_model()
    except Exception as e:
        st.error(f"⚠️ خطأ في تحميل ملف brain_tumor_model.pth: {e}")
        st.stop()
    active_classes = brain_classes
    active_info = brain_info
    hero_title = "🧠 منصة NeuroScan AI لتشخيص أورام الدماغ"
    hero_desc = "ارفع صورة الرنين المغناطيسي (MRI) للدماغ واحصل على تحليل فوري لاحتمالية وجود ورم ونوعه بدقة."
    active_arch = "ResNet-18"
    file_uploader_title = "اختر ملف صورة رنين مغناطيسي (Axial MRI)..."
    floating_icons = ["🧠", "🧬", "⚡", "🔬", "🧠", "🧬"]
else:
    try:
        active_model = load_lung_model()
    except Exception as e:
        st.error(f"⚠️ خطأ في تحميل ملف lung_cancer_model.pth: {e}")
        st.stop()
    active_classes = lung_classes
    active_info = lung_info
    hero_title = "🫁 منصة PulmoScan AI لتشخيص سرطان الرئة"
    hero_desc = "ارفع صورة الأشعة المقطعية للصدر (Chest CT Scan) لتشخيص نوع الورم الخبيث أو سلامة نسيج الرئة."
    active_arch = "ResNet-50"
    file_uploader_title = "اختر ملف صورة مفراس مقطعي للصدر (CT Scan)..."
    floating_icons = ["🫁", "🩻", "⚡", "🔬", "🫁", "🩻"]

# ============================= خلفية متحركة =============================
particles_html = ""
positions = [
    ("6%", "12%", "46px", "0s", "9s"),
    ("18%", "78%", "34px", "1.5s", "11s"),
    ("42%", "4%", "30px", "0.7s", "13s"),
    ("58%", "88%", "40px", "2.2s", "10s"),
    ("75%", "20%", "36px", "1s", "12s"),
    ("88%", "60%", "44px", "0.4s", "14s"),
]
for icon, (top, left, size, delay, duration) in zip(floating_icons, positions):
    particles_html += f'<div class="float-particle" style="top:{top}; left:{left}; font-size:{size}; animation-delay:{delay}; animation-duration:{duration};">{icon}</div>'

st.markdown(particles_html, unsafe_allow_html=True)

# ============================= حركة الماوس التفاعلية =============================
components.html("""
<script>
(function() {
    const doc = window.parent.document;
    if (doc.getElementById('ns-mouse-layer')) return;

    const wrap = doc.createElement('div');
    wrap.id = 'ns-mouse-layer';
    wrap.style.position = 'fixed';
    wrap.style.inset = '0';
    wrap.style.zIndex = '0';
    wrap.style.pointerEvents = 'none';
    wrap.style.overflow = 'hidden';
    doc.body.appendChild(wrap);

    const icons = ['🩺', '🧬', '✨', '🔬'];
    const items = [];
    for (let i = 0; i < 12; i++) {
        const el = doc.createElement('div');
        el.textContent = icons[i % icons.length];
        el.style.position = 'absolute';
        el.style.fontSize = (22 + Math.random() * 26) + 'px';
        el.style.opacity = 0.10 + Math.random() * 0.16;
        el.style.left = Math.random() * 100 + '%';
        el.style.top = Math.random() * 100 + '%';
        el.style.transition = 'transform 0.7s ease-out';
        el.style.filter = 'blur(0.4px)';
        wrap.appendChild(el);
        items.push({ el: el, factor: 0.02 + Math.random() * 0.06 });
    }

    doc.addEventListener('mousemove', function(e) {
        const cx = doc.documentElement.clientWidth / 2;
        const cy = doc.documentElement.clientHeight / 2;
        const dx = (e.clientX - cx);
        const dy = (e.clientY - cy);
        items.forEach(function(it) {
            it.el.style.transform = 'translate(' + (dx * it.factor) + 'px, ' + (dy * it.factor) + 'px)';
        });
    });
})();
</script>
""", height=0)

# ============================= الهيدر الرئيسي =============================
st.markdown(f"""
    <div class="hero">
        <h1>{hero_title}</h1>
        <p>{hero_desc}</p>
        <div class="badges">
            <span class="badge">Deep Learning</span>
            <span class="badge">{active_arch}</span>
            <span class="badge">4 تصنيفات نسيجية</span>
            <span class="badge">تحليل فوري دقيق</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ============================= التبويبات =============================
tab_diagnosis, tab_about, tab_how = st.tabs(["🔬 بدء التشخيص", "📖 عن المشروع", "⚙️ كيف يعمل النظام"])

# ----------------------------- تبويب: التشخيص -----------------------------
with tab_diagnosis:
    col_upload, col_result = st.columns([1, 1.2], gap="large")

    with col_upload:
        st.subheader("📤 رفع الصورة")
        uploaded_file = st.file_uploader(file_uploader_title, type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="الصورة المدخلة للفحص", use_container_width=True)
        else:
            st.caption("📌 يرجى رفع صورة واضحة ومقصوصة بشكل جيد للحصول على أدق نتيجة.")

    with col_result:
        if uploaded_file is not None:
            st.subheader("🔍 نتيجة التحليل الطبي")

            with st.spinner("جاري قراءة الميزات وتحليل الأنسجة..."):
                time.sleep(0.3)
                data_transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                tensor = data_transforms(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = active_model(tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    confidence, predicted_class_idx = torch.max(probabilities, 0)
                    predicted_class = active_classes[predicted_class_idx.item()]
                    conf_score = confidence.item() * 100

            target_info = active_info[predicted_class]

            st.markdown(f"""
                <div style="background-color: {target_info['color']}22; border: 2px solid {target_info['color']}; padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: right;">
                    <h3 style="color: {target_info['color']}; margin: 0;">{target_info['ar']}</h3>
                    <p style="color: #cbd5e1; margin-top: 8px;">{target_info['desc']}</p>
                    <hr style="border-color: {target_info['color']}44;">
                    <h4 style="margin: 0; color: #f8fafc;">درجة اليقين: <b>{conf_score:.2f}%</b></h4>
                </div>
            """, unsafe_allow_html=True)

            st.write("📊 **توزيع الاحتمالات لجميع الفئات:**")

            ar_labels = [active_info[c]['ar'] for c in active_classes]
            scores = [probabilities[i].item() * 100 for i in range(len(active_classes))]
            colors = [active_info[c]['color'] for c in active_classes]

            fig = go.Figure(go.Bar(
                x=scores,
                y=ar_labels,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{s:.2f}%" for s in scores],
                textposition='outside'
            ))

            fig.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(range=[0, 115], showgrid=True, title="الاحتمالية (%)"),
                yaxis=dict(autorange="reversed"),
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 ملاحظة طبية مهمة"):
                st.write("هذه النتيجة مبنية على نموذج ذكاء اصطناعي تدريبي، وهي أداة مساعدة فقط. القرار التشخيصي النهائي يجب أن يصدر من طبيب أشعة أو اختصاصي أورام مختص بعد مراجعة الصورة كاملة وسجل المريض.")
        else:
            st.info("👈 قم برفع صورة الأشعة الطبية من اليمين للبدء بالتشخيص التلقائي.")

# ----------------------------- تبويب: عن المشروع -----------------------------
with tab_about:
    st.markdown("<h2 style='text-align:right; color:#60a5fa;'>📖 عن منصة MedScan AI</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align:right; color:#cbd5e1; direction:rtl; font-size:1.02rem; line-height:1.8;'>
        <b>MedScan AI</b> هي منصة طبية متكاملة تعتمد على شبكات التعلم العميق (Deep Learning) لدعم أطباء الأشعة
        والأورام في الفرز والتشخيص السريع. تدمج المنظومة بين تحليل أشعة الرنين المغناطيسي <b>(Brain MRI)</b> لتصنيف كتل الدماغ،
        وفحوصات المفراس المقطعي <b>(Chest CT)</b> لكشف وتمييز أنسجة وسرطانات الرئة بدقة حسابية عالية.
        </p>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""
            <div class="feature-card">
                <h4>🎯 شنو يسوي المشروع؟</h4>
                <p>يحلل صور الأشعة الطبية (رنين مغناطيسي أو مفراس)، ويحدد احتمالية الإصابة ونوع النسيج المصاب أو السليم من بين 4 فئات لكل عضو وبنسبة يقين واضحة.</p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="feature-card">
                <h4>💡 ليش ممكن يفيد؟</h4>
                <p>يقلل زمن انتظار الفرز الأولي، ويوفر رأياً استشارياً ثانياً فورياً للكوادر الطبية والمستشفيات، ويدعم دقة القرار الإكلينيكي في الحالات الحرجة.</p>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="feature-card">
                <h4>👥 لمين موجه؟</h4>
                <p>أطباء الأشعة، مراكز الفحص الطبي، طلاب وباحثي الذكاء الاصطناعي والهندسة الطبية، ولكل مطور يتعلم تطبيقات الرؤية الحاسوبية في الطب الحياتي.</p>
            </div>
        """, unsafe_allow_html=True)

    st.write("---")
    st.markdown(f"<h4 style='text-align:right; color:#38bdf8;'>🧬 الفئات المعتمدة للنموذج المختار حالياً ({'الدماغ' if is_brain else 'الرئة'})</h4>", unsafe_allow_html=True)
    for c in active_classes:
        info = active_info[c]
        st.markdown(f"""
            <div style="background-color:{info['color']}15; border-right:4px solid {info['color']};
                        padding:12px 16px; border-radius:8px; margin-bottom:10px; text-align:right; direction:rtl;">
                <b style="color:{info['color']};">{info['ar']}</b>
                <p style="color:#94a3b8; margin:4px 0 0 0; font-size:0.9rem;">{info['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

    st.warning("⚠️ هذا المشروع لأغراض بحثية ودعم أكاديمي، ونتائجه استرشادية لا تُعد بديلاً عن الفحص السريري المباشر.")

# ----------------------------- تبويب: كيف يعمل النظام -----------------------------
with tab_how:
    st.markdown("<h2 style='text-align:right; color:#60a5fa;'>⚙️ آلية عمل خط المعالجة (Pipeline)</h2>", unsafe_allow_html=True)

    steps = [
        ("1️⃣ اختيار نمط الفحص", "يحدد المستخدم نوع العضو المطلوب تحليله عبر أزرار الفحص السريع (أورام الدماغ أو سرطان الرئة)."),
        ("2️⃣ المعالجة المسبقة (Preprocessing)", "تُحوّل الصورة إلى 3 قنوات وتُضبط أبعادها بدقة 224×224 مع تطبيع قيم البكسلات حسب معايير ImageNet."),
        ("3️⃣ الشبكات العصبية الالتفافية (CNN Backbones)", "يتم تمرير الصورة عبر ResNet-18 للدماغ أو ResNet-50 للرئة لاستخلاص الخصائص المجهرية للأنسجة بدقة عالية."),
        ("4️⃣ طبقة التصنيف (Softmax Head)", "تُحسب التوزيعات الاحتمالية لكل صنف طبي، وتُحدد الفئة الفائزة بأعلى نسبة يقين."),
        ("5️⃣ التحليل المرئي (Visualization)", "تُعرض النتيجة الفورية مشفوعة بنسبة اليقين وبطاقة طبية شارحة ورسم بياني تفاعلي كامل."),
    ]
    for title, desc in steps:
        st.markdown(f"""
            <div class="step-box">
                <b>{title}</b><br>
                <span>{desc}</span>
            </div>
        """, unsafe_allow_html=True)

    st.write("---")
    colA, colB = st.columns(2)
    with colA:
        st.metric("معمارية النموذج الفعّال", active_arch)
    with colB:
        st.metric("عدد الفئات المستهدفة", "4 تصنيفات")

# ============================= فوتر المطور =============================
st.markdown("""
    <div class="dev-footer">
        <p style="color:#94a3b8; font-size:13px; margin-bottom:4px;">تصميم وتطوير</p>
        <h3>المهندس مرتضى عبد الكريم</h3>
        <span class="tag">AI Engineer</span>
        <p style="margin-top:10px;">مشروع MedScan AI © 2026 — لأغراض بحثية وتعليمية</p>
    </div>
""", unsafe_allow_html=True)