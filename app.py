%%writefile app.py

# ===========================================
# 🎓 المساعد الجامعي الذكي
# - بحث ووصف المواد من: DATASET des 2.xlsx
# - اقتراح جدول من: DATASET table.csv + student_test.csv
# ===========================================
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re, difflib, traceback, os
from fuzzywuzzy import fuzz
from huggingface_hub import login
import logging, uuid, time
from flask import session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def dbg(tag, **kw):
    # سجل مختصر وواضح
    safe = {k: (str(v)[:200] if v is not None else None) for k, v in kw.items()}
    logging.info(f"[{tag}] {safe}")


# ✅ مهم: لا تضع التوكن داخل الكود.
# في كولاب: من Settings -> Secrets -> أضف HF_TOKEN
# أو قبل التشغيل:
# %env HF_TOKEN=xxxxx
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if HF_TOKEN:
    login(HF_TOKEN)

# ===========================================
# 🧠 Chat Memory (per session)
# ===========================================
chat_memory = {}  # { session_id: { major, course_code, course_name, ts } }

# ===========================================
# 🧩 تحميل البيانات
# ===========================================
PATH_TABLE = "/content/DATASET table 2.xlsx"
PATH_STUDS = "/content/student_test 2.xlsx"
PATH_DES   = "/content/DATASET des 2.xlsx"   # ← عدّل المسار لو مختلف

students_df = pd.read_excel(PATH_STUDS)
courses_df  = pd.read_excel(PATH_TABLE)
desc_df     = pd.read_excel(PATH_DES)

students_df.fillna("", inplace=True)
courses_df.fillna("", inplace=True)
desc_df.fillna("", inplace=True)

# توحيد أنواع
courses_df = courses_df.astype(str)
desc_df    = desc_df.astype(str)

# تأكد من الأعمدة الضرورية في desc_df
for col in ['رقم المادة','اسم المادة','عدد الساعات','الوصف',
            'المواد المتزامنة','المتطلب السابق','نوع المادة','اسم التخصص','ملاحظات']:
    if col not in desc_df.columns:
        desc_df[col] = ''

# تأكد من الأعمدة الضرورية في courses_df (للجدول)
for col in ['اسم المادة','رقم المادة','عدد الساعات','نوع المادة',
            'المواد المتزامنة','المتطلب السابق','ملاحظات','اسم التخصص']:
    if col not in courses_df.columns:
        courses_df[col] = ''

# ===========================================
# ⚙️ نموذج التضمين + كاش للفهرسة حسب التخصص
# ===========================================
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")

def build_text(row):
    """
    نص شامل للمادة من ملف الوصف:
    ندمج أهم الأعمدة في جملة واحدة.
    """
    parts = []
    for col in ['اسم المادة','رقم المادة','عدد الساعات','نوع المادة','الوصف',
                'المواد المتزامنة','المتطلب السابق','ملاحظات','اسم التخصص']:
        val = str(row.get(col, "")).strip()
        if val:
            parts.append(f"{col}: {val}")
    return " - ".join(parts)

# كاش لكل تخصص: { major: {"df":..., "texts":..., "index":...} }
major_cache = {}
embedding_dim = embedding_model.get_sentence_embedding_dimension()

def get_major_index(major_name):
    """
    يبني (أو يسترجع من الكاش) فهرس FAISS + DataFrame + النصوص لتخصص معيّن.
    """
    major_name = str(major_name or "").strip()
    if not major_name:
        return None

    if major_name in major_cache:
        return major_cache[major_name]

    sub = desc_df[
        desc_df["اسم التخصص"].astype(str).str.contains(major_name, regex=False, na=False)
    ].copy()

    if sub.empty:
        return None

    texts = sub.apply(build_text, axis=1).tolist()
    embs = embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embs)

    major_cache[major_name] = {
        "df": sub.reset_index(drop=True),
        "texts": texts,
        "index": index
    }
    return major_cache[major_name]

# ===========================================
# 🧠 تصحيح النصوص (Spell Correction)
# ===========================================
subjects = courses_df["اسم المادة"].dropna().astype(str).tolist()
subject_words = list({w for s in subjects for w in s.replace("(", "").replace(")", "").split()})
alias_dict = {
    "الدكا": "الذكاء",
    "الاستناعي": "الاصطناعي",
    "مفدمت": "مقدمة",
    "الربوتات": "الروبوتات"
}

# ===========================================
# 🔤 Arabic Text Normalization (حل عام للأخطاء الكتابية)
# ===========================================
def normalize_ar(text: str) -> str:
    text = str(text or "").strip()

    # توحيد الألف
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

    # توحيد الياء والألف المقصورة
    text = text.replace("ى", "ي")

    # توحيد التاء المربوطة والهاء (مفيد للكشف، وليس للعرض)
    text = text.replace("ة", "ه")

    # إزالة التطويل
    text = text.replace("ـ", "")

    return text.lower()

def is_lab_query(text):
    t = normalize_ar(text)
    return any(k in t for k in [
        "مختبر", "لاب", "lab", "عملي"
    ])


def mentions_course_name(text: str, df) -> bool:
    q = normalize_ar(text)
    for _, row in df.iterrows():
        name = normalize_ar(row.get("اسم المادة", ""))
        if name and name in q:
            return True
    return False


def find_course_by_name_only(query, df):
    q = normalize_ar(query)

    best_row = None
    best_score = 0

    for _, row in df.iterrows():
        name = normalize_ar(row.get("اسم المادة", ""))
        if not name:
            continue

        # تطابق صريح
        if name in q:
            return row

        # fuzzy قوي على الاسم فقط
        s = fuzz.token_set_ratio(q, name)
        if s > best_score:
            best_score = s
            best_row = row

    if best_row is not None and best_score >= 85:
        return best_row

    return None

def correct_words_with_alias(text, word_list, alias_dict):
    words = text.split()
    corrected = []
    for word in words:
        if word in alias_dict:
            corrected.append(alias_dict[word])
        else:
            match = difflib.get_close_matches(word, word_list, n=1, cutoff=0.7)
            corrected.append(match[0] if match else word)
    return " ".join(corrected)

# ===========================================
# 🔍 البحث عن مادة داخل تخصص معيّن
# ===========================================
def search_course_for_major(query, major, top_k=3):
    idx_data = get_major_index(major)
    if idx_data is None:
        return None, query, 0.0

    base_df = idx_data["df"]
    want_lab = is_lab_query(query)

    # 🔹 اختر DF المناسب
    if want_lab:
        df = base_df[
            base_df["اسم المادة"].astype(str).str.contains("مختبر", na=False)
            | base_df["نوع المادة"].astype(str).str.contains("مختبر|عملي", na=False)
        ].copy()

        if df.empty:
            return None, query, 0.0

        texts = df.apply(build_text, axis=1).tolist()
        embs = embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embs)

    else:
        df = base_df
        index = idx_data["index"]

    # ✅ 1) حسم بالاسم فقط
    name_hit = find_course_by_name_only(query, df)
    if name_hit is not None:
        return name_hit, name_hit["اسم المادة"], 1.0

    # 🔹 2) semantic search
    corrected = correct_words_with_alias(query, subject_words, alias_dict)
    q_emb = embedding_model.encode(
        [corrected],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(q_emb, top_k)

    best_score = float(scores[0][0])
    best_idx   = int(indices[0][0])

    if best_score < 0.3:
        return None, corrected, best_score

    row = df.iloc[best_idx]
    return row, corrected, best_score





def get_session_id():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def remember_course(session_id, major, row):
    code = str(row.get("رقم المادة", "")).strip()
    name = str(row.get("اسم المادة", "")).strip()

    chat_memory[session_id] = {
        "major": major,
        "course_code": code,
        "course_name": name,
        "ts": time.time()
    }

    dbg("MEM_SAVE", sid=session_id, code=code, name=name)


def recall_course(session_id):
    mem = chat_memory.get(session_id)

    if not mem:
        dbg("MEM_MISS", sid=session_id)
        return None

    # TTL = 30 minutes
    if time.time() - mem["ts"] > 30 * 60:
        chat_memory.pop(session_id, None)
        dbg("MEM_EXPIRED", sid=session_id)
        return None

    dbg("MEM_HIT", sid=session_id, code=mem["course_code"])
    return mem

def get_row_by_code_for_major(course_code, major):
    course_code = str(course_code or "").strip()
    if not course_code:
        return None

    sub = desc_df[
        desc_df["اسم التخصص"].astype(str)
        .str.contains(str(major), regex=False, na=False)
    ]

    if sub.empty:
        return None

    hit = sub[sub["رقم المادة"].astype(str).str.strip() == course_code]
    return hit.iloc[0] if not hit.empty else None


# ===========================================
# 🤖 إعداد النموذج
# ===========================================
LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer_llm = AutoTokenizer.from_pretrained(LLM_NAME)

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

def generate_llm_response(system_prompt, user_prompt, max_tokens=300):
    system_prompt = f"""
{system_prompt}

⚠️ تعليمات إلزامية:
- أجب باللغة العربية فقط.
- يمنع استخدام أي كلمة أو مصطلح غير عربي.
- إذا كان سؤال الطالب بلغة أخرى، قم بترجمة المعنى وأجب بالعربية.
- لا تذكر أنك تترجم.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer_llm.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer_llm(prompt, return_tensors="pt").to(llm_model.device)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9
    )

    full_text = tokenizer_llm.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in full_text:
        return full_text.split("assistant")[-1].strip()

    return full_text.strip()

# ===========================================
# 💬 رد تفاعلي بناءً على معلومة دقيقة (Course Info)
# ===========================================
def make_response_interactive(fact_text, user_question):
    casual_triggers = ["شو", "كم", "ليش", "بتنزل", "مع مين", "احكيلي", "اشرحلي"]
    is_casual = any(w in user_question for w in casual_triggers)

    tone = "أسلوب شبابي بسيط" if is_casual else "أسلوب رسمي جامعي"

    system = """
أنت مساعد جامعي ذكي وودود.
ممنوع استخدام أي لغة غير العربية.
أجب دائمًا باللغة العربية فقط.
"""

    user = f"""
هذه معلومة دقيقة من قاعدة البيانات:
{fact_text}

سؤال الطالب:
{user_question}

{tone}.
جاوب على قد السؤال بس.
اطبعها بنفس المعلومات لكن بتنسيق أبسط.
"""

    return generate_llm_response(system, user)

# ===========================================
# 💬 رد تفاعلي لتحية فقط
# ===========================================
def reply_greeting(user_text):
    system = """
أنت مساعد جامعي ذكي وودود.
ممنوع استخدام أي لغة غير العربية.
أجب دائمًا باللغة العربية فقط.
"""

    user = f"""
الطالب كتب تحية:
{user_text}

رد قصير وودود واطلب منه يسأل عن المواد أو الجدول.
"""
    return generate_llm_response(system, user, max_tokens=100)

# ===========================================
# 🧩 استخراج إجابات مباشرة من صف المادة
# ===========================================
def resolve_course_field(val):
    val = (val or "").strip()
    if not val:
        return ""

    # إذا فيه أرقام → حاول تفسيره ككود
    if any(ch.isdigit() for ch in val):
        codes = re.findall(r"[A-Za-z]{2,3}\d{1,3}", val)
        if codes:
            names = []
            for c in codes:
                match = desc_df[desc_df["رقم المادة"].astype(str).str.strip() == c]
                if not match.empty:
                    names.append(f"{match.iloc[0]['اسم المادة']} ({c})")
                else:
                    names.append(c)
            return "، ".join(names)

    # غير ذلك → اسم مادة عربي، اطبعه كما هو
    return val

def handle_direct_question(user_text, row):
    """
    يحاول يستخرج إجابات مباشرة:
    - متطلب سابق
    - مواد متزامنة
    - عدد الساعات
    - نوع المادة
    من صف المادة (row) القادم من desc_df.
    """
    text = normalize_ar(user_text)

    def get_course_name_from_code(code):
        code = str(code or "").strip()
        if not code:
            return ""
        match = desc_df[desc_df['رقم المادة'].astype(str).str.strip() == code]
        if not match.empty:
            name = match.iloc[0]['اسم المادة']
            return f"{name} ({code})"
        return f"({code})"

    # 🔹 المواد المتزامنة
    if any(k in text for k in [
        normalize_ar("متزامنة"),
        normalize_ar("المواد المتزامنة"),
        normalize_ar("تنزل مع"),
        normalize_ar("بتنزل مع"),
        normalize_ar("متزامنه"),
    ]):
        val = resolve_course_field(row.get('المواد المتزامنة', ''))
        return f"المادة المتزامنة: {val}" if val else "لا توجد مادة متزامنة لهذه المادة."


    # 🔹 المتطلب السابق
    if any(k in text for k in [
        normalize_ar("متطلب"),
        normalize_ar("المتطلب السابق"),
        normalize_ar("شرط"),
    ]):
        val = resolve_course_field(row.get('المتطلب السابق', ''))
        return f"المتطلب السابق: {val}" if val else "لا يوجد متطلب سابق لهذه المادة."


    # 🔹 عدد الساعات
    if any(k in text for k in [
        normalize_ar("عدد الساعات"),
        normalize_ar("كم ساعة"),
        normalize_ar("ساعات"),
        normalize_ar("ساعه"),
    ]):
        val = (row.get('عدد الساعات', '') or '').strip()
        return f"عدد الساعات: {val}" if val else "عدد الساعات غير محدد في البيانات."

    # 🔹 نوع المادة
    if any(k in text for k in [
        normalize_ar("نوع المادة"),
        normalize_ar("تصنيف"),
        normalize_ar("اجباري"),
        normalize_ar("اختياري"),
    ]):
        val = (row.get('نوع المادة', '') or '').strip()
        return f"نوع المادة: {val}" if val else "نوع المادة غير محدد في البيانات."

    return None

# ===========================================
# 🤖 وصف المادة باستخدام النموذج (من صف مباشرة)
# ===========================================
def describe_course_from_row(corrected_query, row):
    items = []
    for col in ['اسم المادة','رقم المادة','عدد الساعات','نوع المادة','الوصف',
                'المواد المتزامنة','المتطلب السابق','ملاحظات','اسم التخصص']:
        val = str(row.get(col, "")).strip()
        if val:
            items.append(f"{col}: {val}")

    info = "\n".join(items)

    system = """
أنت مساعد جامعي ذكي وودود.
ممنوع استخدام أي لغة غير العربية.
أجب دائمًا باللغة العربية فقط.
"""

    user = f"""
سؤال الطالب:
{corrected_query}

معلومات المادة:
{info}

هذه بيانات رسمية من قاعدة بيانات جامعية.
ممنوع التفسير أو التخمين.
ممنوع تغيير أسماء المواد أو المتطلبات.
ممنوع إضافة أي معلومة غير موجودة في البيانات.

اطبع معلومات المادة نفسها بدون تغيير أو نقصان.
جاوب على قد السؤال فقط.
بصيغة بسيطة ومفهومة وتشمل كل المعلومات المهمة.
"""

    return generate_llm_response(system, user)

# ===========================================
# 🧠 NLP مبسّط: كشف نية السؤال (Intent)
# ===========================================
intent_examples = {
    "greeting": [
        "مرحبا", "هلا", "السلام عليكم", "أهلًا", "أهلين",
        "صباح الخير", "مساء الخير", "كيف حالك", "هاي", "hi"
    ],
    "course_query": [
        "ما هو متطلب", "كم ساعة مادة", "شرح مادة", "تنزل مع",
        "ما هي مادة", "اريد معلومات عن المادة", "شو بتنزل مع",
        "شو متطلب مادة", "شو المتزامنة مع", "ما هو المتطلب السابق", "عدد الساعات لمادة", "نوع المادة",
        "وصف مادة"
    ],
    "other": [
        "كيف الجو اليوم", "من افضل لاعب", "كيف اطبخ",
        "شو رأيك في الحياة", "نصيحة للحب", "كيف أنحف"
    ]
}

intent_vectors = {}
for intent, samples in intent_examples.items():
    intent_vectors[intent] = embedding_model.encode(
        samples,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def detect_intent(text):
    clean = str(text or "").strip()
    if not clean:
        return "other", 0.0

    q = embedding_model.encode([clean], convert_to_numpy=True, normalize_embeddings=True)[0]
    best_intent = "other"
    best_score = -1.0

    for intent, vecs in intent_vectors.items():
        scores = np.dot(vecs, q)
        max_score = float(np.max(scores))
        if max_score > best_score:
            best_score = max_score
            best_intent = intent

    return best_intent, best_score

# ===========================================
# 📅 محرك الجدول المتقدم (من table + students فقط)
# ===========================================
def full_schedule_engine(student_id, password, max_hours=18):
    local_students = pd.read_excel(PATH_STUDS)
    local_courses  = pd.read_excel(PATH_TABLE)

    student_row = local_students[
        (local_students["ID"].astype(str) == str(student_id)) &
        (local_students["password"].astype(str) == str(password))
    ]
    if student_row.empty:
        return None

    student = student_row.iloc[0]
    student_year = int(student.get("year", 1))
    completed_hours = float(student.get("Completed_Hours", 0))
    student_major = str(student.get("student_major", "")).strip()

    completed_codes_raw = set(str(x).strip() for x in str(student.get("Completed_Course_ids","")).split(";") if x.strip())
    completed_names = [n.strip() for n in str(student.get("Completed_Course_Names","")).split(";") if n.strip()]

    def normalize_code(s):
        if not s:
            return ""
        s = str(s).translate(str.maketrans("٠١٢٢٣٤٥٦٧٨٩","01223456789"))
        s = str(s).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩","0123456789"))
        return re.sub(r"[^A-Za-z0-9]","",s).upper()

    norm_completed_codes = { normalize_code(c) for c in completed_codes_raw }

    df = local_courses.copy()
    df.columns = df.columns.str.strip()
    if "اسم التخصص" in df.columns and student_major:
        df = df[df["اسم التخصص"].astype(str).str.contains(student_major, regex=False, na=False)].copy()

    def extract_year(note):
        m = re.search(r"السنة\s*-\s*(\d+)", str(note))
        return int(m.group(1)) if m else None

    def extract_required_hours(note):
        nt = str(note).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩","0123456789"))
        m = re.search(r"قطع\s*(\d+)\s*ساعة", nt)
        return int(m.group(1)) if m else 0

    df["Target_Year"] = df["ملاحظات"].apply(extract_year)
    df["Required_Hours"] = df["ملاحظات"].apply(extract_required_hours)

    def split_tokens(txt):
        if not txt or not str(txt).strip():
            return []
        return [p.strip() for p in re.split(r"[;,،]\s*", str(txt)) if p.strip()]

    def prereq_ok(field):
        toks = split_tokens(field)
        if not toks:
            return True
        return all(normalize_code(t) in norm_completed_codes for t in toks)

    def completed_by_name(name):
        if not name:
            return False
        return any(fuzz.token_set_ratio(name, cn) >= 90 for cn in completed_names)

    def year_ok(target, sy):
        if pd.isna(target):
            return True
        try:
            target = int(target)
            return target <= sy + 1
        except:
            return True

    df["Norm_Code"] = df["رقم المادة"].apply(normalize_code)
    df["Prereq_OK"] = df["المتطلب السابق"].apply(prereq_ok)
    df["Completed_By_Code"] = df["Norm_Code"].apply(lambda c: c in norm_completed_codes)
    df["Completed_By_Name"] = df["اسم المادة"].apply(completed_by_name)
    df["Not_Completed"] = ~(df["Completed_By_Code"] | df["Completed_By_Name"])
    df["Hours_OK"] = df["Required_Hours"].apply(lambda rq: completed_hours >= rq)
    df["Year_OK"] = df["Target_Year"].apply(lambda y: year_ok(y, student_year))

    df = df[df["Prereq_OK"] & df["Hours_OK"] & df["Not_Completed"] & df["Year_OK"]].copy()

    def parse_concurrent(txt):
        toks = split_tokens(txt)
        return [normalize_code(t) for t in toks]

    df["Concurrent_List"] = df["المواد المتزامنة"].apply(parse_concurrent)

    def normalize_type(t):
        t = str(t or "").strip()
        t = re.sub(r"\bال","",t)
        t = re.sub(r"\s+"," ",t)
        return t

    priority_rules = [
        (re.compile(r"متطلب كلية اجباري"), 1),
        (re.compile(r"متطلب تخصص اجباري"), 2),
        (re.compile(r"متطلبات تخصص مساندة|مساندة"), 3),
        (re.compile(r"متطلب جامعة اجباري"), 4),
        (re.compile(r"تخصص اختياري"), 5),
        (re.compile(r"جامعة اختياري"), 6),
    ]

    def get_priority(nt):
        nt = normalize_type(nt)
        for pat, p in priority_rules:
            if pat.search(nt):
                return p
        return 99

    df["نوع المادة"] = df["نوع المادة"].apply(normalize_type)
    df["Priority"] = df["نوع المادة"].apply(get_priority)

    df["Norm_Code"] = df["Norm_Code"].astype(str)

    def make_group_key(row):
        main = row["Norm_Code"]
        conc = [c for c in row["Concurrent_List"] if str(c).strip().upper() != "NAN"]
        group = {main} | set(conc)
        valid = set(df["Norm_Code"])
        group = group & valid
        return "|".join(sorted(group))

    df["Group_Key"] = df.apply(make_group_key, axis=1)

    blocks = []
    for key, group in df.groupby("Group_Key"):
        total_hours = group["عدد الساعات"].astype(float).sum()
        block_year = group["Target_Year"].min()
        block_priority = group["Priority"].min()
        samples = group.to_dict("records")

        blocks.append({
            "key": key,
            "courses": samples,
            "hours": total_hours,
            "year": block_year,
            "priority": block_priority
        })

    blocks = sorted(blocks, key=lambda b: (b["year"], b["priority"]))

    chosen_uni_opt = 0
    chosen_major_opt = 0
    max_uni_opt = 2
    max_major_opt = 4

    final_courses = []
    total = 0.0

    for block in blocks:
        block_type_set = {c["نوع المادة"] for c in block["courses"]}

        if any("جامعة اختياري" in t for t in block_type_set):
            if chosen_uni_opt >= max_uni_opt:
                continue

        if any("تخصص اختياري" in t for t in block_type_set):
            if chosen_major_opt >= max_major_opt:
                continue

        if total + block["hours"] > max_hours:
            continue

        for course in block["courses"]:
            final_courses.append(course)

        total += block["hours"]

        if any("جامعة اختياري" in t for t in block_type_set):
            chosen_uni_opt += 1

        if any("تخصص اختياري" in t for t in block_type_set):
            chosen_major_opt += 1

    if not final_courses:
        return None

    final_df = pd.DataFrame(final_courses)

    html = final_df[
        ["رقم المادة", "اسم المادة", "عدد الساعات", "نوع المادة", "Target_Year"]
    ].to_html(index=False, classes="table table-bordered table-striped", justify="center")

    def make_reason(row):
        return f"""
تم اختيار {row['اسم المادة']} ({row['رقم المادة']})
لأنها من السنة {row['Target_Year']}
ولأنها الأقرب لسنة الطالب
ولأن الطالب مستوفي كل شروط تنزيل المادة.
"""

    meta = []
    for _, r in final_df.iterrows():
        meta.append({
            "code": r["رقم المادة"],
            "name": r["اسم المادة"],
            "reason_prompt": make_reason(r)
        })

    return {
        "table_html": html,
        "meta": meta,
        "total_hours": total,
        "df": final_df
    }

def suggest_for_student(student_id, password, max_hours=18):
    return full_schedule_engine(student_id, password, max_hours)

# ===========================================
# 🌐 Flask
# ===========================================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        if request.is_json:
            data = request.get_json()
            phase = data.get("phase")
        else:
            data = request.form.to_dict()
            phase = data.get("phase")

        # 1) طلب قائمة التخصصات
        if phase == "get_majors":
            majors = sorted(
                desc_df.get("اسم التخصص", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            return jsonify({
                "type": "majors_list",
                "majors": majors
            })

        # 2) طلب اقتراح جدول
        if phase == "creds":
            sid = (data.get("student_id","") or "").strip()
            pwd = (data.get("password","") or "").strip()
            hours = int(data.get("max_hours") or 18)

            result = suggest_for_student(sid, pwd, hours)
            if result is None or result["df"].empty:
                return jsonify({"type":"bot_text","message":"⚠️ لم أجد مواد مناسبة أو بيانات خاطئة."})

            # ✅ مهم: لا تمرّر HTML للـ LLM حتى لا يخرب التنسيق
            return jsonify({
                "type": "bot_html",
                "message": result["table_html"],
                "total_hours": result["total_hours"],
                "meta": result["meta"]
            })

        # 3) أسئلة نصية عامة
        text = (data.get("text","") or "").strip()
        major = (data.get("major","") or "").strip()

        if not text:
            return jsonify({"type":"bot_text","message":"⚠️ الرجاء كتابة استفسار أولاً."})

        # أولوية خاصة: سؤال عن جدول
        if any(w in text for w in ["جدول","اقتراح","schedule","خطتي","اقترح","اقترحلي","الفصل","القادم","انزل"]):
            return jsonify({
                "type":"ask_credentials",
                "message":"📘 من فضلك أدخل رقمك الجامعي وكلمة المرور وعدد الساعات المطلوبة:"
            })

        # لازم يختار تخصص قبل أسئلة المواد
        if not major:
            majors = sorted(
                desc_df.get("اسم التخصص", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            return jsonify({
                "type": "need_major",
                "message": "🎓 من فضلك اختر تخصصك أولاً قبل طرح الأسئلة عن المواد.",
                "majors": majors
            })

        # 4) كشف نية السؤال (NLP)
        intent, score = detect_intent(text)

        # A) لو تحية → LLM فقط
        if intent == "greeting":
            reply = reply_greeting(text)
            return jsonify({"type":"bot_text","message": reply})

        # B) لو سؤال عن مادة → نبحث في التخصص + ذاكرة متابعة
        if intent == "course_query":
          session_id = get_session_id()
          dbg("REQ", sid=session_id, text=text, major=major)
          row, corrected, best_score = search_course_for_major(text, major, top_k=3)
          memory = recall_course(session_id)
          direct = None
          followup_triggers = [
              "المتطلب", "متطلب",
              "المتزامن", "متزامنة", "تنزل مع",
              "كم", "عدد الساعات",
              "وصف", "نوع", "شرح", "ملخص",
              "متطلبها", "المتزامنة معها", " ساعاتها"
          ]
          df_major = get_major_index(major)["df"]
          explicit_course = mentions_course_name(text, df_major)
          is_followup = (
              memory is not None
              and not explicit_course   # ← هذا هو الحل
              and any(t in text for t in followup_triggers)
            )

          if memory and is_followup:
            row = get_row_by_code_for_major(memory["course_code"], major)
            corrected = memory["course_name"]
            if row is None:
              return jsonify({
                  "type": "bot_text",
                  "message": "⚠️ لم أتمكن من استرجاع بيانات المادة. يرجى ذكر اسمها مرة أخرى."
                  })

          else:
            # بحث عادي فقط إذا لم يكن متابعة
            row, corrected, best_score = search_course_for_major(text, major, top_k=3)
            if row is None:
              return jsonify({
                  "type": "bot_text",
                  "message": "يرجى تحديد اسم المادة بشكل واضح (مثال: ما هو المتطلب السابق لمادة قواعد البيانات؟)."
                  })
            remember_course(session_id, major, row)
          direct = handle_direct_question(text, row)
          if direct:
            return jsonify({"type":"bot_text","message": direct})
          reply = describe_course_from_row(corrected, row)
          return jsonify({"type":"bot_text","message": reply})


        # C) أي شيء آخر
        return jsonify({
            "type":"bot_text",
            "message":"أنا فقط أجيب عن الأسئلة المتعلقة بالمواد والمتطلبات والخطة الدراسية."
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"type":"bot_text","message":f"⚠️ خطأ في الخادم: {e}"})

# ===========================================
# 🚀 تشغيل التطبيق
# ===========================================
if __name__ == "__main__":
    print("🚀 بدء تشغيل المساعد الجامعي الذكي...")
    app.run(host="0.0.0.0", port=5040)
