# src/fusion.py

from typing import List, Dict, Any, Tuple


def _span_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Ikki oraliq (start, end) uchun IoU (intersection over union) hisoblaydi.
    """
    (s1, e1), (s2, e2) = a, b
    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter == 0:
        return 0.0
    union = max(e1, e2) - min(s1, s2)
    if union == 0:
        return 0.0
    return inter / union


def merge_entities(
    ents_ml: List[Dict[str, Any]],
    ents_rules: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    ML-NER va rules-NER natijalarini birlashtiradi.

    Qoidalar:
    - Agar ML va rules bir xil TYPE uchun kuchli overlap qilsa (IoU >= threshold):
        * uzunroq span tanlanadi
        * `source = "ml+rules"`
    - Agar faqat ML topsa: `source = "ml"`
    - Agar faqat rules topsa: `source = "rules"`
    """

    merged: List[Dict[str, Any]] = []

    # Avval ML entitylarni qo‘shamiz
    for ent in ents_ml:
        ent_copy = dict(ent)
        # Agar source oldindan bo‘lsa ham, "ml" deb normalizatsiya qilamiz
        ent_copy["source"] = "ml"
        merged.append(ent_copy)

    # Keyin rules entitylarni qo‘shamiz
    for r_ent in ents_rules:
        r_start, r_end = r_ent["start"], r_ent["end"]
        r_type = r_ent.get("type")

        best_iou = 0.0
        best_idx = -1

        # ML bilan overlapni tekshirish
        for i, m_ent in enumerate(merged):
            if m_ent.get("type") != r_type:
                continue

            m_start, m_end = m_ent["start"], m_ent["end"]
            iou = _span_iou((r_start, r_end), (m_start, m_end))

            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou >= iou_threshold:
            # Bir xil type va yetarli overlap – bitta span tanlaymiz
            m_ent = merged[best_idx]
            # Uzunroq span qolsin
            m_len = m_ent["end"] - m_ent["start"]
            r_len = r_end - r_start

            if r_len > m_len:
                # rules spanini ustun qo‘yamiz, lekin source = "ml+rules"
                new_ent = dict(r_ent)
                new_ent["source"] = "ml+rules"
                merged[best_idx] = new_ent
            else:
                # ML spani qolsin, faqat source'ni yangilaymiz
                m_ent["source"] = "ml+rules"
        else:
            # ML bilan kuchli overlap yo‘q – rules entityni alohida qo‘shamiz
            new_ent = dict(r_ent)
            new_ent["source"] = "rules"
            merged.append(new_ent)

    # start bo‘yicha sort qilib qaytaramiz
    merged = sorted(merged, key=lambda x: x["start"])
    return merged


def fuse_sentiment(
    ml_label: str,
    ml_score: float,
    lex_label: str,
    lex_score: float,
) -> str:
    """
    ML va leksikon sentiment natijalarini birlashtirish.

    Oddiy qoidalar:
    - Agar ML ishonchi juda yuqori (>= 0.99) va label neutral emas -> ML ustun.
    - Aks holda, agar leksikon bal absolute qiymati katta bo‘lsa -> leksikon ustun.
    - Agar farq katta bo‘lmasa -> ML natijasini qoldiramiz.
    """

    # 1) Juda ishonchli ML natija – ustunlik
    if ml_label != "neutral" and ml_score >= 0.99:
        return ml_label

    # 2) Leksikon signal kuchli bo‘lsa (masalan |score| >= 2)
    if abs(lex_score) >= 2.0:
        return lex_label

    # 3) Aks holda ML'ni qoldiramiz
    return ml_label
