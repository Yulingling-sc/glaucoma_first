"""年龄清洗：与训练一致（缺失/异常用中位数 58，正常值 clip 到 [10,100]）。"""
AGE_MEDIAN = 58.0


def clean_age(v):
    try:
        a = float(v)
    except (ValueError, TypeError):
        return AGE_MEDIAN
    if a < 1 or a > 110:
        return AGE_MEDIAN
    return min(max(a, 10.0), 100.0)