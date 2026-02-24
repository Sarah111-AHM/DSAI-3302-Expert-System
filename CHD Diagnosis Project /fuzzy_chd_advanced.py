import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# =============================================================================
# PART 1: ENHANCED MEMBERSHIP FUNCTIONS (with new factors)
# =============================================================================

def triangular(x, a, b, c):
    """Triangular membership function"""
    if x <= a or x >= c:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    return 0

def trapezoidal(x, a, b, c, d):
    """Trapezoidal membership function"""
    if x <= a or x >= d:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1
    elif c < x < d:
        return (d - x) / (d - c)
    return 0

# =============================================================================
# NEW FACTOR 1: AGE Membership Functions
# =============================================================================

def fuzzify_age(age):
    """
    Age categories (medical guidelines):
        Young: < 30
        Middle-aged: 30-55
        Old: > 50
    """
    return {
        'Young': trapezoidal(age, 0, 0, 25, 35),
        'Middle': trapezoidal(age, 30, 40, 50, 60),
        'Old': trapezoidal(age, 50, 65, 100, 120)
    }

# =============================================================================
# NEW FACTOR 2: SMOKING Status
# =============================================================================

def fuzzify_smoking(packs_per_day):
    """
    Smoking intensity (packs per day):
        None: 0 packs
        Light: 0-0.5 packs
        Heavy: >0.5 packs
    """
    return {
        'None': trapezoidal(packs_per_day, -0.2, 0, 0, 0.2),
        'Light': triangular(packs_per_day, 0, 0.3, 0.7),
        'Heavy': trapezoidal(packs_per_day, 0.5, 1, 3, 4)
    }

# =============================================================================
# NEW FACTOR 3: DIABETES Status
# =============================================================================

def fuzzify_diabetes(glucose_level):
    """
    Diabetes based on blood glucose (mg/dL):
        No: < 100
        Pre-diabetic: 100-125
        Diabetic: > 125
    """
    return {
        'No': trapezoidal(glucose_level, 0, 0, 90, 110),
        'Pre': triangular(glucose_level, 100, 112.5, 125),
        'Yes': trapezoidal(glucose_level, 120, 150, 300, 400)
    }

# =============================================================================
# ORIGINAL FACTORS (from Phase 8)
# =============================================================================

def fuzzify_bp(bp):
    return {
        'Low': triangular(bp, 100, 115, 130),
        'Medium': triangular(bp, 120, 145, 170),
        'High': triangular(bp, 160, 180, 200)
    }

def fuzzify_chol(chol):
    return {
        'Low': trapezoidal(chol, 100, 120, 180, 200),
        'High': trapezoidal(chol, 180, 200, 260, 280)
    }

def fuzzify_hr(hr):
    return {
        'Slow': triangular(hr, 50, 65, 80),
        'Moderate': triangular(hr, 70, 85, 100),
        'Fast': triangular(hr, 90, 145, 200)
    }

# Output centers
HEALTHY_CENTER = 0.75
MIDDLE_CENTER = 2.0
SICK_CENTER = 3.25
