import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load feedback data
df = pd.read_csv("feedback.csv")

# Aggregate feedback by course
course_stats = df.groupby('course_id').agg({
    'rating_clarity': 'mean',
    'rating_workload': 'mean',
    'rating_interaction': 'mean',
    'rating_attendance_strictness': 'mean',
    'rating_assignments': 'mean',
    'overall_rating': 'mean',
    'course_organization': 'mean'
}).reset_index()

# Normalize ratings to 0-1 scale
scaler = MinMaxScaler(feature_range=(0, 1))
rating_cols = ['rating_clarity', 'rating_workload', 'rating_interaction', 
               'rating_attendance_strictness', 'rating_assignments', 
               'course_organization', 'overall_rating']
course_stats[rating_cols] = scaler.fit_transform(course_stats[rating_cols])

# Create latent factors
course_stats['latent_computational_math_heavy'] = (
    course_stats['rating_workload'] * 0.4 + 
    course_stats['rating_assignments'] * 0.4 + 
    (1 - course_stats['rating_clarity']) * 0.2  # lower clarity = more computational
)

course_stats['latent_theory_heavy'] = (
    course_stats['course_organization'] * 0.35 +
    course_stats['rating_clarity'] * 0.35 +
    (1 - course_stats['rating_assignments']) * 0.3  # fewer assignments = more theory
)

course_stats['latent_difficulty_rigor'] = (
    course_stats['rating_workload'] * 0.5 +
    course_stats['rating_assignments'] * 0.5
)

course_stats['latent_practicality'] = (
    course_stats['rating_assignments'] * 0.4 +
    course_stats['rating_interaction'] * 0.3 +
    course_stats['rating_workload'] * 0.3
)

course_stats['latent_ease'] = (
    (1 - course_stats['rating_workload']) * 0.4 +
    course_stats['rating_clarity'] * 0.4 +
    course_stats['course_organization'] * 0.2
)

# Save all course stats + latent factors to CSV
course_stats.to_csv("course_latent_factors.csv", index=False)
latent_cols = ['latent_computational_math_heavy', 'latent_theory_heavy', 
               'latent_difficulty_rigor', 'latent_practicality', 'latent_ease']

print("✓ Course latent factors generated!")
print(f"Courses: {len(course_stats)}")
print("\nSample (first 5 courses):")
print(course_stats.head())
print("\nLatent Factor Statistics:")
for col in latent_cols:
    print(f"{col}: min={course_stats[col].min():.3f}, max={course_stats[col].max():.3f}, mean={course_stats[col].mean():.3f}")
